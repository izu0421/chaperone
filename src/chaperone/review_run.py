"""Review agent: audits a completed pipeline run for quality issues instead
of trusting it blindly. Two layers:

1. Deterministic checks (fast, free, no LLM) over verdicts.csv +
   validation_strategies.json + the log/*.json tool-call ledgers: malformed
   enum fields, empty/truncated-looking text fields, candidates with no
   traceable tool-call ledger, and how selective the "recommended for
   validation" tier actually is.
2. An LLM audit pass (claude-sonnet-5 — same reasoning as design_validation.py:
   fable-5's bio-safety classifier refuses this kind of scrutiny of wet-lab
   protocol content) over each recommended candidate: does the chosen assay
   method actually fit the stated subcellular evidence, do the target tissues
   appear anywhere in the cited evidence, is the rationale specific or
   generic boilerplate.

Usage:
    python -m chaperone.review_run data/verdicts_full.csv log \
        --strategies data/validation_strategies.json --out data/review_findings.json
"""
import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv


from anthropic import AsyncAnthropic, beta_async_tool  # noqa: E402
from .build_report import classify_validation  # noqa: E402
from .design_validation import summarize_hpa  # noqa: E402
from .sources.hpa_client import fetch_gene_profile  # noqa: E402
from .strategy_utils import normalize_str_list  # noqa: E402
from .paths import PROJECT_ROOT as ROOT  # noqa: E402

MODEL = "claude-sonnet-5"
VALID_METHODS = {"PLA", "stimulation_assay", "co_ip", "other"}
FREE_TEXT_FIELDS = ["method_rationale", "tissue_rationale", "protocol_notes", "controls", "caveats"]


def make_client() -> AsyncAnthropic:
    # Same proxy workaround as design_validation.py — this request shape
    # (long system prompt + wide tool schema) gets dropped by the local dev
    # proxy; api.anthropic.com is directly reachable from this host.
    return AsyncAnthropic(http_client=httpx.AsyncClient(trust_env=False))


def looks_truncated(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 15:
        return True
    return text[-1] not in ".!?\"')"


def deterministic_checks(rows: list[dict], strategies: list[dict], log_dir: Path) -> list[dict]:
    findings = []

    n_recommended = sum(1 for r in rows if classify_validation(r, []))
    ratio = n_recommended / len(rows) if rows else 0
    findings.append({
        "severity": "info",
        "pair": None,
        "check": "recommended_tier_ratio",
        "detail": f"{n_recommended}/{len(rows)} candidates ({ratio:.0%}) are tiered 'recommended for validation'. "
                  f"If this is most of the dataset, the tiers are excluding little and may not feel like a real "
                  f"prioritization — worth a human sanity check on whether the tier boundaries are too permissive.",
    })

    for row in rows:
        pair = f"{row['protein_a']}/{row['protein_b']}"
        log_path = log_dir / f"{row['protein_a']}__{row['protein_b']}.json"
        n_calls = 0
        if log_path.exists():
            try:
                n_calls = len(json.loads(log_path.read_text()).get("_tool_calls", []))
            except json.JSONDecodeError:
                pass
        if n_calls == 0:
            findings.append({
                "severity": "error",
                "pair": pair,
                "check": "missing_tool_call_ledger",
                "detail": f"No tool-call ledger found at {log_path} (or it's empty) — this verdict currently isn't "
                          f"traceable to raw evidence. Usually means the wrong log directory was passed to "
                          f"build_report.py, not that the agent made no calls.",
            })

        if not (row.get("rationale") or "").strip():
            findings.append({"severity": "error", "pair": pair, "check": "empty_rationale", "detail": "Rationale field is empty."})

    for s in strategies:
        pair = f"{s['protein_a']}/{s['protein_b']}"
        method = s.get("method", "")
        if method not in VALID_METHODS:
            findings.append({
                "severity": "error",
                "pair": pair,
                "check": "non_enum_method",
                "detail": f"method={method!r} is not one of {sorted(VALID_METHODS)} — likely a full sentence stuffed "
                          f"into a field meant to be a short label, which reads as inconsistent/cut-off wherever it's "
                          f"rendered as a table cell.",
            })
        if not s.get("target_tissues"):
            findings.append({"severity": "error", "pair": pair, "check": "empty_target_tissues", "detail": "target_tissues is empty."})
        for field in ("target_tissues", "cell_types"):
            if s.get(field) is not None and not isinstance(s.get(field), list):
                findings.append({
                    "severity": "error",
                    "pair": pair,
                    "check": "non_list_field",
                    "detail": f"{field} is a {type(s.get(field)).__name__}, not a list — likely got "
                              f"character-by-character corrupted wherever code does ', '.join({field}). "
                              f"Run normalize_str_list from strategy_utils.py over it.",
                })
        for field in FREE_TEXT_FIELDS:
            if looks_truncated(s.get(field, "")):
                findings.append({
                    "severity": "warning",
                    "pair": pair,
                    "check": "possibly_truncated_field",
                    "detail": f"{field} is very short or doesn't end in sentence-ending punctuation: {s.get(field, '')!r}",
                })

    return findings


def make_audit_tool(sink: dict):
    @beta_async_tool
    async def record_review_findings(consistent: bool, issues: list[str]) -> str:
        """Record the review outcome for this candidate.

        Args:
            consistent: true if the chosen method, target tissues, and
                rationale are genuinely consistent with the evidence given —
                false if you found a real problem.
            issues: list of specific problems found (empty if consistent is
                true). E.g. "method is PLA but the two proteins are in
                different compartments per the evidence given, with no
                paracrine/co-culture justification", or "target tissue X is
                not mentioned anywhere in the evidence given".
        """
        sink["result"] = {"consistent": consistent, "issues": issues}
        return "recorded"

    return record_review_findings


AUDIT_SYSTEM_PROMPT = """\
You are auditing a completed protein-protein-interaction triage decision and \
its proposed wet-lab validation strategy for INTERNAL CONSISTENCY — not \
redoing the triage from scratch. You're given: the verdict + rationale + \
evidence the triage agent cited, and the validation strategy (method, target \
tissues, cell types) designed afterward.

Check specifically:
1. Does the chosen method (PLA needs same-cell/same-compartment proximity;
   stimulation_assay fits a secreted-ligand + membrane-receptor pair; co_ip
   is a lysate-based fallback) actually fit the subcellular/secretome
   evidence given? Flag a mismatch (e.g. PLA chosen for two proteins in
   clearly different compartments with no paracrine/co-culture justification
   given).
2. Do the named target tissues actually appear anywhere in the evidence
   provided? Flag any tissue that seems invented.
3. Is the rationale specific to these two proteins, or could it be copy-
   pasted boilerplate that doesn't actually engage with the evidence?

Do not flag stylistic preferences or second-guess the underlying triage
verdict itself — only flag genuine internal inconsistency between the
evidence given and the strategy built on top of it. You MUST end by calling
record_review_findings exactly once."""


async def audit_one(client: AsyncAnthropic, row: dict, strategy: dict) -> dict:
    sink = {}
    tool = make_audit_tool(sink)
    # Tissue/cell-type claims in the strategy are grounded in a fresh HPA
    # lookup done by design_validation.py at design time, NOT in the triage
    # evidence fields (which cover PTM/cooccurrence/known-interaction only,
    # never tissue nTPM numbers) — check tissue claims against the same HPA
    # data design_validation.py actually saw, not against the wrong reference.
    profile_a = fetch_gene_profile(row["protein_a"])
    profile_b = fetch_gene_profile(row["protein_b"])
    user_prompt = (
        f"Candidate: {row['protein_a']} <-> {row['protein_b']}\n"
        f"Verdict: {row['verdict']} ({row['confidence']} confidence)\n"
        f"Rationale: {row['rationale']}\n"
        f"PTM/glycosylation evidence: {row.get('ptm_glycosylation_evidence')}\n"
        f"PubMed co-occurrence evidence: {row.get('cooccurrence_evidence')}\n"
        f"Known-interaction evidence (STRING/CellPhoneDB): {row.get('known_interaction_evidence')}\n\n"
        f"HPA data available at strategy-design time (this is the ground truth for tissue/cell-type claims):\n"
        f"{summarize_hpa(row['protein_a'], profile_a)}\n{summarize_hpa(row['protein_b'], profile_b)}\n\n"
        f"Validation strategy designed:\n"
        f"Method: {strategy.get('method')} ({strategy.get('method_variant')})\n"
        f"Method rationale: {strategy.get('method_rationale')}\n"
        f"Target tissues: {normalize_str_list(strategy.get('target_tissues'))}\n"
        f"Tissue rationale: {strategy.get('tissue_rationale')}\n\n"
        "Audit this for internal consistency. Judge the tissue/cell-type claims against the HPA data given "
        "above (the actual source), not against the PTM/cooccurrence/known-interaction evidence, which was "
        "never expected to mention tissues."
    )
    response = None
    for attempt in range(6):
        try:
            response = await client.beta.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=AUDIT_SYSTEM_PROMPT,
                tools=[tool.to_dict()],
                tool_choice={"type": "tool", "name": "record_review_findings"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            break
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            await asyncio.sleep(min(20, 2 * (attempt + 1)))
    # A refusal (stop_reason="refusal", empty tool input {}) must NOT be
    # silently treated as "consistent: True" (the default llm_audit's caller
    # falls back to) — that masked a real bug in reconsider_with_gate.py
    # this same session, where an entire batch's refusals were
    # indistinguishable from "genuinely reconsidered, nothing to flag."
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Model refused to audit {row['protein_a']}/{row['protein_b']} (stop_reason=refusal)")
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_review_findings":
            if "consistent" not in block.input:
                raise RuntimeError(f"record_review_findings call missing 'consistent' for {row['protein_a']}/{row['protein_b']}")
            return {"protein_a": row["protein_a"], "protein_b": row["protein_b"], **block.input}
    raise RuntimeError(f"No record_review_findings call for {row['protein_a']}/{row['protein_b']}")


async def llm_audit(rows: list[dict], strategies: list[dict], concurrency: int = 6) -> list[dict]:
    by_pair = {(r["protein_a"], r["protein_b"]): r for r in rows}
    client = make_client()
    semaphore = asyncio.Semaphore(concurrency)
    findings = []

    async def worker(s):
        row = by_pair.get((s["protein_a"], s["protein_b"]))
        if not row:
            return
        async with semaphore:
            try:
                result = await audit_one(client, row, s)
            except Exception as exc:  # noqa: BLE001
                print(f"  AUDIT FAILED {s['protein_a']}/{s['protein_b']}: {exc}")
                return
            if not result.get("consistent", True):
                pair = f"{s['protein_a']}/{s['protein_b']}"
                for issue in result.get("issues", []):
                    findings.append({"severity": "warning", "pair": pair, "check": "llm_audit", "detail": issue})
                print(f"  {pair}: {len(result.get('issues', []))} issue(s) flagged")

    await asyncio.gather(*(worker(s) for s in strategies))
    return findings


async def main():
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("log_dir")
    ap.add_argument("--strategies", default=str(ROOT / "data" / "validation_strategies.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "review_findings.json"))
    ap.add_argument("--skip-llm-audit", action="store_true", help="Only run the free deterministic checks")
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    strategies_path = Path(args.strategies)
    strategies = json.loads(strategies_path.read_text()) if strategies_path.exists() else []

    findings = deterministic_checks(rows, strategies, Path(args.log_dir))
    print(f"Deterministic checks: {len(findings)} findings")

    if not args.skip_llm_audit and strategies:
        print(f"Running LLM consistency audit over {len(strategies)} recommended candidates...")
        findings += await llm_audit(rows, strategies)

    Path(args.out).write_text(json.dumps(findings, indent=2))
    by_sev = {}
    for f in findings:
        by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
    print(f"\nWrote {len(findings)} findings to {args.out}: {by_sev}")


if __name__ == "__main__":
    asyncio.run(main())
