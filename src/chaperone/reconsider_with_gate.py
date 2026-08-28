"""Feed each deterministic-gate contradiction (see deterministic_gate.py)
back to the model so it genuinely reconsiders the verdict, instead of a rule
guessing the "correct" replacement label itself. Only processes candidates
deterministic_gate.py actually flagged — most candidates are left untouched.

Usage:
    python -m chaperone.reconsider_with_gate
"""
import asyncio
import csv
import sys
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv

from anthropic import AsyncAnthropic  # noqa: E402
from .verdict_tool import make_record_verdict_tool  # noqa: E402
from .paths import PROJECT_ROOT as ROOT  # noqa: E402
from .deterministic_gate import (  # noqa: E402
    compute_gate_facts, find_contradictions, load_tool_calls, load_fold_results, gather_fold_results,
)

# claude-fable-5's bio-safety classifier silently REFUSED reconsideration
# calls whose contradiction text discusses glycan-mediated binding
# mechanisms (stop_reason="refusal", empty tool input {}) — the same
# classifier behavior already found and worked around in
# design_validation.py, now also hit here. The refusal was going
# UNDETECTED: merging the empty {} onto the original row silently returned
# the row unchanged, masquerading as "reconsideration decided nothing
# needed to change" for potentially many candidates in a batch. Fixed two
# ways: detect refusal explicitly (see reconsider_one) AND run this script
# on sonnet-5, which doesn't refuse this content, matching the established
# mitigation used elsewhere in this project.
MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """\
You previously triaged a candidate protein-protein interaction (PPI) \
predicted by AlphaFold3 and produced a verdict. A separate, code-based \
check (not another LLM guess) has now compared your verdict against the \
raw structured tool outputs from your own original evidence-gathering \
(STRING's per-channel scores, CellPhoneDB's curated hits, HPA's protein \
class/secretome/expression data, and any real AF3 fold's structural \
analysis) and found a specific, concrete contradiction — stated below.

Reconsider the verdict genuinely in light of this specific contradiction. \
You are not being asked to redo the whole triage from scratch — engage \
directly with the contradiction: either the verdict should change, or \
explain (briefly, in the rationale) why the flagged evidence doesn't \
actually override your original reasoning (e.g. a fold-based structural \
concern doesn't automatically mean the whole interaction is IMPLAUSIBLE if \
other strong evidence like real observed co-localization still supports \
it — but say so explicitly rather than silently repeating the old verdict).

Guidance specific to the two most common contradiction types:
- A STRING hit's combined_score can be driven entirely by textmining/
  coexpression channels with zero database/experiments backing — that's a
  literature co-mention signal, not proof of a real known interaction, and
  should NOT push a verdict toward ALREADY_KNOWN or reduce novelty credit.
  Separately: even a STRING hit with real database/experiments channel
  backing is NOT sufficient for ALREADY_KNOWN on its own — the "database"
  channel can reflect curated pathway/interactome membership rather than a
  published paper on this specific pair. The known_interaction_strength
  check below only reports "confirmed" when it ALSO found at least one real
  pubmed_cooccurrence hit for this pair — so if this contradiction says
  "confirmed", both a strong STRING channel AND a literature trace exist,
  and that combination IS real evidence of a known interaction. A pair
  found ONLY in STRING with zero literature trace stays novel.
- HPA showing one protein as intracellular-only (no membrane/secreted
  annotation) and the other as surface/secreted, WITH zero shared tissue or
  single-cell-type expression, means there's no real evidence in the data
  given for how these two would ever physically meet — a strong argument for
  IMPLAUSIBLE (proteins can moonlight, but that needs to be a substantiated,
  not just hypothetical, exception).
- If either protein is a galectin (LGALS1/2/3/4/7/8/9/12/13/14): its
  physiological binding is fundamentally glycan-mediated. A real AF3 fold
  showing "no annotated PTM/glycosylation site on the specific interface
  residues modeled" does NOT clear this concern — AF3 never represents
  glycans at all, so the true binding mode may not be the modeled bare-
  protein contact in the first place, regardless of which residues happen
  to touch. This should generally push toward LIKELY_ARTIFACT_PTM unless
  you have a specific, substantiated reason the glycan-mediated mechanism
  doesn't apply to this particular pair.

Your new rationale text MUST NOT be a verbatim or near-verbatim copy of the
original rationale given to you — if you find yourself about to write the
same sentences, stop and rewrite it to explicitly name and address each
contradiction listed above, sentence by sentence if needed. A reply that
just repeats the original text will be rejected and you will be asked
again with a stronger prompt, wasting a turn — so do the real work now.

You MUST end by calling record_verdict exactly once, with a rationale that
explicitly engages with EVERY contradiction given (not just restating the
original rationale)."""


def make_client() -> AsyncAnthropic:
    # Same proxy workaround as design_validation.py.
    return AsyncAnthropic(http_client=httpx.AsyncClient(trust_env=False))


async def reconsider_one(client: AsyncAnthropic, row: dict, facts: dict, contradictions: list) -> dict:
    sink: dict = {}
    record_verdict = make_record_verdict_tool(sink)
    contradictions_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(contradictions))
    user_prompt = (
        f"Candidate: {row['protein_a']} <-> {row['protein_b']}\n"
        f"Original verdict: {row['verdict']} ({row['confidence']} confidence)\n"
        f"Original rationale: {row['rationale']}\n"
        f"Original known_interaction_evidence: {row.get('known_interaction_evidence')}\n"
        f"Original ptm_glycosylation_evidence: {row.get('ptm_glycosylation_evidence')}\n\n"
        f"DETERMINISTIC CHECK FOUND {len(contradictions)} CONTRADICTION(S):\n{contradictions_text}\n\n"
        f"(Full evidence facts: known_interaction_strength={facts['known_interaction_strength']}, "
        f"hpa_plausibility={facts['hpa_plausibility']}, fold_evidence_status={facts['fold_evidence_status']})\n\n"
        "Reconsider the verdict given ALL of the contradictions above and record it — engage with each one, "
        "not just the first."
    )

    response = None
    for attempt in range(8):
        try:
            response = await client.beta.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=[record_verdict.to_dict()],
                tool_choice={"type": "tool", "name": "record_verdict"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            break
        except (anthropic.APIConnectionError, anthropic.InternalServerError):
            if attempt == 7:
                raise
            await asyncio.sleep(min(20, 2 * (attempt + 1)))

    # A refusal produces an empty tool_use input ({}) — merging that onto
    # the original row silently returns it unchanged, indistinguishable
    # from "genuinely reconsidered, decided to keep it." That's exactly
    # what happened when this ran on fable-5 (see the MODEL comment above):
    # a batch of "reconsidered" candidates were actually all silent
    # refusals. Fail loudly instead of masking it.
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Model refused to reconsider {row['protein_a']}/{row['protein_b']} (stop_reason=refusal)")

    tool_block = next((b for b in response.content if b.type == "tool_use" and b.name == "record_verdict"), None)
    if tool_block is None or "verdict" not in tool_block.input:
        raise RuntimeError(f"No valid record_verdict tool call for {row['protein_a']}/{row['protein_b']}")

    new_rationale = (tool_block.input.get("rationale") or "").strip()
    original_rationale = (row.get("rationale") or "").strip()
    if new_rationale == original_rationale:
        raise RuntimeError(
            f"{row['protein_a']}/{row['protein_b']}: model returned a rationale byte-identical to the "
            f"original — not genuine reconsideration, treating as a failure rather than silently accepting it."
        )

    return {**row, **tool_block.input}


async def run_gate_reconsideration(csv_path: Path, log_dir: Path, fold_summary_path: Path, concurrency: int = 4) -> int:
    """Runs the deterministic gate over every row in csv_path, reconsiders
    only the flagged candidates, and rewrites csv_path in place. Returns the
    number of candidates flagged/reconsidered."""
    client = make_client()

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    fold_by_pair = load_fold_results(fold_summary_path)

    targets = []
    for row in rows:
        tool_calls = load_tool_calls(row, log_dir)
        fold_results = gather_fold_results(row, tool_calls, fold_by_pair)
        facts = compute_gate_facts(row, tool_calls, fold_results)
        contradictions = find_contradictions(row, facts)
        if contradictions:
            targets.append((row, facts, contradictions))
    print(f"Reconsidering {len(targets)} flagged candidates...")

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(row, facts, contradictions):
        async with semaphore:
            try:
                updated = await reconsider_one(client, row, facts, contradictions)
                print(f"  {row['protein_a']}/{row['protein_b']}: {row['verdict']} -> {updated['verdict']}")
                return updated
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {row['protein_a']}/{row['protein_b']}: {exc}")
                return row

    updated_rows = await asyncio.gather(*(worker(row, facts, c) for row, facts, c in targets))
    updated_by_pair = {f"{r['protein_a']}/{r['protein_b']}": r for r in updated_rows}
    final_rows = [updated_by_pair.get(f"{r['protein_a']}/{r['protein_b']}", r) for r in rows]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in final_rows:
            out = {k: r.get(k) for k in fieldnames}
            if isinstance(out.get("other_subunits"), list):
                out["other_subunits"] = ";".join(out["other_subunits"])
            writer.writerow(out)

    print(f"\nUpdated {csv_path}")
    return len(targets)


async def main():
    load_dotenv(ROOT / ".env")
    await run_gate_reconsideration(
        ROOT / "data" / "verdicts_full.csv",
        ROOT / "log",
        ROOT / "fold_runs" / "followups" / "_summary.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
