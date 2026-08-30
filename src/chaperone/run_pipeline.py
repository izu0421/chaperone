"""Batch entrypoint: run the agent over every candidate in a CSV and write a
ranked verdict report. Usage:

    chaperone path/to/candidates.csv [--concurrency 4]

Requires ANTHROPIC_API_KEY in the environment (loaded from .env if present).
"""
import argparse
import asyncio
import csv
import itertools
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv


from .agent import DEFAULT_FOLD_GPUS, evaluate_candidate, hpa_mcp_session  # noqa: E402
from .ingest import load_candidates  # noqa: E402
from .design_validation import run_validation_design  # noqa: E402
from .build_report import build_report  # noqa: E402
from .review_run import deterministic_checks, llm_audit  # noqa: E402
from .reconsider_with_gate import run_gate_reconsideration  # noqa: E402
from .reconsider_with_fold import run_fold_reconsideration  # noqa: E402
from .anthropic_client import make_client  # noqa: E402
from .paths import PROJECT_ROOT as ROOT  # noqa: E402

LOG_DIR = ROOT / "log"
DATA_DIR = ROOT / "data"


async def run_batch(csv_path: str, concurrency: int, fold_gpus: list = None) -> list[dict]:
    candidates = load_candidates(csv_path)
    # The core triage agent makes ~12 sequential API calls per candidate on
    # average — enough round trips that the local dev proxy's flakiness (see
    # anthropic_client.py) hit EVERY SINGLE candidate in a real 75-candidate
    # run, each falling back to "Agent run failed: Connection error" and
    # silently discarding real work already done in that turn (including,
    # in one run, 26 real AF3 folds that had actually completed before the
    # conversation's next API call dropped). Bypass the proxy here too.
    client = make_client()
    semaphore = asyncio.Semaphore(concurrency)
    fold_gpus = fold_gpus if fold_gpus is not None else DEFAULT_FOLD_GPUS
    enable_fold = bool(fold_gpus)
    # Bounds concurrent agent-triggered fold_complex calls to the number of
    # GPUs actually offered, shared across every candidate in this batch.
    fold_semaphore = asyncio.Semaphore(max(len(fold_gpus), 1)) if enable_fold else None
    fold_gpu_cycle = itertools.cycle(fold_gpus) if enable_fold else None
    results = [None] * len(candidates)

    async with hpa_mcp_session() as mcp_session:

        async def worker(i, candidate):
            async with semaphore:
                result = None
                last_exc = None
                for attempt in range(3):
                    try:
                        result = await evaluate_candidate(
                            client, mcp_session, candidate, fold_semaphore, fold_gpu_cycle, enable_fold
                        )
                        break
                    except (anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
                        last_exc = exc
                        if attempt < 2:
                            print(f"  {candidate.pair_id}: transient error ({exc}), retrying ({attempt + 1}/2)...")
                            await asyncio.sleep(5)
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        break
                if result is None:
                    result = {
                        "pair": [candidate.protein_a, candidate.protein_b],
                        "verdict": "INSUFFICIENT_EVIDENCE",
                        "confidence": "low",
                        "rationale": f"Agent run failed: {last_exc}",
                        "other_subunits": [],
                        "follow_up": "Retry this candidate.",
                        "_transcript": [],
                    }
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                (LOG_DIR / f"{candidate.pair_id}.json").write_text(json.dumps(result, indent=2))
                results[i] = {**result, "iptm": candidate.iptm, "ptm": candidate.ptm}

        await asyncio.gather(*(worker(i, c) for i, c in enumerate(candidates)))

    return results


def write_report(results: list[dict], out_path: Path) -> None:
    fields = [
        "protein_a",
        "protein_b",
        "verdict",
        "confidence",
        "iptm",
        "ptm",
        "rationale",
        "ptm_glycosylation_evidence",
        "cooccurrence_evidence",
        "known_interaction_evidence",
        "other_subunits",
        "follow_up",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "protein_a": r["pair"][0],
                    "protein_b": r["pair"][1],
                    "verdict": r.get("verdict"),
                    "confidence": r.get("confidence"),
                    "iptm": r.get("iptm"),
                    "ptm": r.get("ptm"),
                    "rationale": r.get("rationale"),
                    "ptm_glycosylation_evidence": r.get("ptm_glycosylation_evidence"),
                    "cooccurrence_evidence": r.get("cooccurrence_evidence"),
                    "known_interaction_evidence": r.get("known_interaction_evidence"),
                    "other_subunits": ";".join(r.get("other_subunits") or []),
                    "follow_up": r.get("follow_up"),
                }
            )


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default=str(DATA_DIR / "verdicts.csv"))
    parser.add_argument(
        "--api-key", default=None,
        help="Anthropic API key. Overrides ANTHROPIC_API_KEY from the environment/.env if given.",
    )
    parser.add_argument(
        "--fold-gpus", default=",".join(DEFAULT_FOLD_GPUS),
        help="Comma-separated GPU device IDs the agent's fold_complex tool may use "
             "(bounds concurrent real AF3 folds across the whole batch); empty string disables folding",
    )
    parser.add_argument(
        "--no-fold", action="store_true",
        help="Disable the fold_complex tool entirely (no local AlphaFast install required). "
             "Equivalent to --fold-gpus \"\".",
    )
    parser.add_argument("--skip-validation-strategies", action="store_true",
                         help="Skip designing PLA/stimulation/co-IP validation strategies for recommended candidates")
    parser.add_argument("--skip-report", action="store_true",
                         help="Skip building the traceable HTML report")
    parser.add_argument("--report-out", default=str(DATA_DIR / "report.html"))
    parser.add_argument("--skip-review", action="store_true",
                         help="Skip the review agent's audit of verdicts + validation strategies")
    parser.add_argument("--skip-gate", action="store_true",
                         help="Skip the deterministic evidence gate that flags and reconsiders verdicts "
                              "contradicted by STRING/CellPhoneDB/HPA/fold evidence")
    parser.add_argument("--skip-fold-reconsideration", action="store_true",
                         help="Skip re-reconsidering verdicts against any REAL executed follow-up folds in "
                              "--fold-summary (e.g. a LIKELY_ARTIFACT_PTM/LIKELY_SUBCOMPLEX verdict whose named "
                              "concern a real fold actually checked and cleared/confirmed gets genuinely revisited, "
                              "not just noted in the report)")
    parser.add_argument("--fold-summary", default=str(ROOT / "fold_runs" / "followups" / "_summary.json"))
    args = parser.parse_args()

    if args.api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.api_key

    fold_gpus = [] if args.no_fold else [g for g in args.fold_gpus.split(",") if g]
    results = asyncio.run(run_batch(args.csv_path, args.concurrency, fold_gpus))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    write_report(results, out_path)
    print(f"Wrote {len(results)} verdicts to {out_path}")

    if not args.skip_gate:
        asyncio.run(run_gate_reconsideration(out_path, LOG_DIR, Path(args.fold_summary)))

    if not args.skip_fold_reconsideration:
        asyncio.run(run_fold_reconsideration(out_path, Path(args.fold_summary)))

    strategies = []
    if not args.skip_validation_strategies:
        strategies_path = DATA_DIR / "validation_strategies.json"
        strategies = asyncio.run(run_validation_design(out_path, strategies_path))

    if not args.skip_report:
        build_report(str(out_path), str(LOG_DIR), args.report_out)

    if not args.skip_review:
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        findings = deterministic_checks(rows, strategies, LOG_DIR, Path(args.fold_summary))
        if strategies:
            findings += asyncio.run(llm_audit(rows, strategies))
        findings_path = DATA_DIR / "review_findings.json"
        findings_path.write_text(json.dumps(findings, indent=2))
        by_sev = {}
        for f in findings:
            by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
        print(f"Review agent: {len(findings)} findings ({by_sev}) -> {findings_path}")


if __name__ == "__main__":
    main()
