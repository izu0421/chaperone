"""Count what the gate says about a verdicts file, for the figure's cycle panel.

Separate from assemble_screen_verdicts.py because the gate needs gemmi and the
assembler needs pandas, and no environment on this box has both. Writes
gate_stats.json beside the verdicts file so the assembler can read it without
either importing the other.

Usage:
    ./.venv/bin/python scripts/gate_stats.py reports/screen_verdicts_current/verdicts.csv
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chaperone.deterministic_gate import (  # noqa: E402
    compute_gate_facts, find_contradictions, gather_fold_results,
    load_fold_results, load_tool_calls,
)

def main() -> int:
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(open(path, newline="")))
    fold_by_pair = load_fold_results(ROOT / "fold_runs" / "followups" / "_summary.json")
    flagged = topo = with_fold = 0
    for r in rows:
        tc = load_tool_calls(r, ROOT / "log")
        fr = gather_fold_results(r, tc, fold_by_pair)
        if fr:
            with_fold += 1
        facts = compute_gate_facts(r, tc, fr)
        if facts["fold_evidence_status"] == "topology_violation":
            topo += 1
        if find_contradictions(r, facts):
            flagged += 1
    out = {"candidates": len(rows), "gate_flagged": flagged,
           "topology_violations": topo, "with_a_real_fold": with_fold}
    (path.parent / "gate_stats.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
