"""Publish the re-gated verdicts into the report directory, with the summary
counts recomputed from the files rather than typed.

The gate was inert for two of its rules until 2026-11-23 (partner_breadth_status
read a column that only existed on the candidate sheet, and the topology rule
searched an empty fold directory). After both were fixed the gate flags far more
candidates, so the verdicts it reconsidered are not the ones published at 14:17.
This script moves the corrected file in and records what changed, so the figure
and the manuscript read the same numbers.

gate_flagged is measured by running the FIXED gate over the ORIGINAL verdicts,
which is what the reconsideration pass saw. gate_changed is a diff of the two
verdict files. Neither is inferred from the verdict categories: 53 verdicts sit
in flaggable categories and the gate flagged a different set.
"""
import csv, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chaperone.deterministic_gate import (  # noqa: E402
    compute_gate_facts, find_contradictions, gather_fold_results,
    load_fold_results, load_tool_calls,
)

REPORT = ROOT / "reports" / "orphan_shortlist_2026-11-23"
REGATED = ROOT / "data" / "verdicts_prefix.csv"
PUBLISHED = REPORT / "verdicts.csv"
ORIGINAL = REPORT / "verdicts.csv.pre-regate"     # the first published run
SUMMARY = REPORT / "run_summary.json"
LOG_DIR = ROOT / "log"
FOLD_SUMMARY = ROOT / "fold_runs" / "followups" / "_summary.json"

RECOMMENDED = {"CONFIRMED_NOVEL"}


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return f"{row['protein_a']}/{row['protein_b']}"


def main():
    old, new = read(PUBLISHED), read(REGATED)
    if len(old) != len(new):
        sys.exit(f"refusing to publish: {len(old)} published rows, {len(new)} re-gated rows")
    if {key(r) for r in old} != {key(r) for r in new}:
        sys.exit("refusing to publish: the two files describe different pairs")

    # The gate has now been repaired twice, so "flagged" and "changed" are
    # reported against the FIRST published run, not against whatever the previous
    # round left behind. Otherwise each round reports only its own increment and
    # the figure shows the smallest of them.
    first = read(ORIGINAL) if ORIGINAL.exists() else old
    fold_by_pair = load_fold_results(FOLD_SUMMARY)
    flagged = []
    for row in first:
        tool_calls = load_tool_calls(row, LOG_DIR)
        facts = compute_gate_facts(row, tool_calls,
                                   gather_fold_results(row, tool_calls, fold_by_pair))
        if find_contradictions(row, facts):
            flagged.append(key(row))

    first_by, new_by = {key(r): r for r in first}, {key(r): r for r in new}
    changed = [k for k in first_by if first_by[k]["verdict"] != new_by[k]["verdict"]]
    this_round = [k for k in {key(r): r for r in old}
                  if {key(r): r for r in old}[k]["verdict"] != new_by[k]["verdict"]]

    print(f"against the first published run ({ORIGINAL.name if ORIGINAL.exists() else PUBLISHED.name}):")
    print(f"  gate flags {len(flagged)}/{len(first)} candidates with every rule live")
    print(f"  {len(changed)} verdicts differ, {len(this_round)} of them changed in this round")
    for k in sorted(changed):
        mark = '*' if k in this_round else ' '
        print(f"  {mark} {k}: {first_by[k]['verdict']} -> {new_by[k]['verdict']}")

    # keep every earlier published copy: the gate has been fixed more than once
    # and overwriting the backup would lose the only record of what it said before
    if PUBLISHED.read_bytes() == REGATED.read_bytes():
        print("published file already matches the re-gated one; no new backup")
        n = backup = None
    else:
        n, backup = 1, PUBLISHED.with_suffix(".csv.pre-regate")
    if backup is not None:
        while backup.exists():
            n += 1
            backup = PUBLISHED.with_suffix(f".csv.pre-regate{n}")
        shutil.copy2(PUBLISHED, backup)
        print(f"previous verdicts kept at {backup.name}")
    shutil.copy2(REGATED, PUBLISHED)

    summary = json.loads(SUMMARY.read_text())
    summary["gate_flagged"] = len(flagged)
    summary["gate_reconsidered"] = len(flagged)
    summary["gate_changed"] = len(changed)
    summary["gate_rounds"] = 1 + sum(1 for q in REPORT.glob("verdicts.csv.pre-regate*"))
    summary["recommended"] = sum(r["verdict"] in RECOMMENDED for r in new)
    summary["note"] = ("gate_flagged/reconsidered are the fixed gate run over the "
                       "pre-reconsideration verdicts; gate_changed is a diff of the "
                       "two verdict files; every other number is counted from the "
                       "files in this directory")
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nrecommended {summary['recommended']} (was 14). Wrote {PUBLISHED} and {SUMMARY}")


if __name__ == "__main__":
    main()
