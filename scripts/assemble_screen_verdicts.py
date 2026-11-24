"""Assemble the screen's current verdicts from the runs that produced them.

The agent layer has been run more than once, on different candidate sets,
because the two arms were re-gated at different times:

  protein arm     re-gated on interface PAE (2026-11-24) and re-triaged whole,
                  57 candidates -> reports/protein_pae_2026-11-24/
  metabolite arm  gated on the calibrated small-molecule cuts, 6 receptor-
                  molecule pairs. Three were triaged in the 2026-11-23 run;
                  the other three were not, because that run's metabolite
                  selection used the older ad-hoc gate and covered a different
                  set. They were triaged separately into
                  verdicts_metabolite_gap.csv.

So no single verdicts file describes the screen as it now stands, and reading
any one of them as if it did would report an arm that is half covered. This
writes the union, with each row carrying the run it came from, plus a summary
counted from the rows rather than typed.

A row is only included if the fold behind it clears the arm's CURRENT gate.
That is the check that matters: the 2026-11-23 run triaged 81 protein
candidates selected by three older routes, and carrying those forward would
describe a shortlist the screen no longer produces.

Usage:
    python scripts/assemble_screen_verdicts.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SM = Path("/data/yzy21/yy/space_agent/structural_modelling")
OUT = ROOT / "reports" / "screen_verdicts_current"

PROTEIN_RUN = ROOT / "reports" / "protein_pae_2026-11-24" / "verdicts.csv"
METAB_OLD = ROOT / "reports" / "orphan_shortlist_2026-11-23" / "verdicts.csv"
METAB_GAP = ROOT / "reports" / "protein_pae_2026-11-24" / "verdicts_metabolite_gap.csv"

RECOMMENDED = {"CONFIRMED_NOVEL"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    prot = pd.read_csv(PROTEIN_RUN)
    prot = prot[prot.partner_kind.eq("protein")].copy()
    prot["arm"] = "protein"
    prot["run"] = "protein_pae_2026-11-24"

    # the metabolite arm's current shortlist, by identity and not by position
    hits = pd.read_csv(SM / "results" / "metabolite_calibrated_hits.csv")
    ident = pd.read_csv(SM / "results" / "chaperone_metabolite_identities.csv")
    std = dict(zip(ident.screen_name, ident.standard_name))
    want = {(r.receptor.upper(), str(std.get(r.metabolite, r.metabolite)).lower())
            for r in hits.itertuples()}

    met_rows = []
    for path, run in ((METAB_OLD, "orphan_shortlist_2026-11-23"),
                      (METAB_GAP, "protein_pae_2026-11-24_metabolite_gap")):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if "partner_kind" in d:
            d = d[d.partner_kind.ne("protein")]
        d = d[d.apply(lambda r: (str(r.protein_a).upper(), str(r.protein_b).lower()) in want,
                      axis=1)].copy()
        d["arm"] = "metabolite"
        d["run"] = run
        met_rows.append(d)
    met = pd.concat(met_rows, ignore_index=True) if met_rows else pd.DataFrame()

    # A pair triaged in more than one run keeps the LATEST verdict, and the
    # duplicate is reported rather than dropped silently.
    if len(met):
        met["_key"] = met.protein_a.str.upper() + "|" + met.protein_b.str.lower()
        dupes = met[met._key.duplicated(keep=False)]
        if len(dupes):
            print(f"{dupes._key.nunique()} metabolite pairs triaged in both runs; "
                  f"keeping the later verdict:")
            for k, g in dupes.groupby("_key"):
                print(f"  {k}: " + ", ".join(f"{r.run} -> {r.verdict}" for r in g.itertuples()))
        met = met.drop_duplicates("_key", keep="last").drop(columns="_key")

    missing = want - {(str(r.protein_a).upper(), str(r.protein_b).lower())
                      for r in met.itertuples()} if len(met) else want
    if missing:
        sys.exit(f"refusing to write: {len(missing)} calibrated metabolite pairs have no "
                 f"verdict in any run: {sorted(missing)}")

    both = pd.concat([prot, met], ignore_index=True)
    dest = OUT / "verdicts.csv"
    both.to_csv(dest, index=False)

    summary = {
        "assembled": "scripts/assemble_screen_verdicts.py",
        "runs": sorted(both.run.unique().tolist()),
        "candidates": int(len(both)),
        "by_arm": both.arm.value_counts().to_dict(),
        "verdicts": both.verdict.value_counts().to_dict(),
        "recommended": int(both.verdict.isin(RECOMMENDED).sum()),
        "recommended_by_arm": both[both.verdict.isin(RECOMMENDED)].arm.value_counts().to_dict(),
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    print(f"\n{len(both)} candidates: {summary['by_arm']}")
    print(both.verdict.value_counts().to_string())
    print(f"\nrecommended {summary['recommended']} {summary['recommended_by_arm']}")
    for r in both[both.verdict.isin(RECOMMENDED)].itertuples():
        print(f"  {r.protein_b} + {r.protein_a}  ({r.arm}, {r.confidence})")
    print(f"\nwrote {dest} and run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
