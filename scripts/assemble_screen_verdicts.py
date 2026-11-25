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

# v2: re-triaged after the membrane-topology rule was removed from the
# contradiction trigger. Measured on the 256-fold chemokine panel, that rule
# rejects 61% of KNOWN binders and 55% of known non-binders, so it was
# discarding AlphaFold3 pose failures and recording them as biology. 47 of the
# 51 rejections in the v1 run cited it; removing it took the shortlist from 4
# to 18.
PROTEIN_RUN = ROOT / "reports" / "protein_pae_2026-11-24_v2" / "verdicts.csv"
# The clash floor was lowered from 1.5 to 0.5 A on 2026-11-24 with a measurement
# table behind it, which admitted 5 more pairs. They were triaged in their own
# run and one came back CONFIRMED_NOVEL -- GPR22 + SLITRK2, interface PAE 2.86,
# better than 50 of the 57 pairs in the main run. It was missing from every
# downstream table for a day because this list was hardcoded and that run was
# not on it. The guard below now refuses to write if the gate admits a pair no
# run has triaged, so a new run cannot be forgotten the same way twice.
PROTEIN_CLASHFLOOR = ROOT / "reports" / "protein_pae_clashfloor" / "verdicts.csv"
METAB_OLD = ROOT / "reports" / "orphan_shortlist_2026-11-23" / "verdicts.csv"
METAB_GAP = ROOT / "reports" / "protein_pae_2026-11-24" / "verdicts_metabolite_gap.csv"
# Both GABA pairs came back INSUFFICIENT_EVIDENCE only because ChEMBL returned
# HTTP 500 to the two checks that discriminate for a small molecule, twice each.
# ChEMBL recovered; re-running the same pipeline on the same two rows resolved
# both. A tool outage is not a result, and leaving it in the table would have
# reported two unmeasured pairs as unmeasurable.
METAB_RERUN = ROOT / "reports" / "metabolite_gaba_rerun_2026-11-24" / "verdicts.csv"

RECOMMENDED = {"CONFIRMED_NOVEL"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    prot_parts = []
    for path in (PROTEIN_RUN, PROTEIN_CLASHFLOOR):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d = d[d.partner_kind.eq("protein")].copy()
        d["arm"] = "protein"
        d["run"] = path.parent.name    # derived, so it cannot name a run we stopped using
        prot_parts.append(d)
    prot = pd.concat(prot_parts, ignore_index=True)
    prot["_k"] = prot.protein_a.astype(str) + "|" + prot.protein_b.astype(str)
    dup = prot[prot._k.duplicated(keep=False)]
    if len(dup):
        print(f"{dup._k.nunique()} protein pairs triaged in more than one run; keeping the later:")
        for k, g in dup.groupby("_k"):
            print("  " + k + ": " + ", ".join(f"{r.run} -> {r.verdict}" for r in g.itertuples()))
    prot = prot.drop_duplicates("_k", keep="last").drop(columns="_k")

    # Does every pair the CURRENT gate admits have a verdict? This is the check
    # that was missing: the gate moved, five pairs were admitted, and nothing
    # downstream noticed that one of them had no row.
    try:
        import json as _json
        sm_sc = pd.read_csv(SM / "results" / "orphan_scores.csv")
        th = _json.loads((SM / "results" / "protein_thresholds.json").read_text())["gate"]
        adm = sm_sc[(sm_sc.kind.eq("protein"))
                    & (sm_sc.pae_min_AB <= th["pae_min_AB_max"])
                    & (sm_sc.min_sep >= th.get("min_sep", 0))]
        want_p = {(str(r.receptor), str(r.partner)) for r in adm.itertuples()}
        got_p = {(str(r.protein_b), str(r.protein_a)) for r in prot.itertuples()}
        missing_p = want_p - got_p
        if missing_p:
            sys.exit(f"refusing to write: the current gate admits {len(missing_p)} protein "
                     f"pairs with no verdict in any run: {sorted(missing_p)}. Triage them "
                     f"or add the run that did.")
        print(f"gate check: all {len(want_p)} pairs the current gate admits have a verdict")
    except FileNotFoundError as exc:
        print(f"gate check skipped: {exc}")

    # the metabolite arm's current shortlist, by identity and not by position
    hits = pd.read_csv(SM / "results" / "metabolite_calibrated_hits.csv")
    ident = pd.read_csv(SM / "results" / "chaperone_metabolite_identities.csv")
    std = dict(zip(ident.screen_name, ident.standard_name))
    want = {(r.receptor.upper(), str(std.get(r.metabolite, r.metabolite)).lower())
            for r in hits.itertuples()}

    met_rows = []
    # order matters: duplicates keep the LAST, so a re-run goes at the end
    for path, run in ((METAB_OLD, METAB_OLD.parent.name),
                      (METAB_GAP, METAB_GAP.parent.name + "_metabolite_gap"),
                      (METAB_RERUN, METAB_RERUN.parent.name)):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if "partner_kind" in d:
            d = d[d.partner_kind.ne("protein")]
        # A run may name a molecule either way: the screen's own name ("GABA") or
        # the standardised one the agent resolved it to ("4-aminobutanoic acid").
        # Matching on the raw string drops every row that used the other name,
        # and the drop is invisible -- a re-run of two pairs vanished this way
        # while the stale rows it was meant to replace stayed in the table.
        d["protein_b"] = [std.get(x, x) for x in d.protein_b]
        before = len(d)
        d = d[d.apply(lambda r: (str(r.protein_a).upper(), str(r.protein_b).lower()) in want,
                      axis=1)].copy()
        if before and not len(d):
            sys.exit(f"refusing to write: {path.parent.name} has {before} metabolite "
                     f"rows and NONE matched the calibrated set. Either the run is for "
                     f"different pairs or the names do not match; check before assuming "
                     f"it is obsolete.")
        if before != len(d):
            print(f"  {path.parent.name}: kept {len(d)} of {before} metabolite rows")
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
