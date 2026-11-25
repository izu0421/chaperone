"""A ligand-receptor pair has to win in BOTH directions, and none of these does.

The gate scores one fold. A high score can mean the receptor prefers this ligand,
or that the ligand fits most pockets, or that this pocket fits most ligands. One
number cannot separate the three, so ask twice:

  ARM A   fix the LIGAND, vary the receptor.
          Does the candidate outrank receptors whose own ligand is known and is
          not this one? For noradrenaline the answer was devastating: the true
          adrenoceptor ranked 3rd of 18 behind two known non-binders, and the
          two orphan hits ranked 6th and 7th behind four.

  ARM B   fix the RECEPTOR, vary the ligand.
          Does the candidate ligand beat the other 16 at the same pocket? GPR85
          scores >= 0.85 with twelve of seventeen unrelated molecules including
          lipids, which is a permissive pocket measured rather than inferred.

A real pair wins both. Measured here, no pair does:

    receptor   >=0.85    best              margin over 2nd   arm A rank
    GPR150     16/17     serotonin 0.94         0.00          3 of 18
    GPR26      13/17     acetylcholine 0.93     0.00          1 of 18
    GPR85      12/17     dopamine 0.90          0.00          7 of 18
    GPR153      5/17     noradrenaline 0.92    +0.02          6 of 18

GPR26 + glutamate wins arm A outright and ties three ways in arm B, against
acetylcholine and dopamine, while still giving 0.87 to anandamide -- a
twenty-carbon lipid that cannot occupy a small-amino-acid site. GPR153 has the
most selective pocket of the four and is the one arm A refuted.

Usage:
    micromamba run -n structmod python -I scripts/49_two_way_specificity.py
"""
from __future__ import annotations

import importlib.util as iu
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("twoway")
ROOT = Path(__file__).resolve().parent.parent

_s = iu.spec_from_file_location("sc35", ROOT / "scripts" / "35_score_subcomplex_and_mature.py")
sc35 = iu.module_from_spec(_s); sys.modules["sc35"] = sc35; _s.loader.exec_module(sc35)

SELECTIVE = 0.85           # the band the cognate pairs occupy


def score_dir(out: Path, manifest: Path) -> pd.DataFrame:
    man = pd.read_csv(manifest)
    rows = []
    for r in man.itertuples():
        d = out / r.job
        if not d.is_dir():
            continue
        s = sc35.score_multi(d)
        if s.get("iptm") is None:
            continue
        rec, lig = r.job.split("__", 1)
        rows.append({"receptor": rec, "ligand": lig, "iptm": s["iptm"],
                     "plddt_B": s.get("plddt_B"), "role": getattr(r, "role", "")})
    return pd.DataFrame(rows)


def main() -> None:
    glu = score_dir(ROOT / "results" / "glutamate_out", ROOT / "work" / "glutamate_manifest.csv")
    nor = pd.read_csv(ROOT / "results" / "noradrenaline_scores.csv")
    nor["receptor"] = nor.pair.str.split("__").str[0]
    nor["ligand"] = nor.pair.str.split("__").str[1]
    both = pd.concat([glu, nor[["receptor", "ligand", "iptm", "plddt_B", "role"]]],
                     ignore_index=True)

    out = []
    log.info("ARM A -- fix the ligand, vary the receptor\n")
    for lig, src in (("glutamate", glu), ("noradrenaline", nor)):
        a = src[src.ligand.eq(lig)].sort_values("iptm", ascending=False).reset_index(drop=True)
        if not len(a):
            continue
        log.info("  %s against %d receptors:", lig, len(a))
        pos = a.index[a.role.eq("positive_control")] if "role" in a else []
        if len(pos):
            log.info("    the receptor that genuinely binds it ranks %d, with %d known "
                     "non-binders above it", int(pos[0]) + 1, int(pos[0]))
        for rec in ("GPR26", "GPR150", "GPR85", "GPR153"):
            if (a.receptor == rec).any():
                k = int(a.index[a.receptor.eq(rec)][0]) + 1
                above = int((a.iloc[:k - 1].role == "negative_control").sum()) if "role" in a else k - 1
                log.info("    %-8s ranks %2d of %d, %d known non-binders above it",
                         rec, k, len(a), above)
                out.append({"receptor": rec, "ligand": lig, "armA_rank": k,
                            "armA_n": len(a), "armA_nonbinders_above": above})
        log.info("")

    log.info("ARM B -- fix the receptor, vary the ligand\n")
    log.info("  %-9s %7s  %-28s %-24s %8s", "receptor", ">=0.85", "best", "second", "margin")
    rows_b = {}
    for rec in ("GPR150", "GPR26", "GPR85", "GPR153"):
        g = both[both.receptor.eq(rec)].drop_duplicates("ligand").sort_values(
            "iptm", ascending=False)
        if len(g) < 5:
            continue
        t, s2 = g.iloc[0], g.iloc[1]
        log.info("  %-9s %3d/%-3d  %-20s %.2f  %-16s %.2f %+8.2f",
                 rec, int((g.iptm >= SELECTIVE).sum()), len(g),
                 t.ligand[:20], t.iptm, s2.ligand[:16], s2.iptm, t.iptm - s2.iptm)
        rows_b[rec] = {"n_ligands": int(len(g)),
                       "n_above_0.85": int((g.iptm >= SELECTIVE).sum()),
                       "best": t.ligand, "best_iptm": float(t.iptm),
                       "margin": float(t.iptm - s2.iptm)}

    t = pd.DataFrame(out)
    for r in t.itertuples():
        b = rows_b.get(r.receptor, {})
        t.loc[r.Index, "armB_above_0.85"] = b.get("n_above_0.85")
        t.loc[r.Index, "armB_best"] = b.get("best")
        t.loc[r.Index, "armB_margin"] = b.get("margin")
    t["wins_both"] = (t.armA_nonbinders_above == 0) & (t.armB_best == t.ligand) & (t.armB_margin > 0)
    t.to_csv(ROOT / "results" / "two_way_specificity.csv", index=False)
    log.info("\nPAIRS WINNING BOTH DIRECTIONS: %d of %d", int(t.wins_both.sum()), len(t))
    for r in t.itertuples():
        log.info("  %-8s + %-14s armA rank %d (%d above), armB best is %s (margin %+.2f)  %s",
                 r.receptor, r.ligand, r.armA_rank, r.armA_nonbinders_above,
                 r.armB_best, r.armB_margin, "WINS BOTH" if r.wins_both else "")
    json.dump(rows_b, open(ROOT / "results" / "pocket_permissiveness.json", "w"), indent=1)
    log.info("\nwrote results/two_way_specificity.csv and pocket_permissiveness.json")


if __name__ == "__main__":
    main()
