"""Score arm P: does the candidate interface survive when only the reachable half is folded?

Three comparisons, in the order that decides whether the arm can be read at all:

  1. the known answers (P_known_*), if folded: can this construct recover a
     real ectodomain interaction and separate it from a matched non-interaction?
     If not, nothing else here means anything.
  2. candidate vs its own PERMUTED negatives: the same receptor against other
     candidates' ectodomains, matched on size, compartment, cut and MSA build.
     A candidate that does not beat its permutations has shown nothing.
  3. candidate vs the screen's own full-length fold: how much of the original
     interface was the part that cannot reach.

Usage:
    micromamba run -n structmod python -I scripts/39_score_ectodomain.py
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ecd-score")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "ectodomain_out"

_s = iu.spec_from_file_location("sc35", ROOT / "scripts" / "35_score_subcomplex_and_mature.py")
sc35 = iu.module_from_spec(_s)
sys.modules["sc35"] = sc35
_s.loader.exec_module(sc35)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    GATE = json.loads((ROOT / "results" / "protein_thresholds.json").read_text()
                      )["gate"]["pae_min_AB_max"]

    man = pd.read_csv(ROOT / "work" / "ectodomain_manifest.csv")
    targets, meta = [], []
    for r in man.itertuples():
        d = OUT / r.job
        if not d.is_dir():
            continue
        for sd in sc35.seed_dirs(d):
            targets.append(sd); meta.append((r.job, r.arm, r.receptor, r.partner))
    log.info("%d of %d jobs have output; scoring %d models",
             len({m[0] for m in meta}), len(man), len(targets))
    if not targets:
        raise SystemExit("nothing folded yet")

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(sc35.score_multi, targets))
    df = pd.DataFrame(rows)
    for i, k in enumerate(("job", "arm", "receptor", "partner")):
        df[k] = [m[i] for m in meta]
    df = df[df.iptm.notna()].copy()
    df.to_csv(ROOT / "results" / "ectodomain_scores.csv", index=False)
    best = df.sort_values("pae_min_AB").groupby(
        ["job", "arm", "receptor", "partner"], as_index=False).first()
    log.info("wrote results/ectodomain_scores.csv")

    # 1. known answers
    K = best[best.arm.str.startswith("P_known")]
    if len(K) >= 6:
        pos = K[K.arm.str.endswith("positive")]
        neg = K[K.arm.str.endswith("negative")]
        au = sc35.auroc(-pos.pae_min_AB, -neg.pae_min_AB)
        log.info("\nKNOWN ANSWERS under this construct: AUROC %.3f (%d positive, %d negative)",
                 au, len(pos), len(neg))
        log.info("  positives median PAE %.2f, negatives %.2f; %d of %d positives clear the "
                 "%.2f A gate", pos.pae_min_AB.median(), neg.pae_min_AB.median(),
                 int((pos.pae_min_AB <= GATE).sum()), len(pos), GATE)
        log.info("  -> a construct that cannot recover a real ectodomain interaction cannot")
        log.info("     refute a candidate one.")
    elif len(K):
        log.info("\nKNOWN ANSWERS: only %d folded so far", len(K))
    else:
        log.info("\nKNOWN ANSWERS: none folded -- arm P is not yet interpretable")

    # 2 and 3. candidates against their permutations and against the screen
    scr = pd.read_csv(ROOT / "results" / "orphan_scores.csv")
    full = {(str(r.receptor), str(r.partner)): r.pae_min_AB for r in scr.itertuples()}
    # A collapse only means "interface lost" if the chains themselves still fold.
    # Compared against each pair's own full-length fold, never an absolute cutoff:
    # a GPCR's mean pLDDT is diluted by its disordered tails, and cutting a
    # partner to its ectodomain REMOVES disordered regions, so the partner's
    # number should rise if anything. It falling is the signal to look at.
    pld = {(str(r.receptor), str(r.partner)): (r.plddt_A, r.plddt_B)
           for r in scr.itertuples()}
    # Beating a permuted negative is not enough. Removing the partner's tail
    # removes the partner's impossible half; the RECEPTOR's cytoplasmic face is
    # still there, and an ectodomain can dock onto it just as happily. Measured
    # on this arm, 5 of 9 candidates keep a mostly cytoplasmic receptor face --
    # and so do their permuted negatives, which is why the PAE comparison alone
    # would have read as a pass for a pose that cannot exist.
    zf = ROOT / "results" / "ectodomain_interface_zone.csv"
    zone = (pd.read_csv(zf).set_index("job").rec_cyto.to_dict() if zf.exists() else {})
    for pref, lab in (("P_", "PARTNER ECTODOMAIN ONLY"), ("P_both_", "BOTH ECTODOMAINS")):
        arm_c = "P_candidate" if pref == "P_" else "P_both_ecd"
        arm_p = "P_permuted" if pref == "P_" else "P_both_permuted"
        C = best[best.arm.eq(arm_c)]
        P = best[best.arm.eq(arm_p)]
        if not len(C):
            continue
        log.info("\n%s (interface PAE, lower is better)", lab)
        log.info("  %-22s %8s %8s  %-11s %-20s %s", "pair", "screen", "cut",
                 "dpLDDT A/B", "best permuted neg", "verdict")
        for r in C.sort_values("pae_min_AB").itertuples():
            perm = P[P.receptor.eq(r.receptor)]
            fu = full.get((r.receptor, r.partner), np.nan)
            pa, pb_ = pld.get((r.receptor, r.partner), (np.nan, np.nan))
            dp = f"{r.plddt_A - pa:+.1f}/{r.plddt_B - pb_:+.1f}"
            cyto = zone.get(r.job, float("nan"))
            if len(perm):
                b = perm.loc[perm.pae_min_AB.idxmin()]
                if cyto >= 0.3:
                    v = f"FAIL -- {100 * cyto:.0f}% of the receptor face is cytoplasmic"
                else:
                    v = ("PASS" if r.pae_min_AB < b.pae_min_AB else
                         "FAIL -- a permuted ectodomain docks as well")
                log.info("  %-22s %8.2f %8.2f  %-11s %-20s %s",
                         f"{r.receptor}+{r.partner}", fu, r.pae_min_AB, dp,
                         f"{b.partner} {b.pae_min_AB:.2f}", v)
            else:
                note = ("(no negative folded yet)" if cyto != cyto or cyto < 0.3 else
                        f"({100 * cyto:.0f}% cytoplasmic face -- unreachable)")
                log.info("  %-22s %8.2f %8.2f  %-11s %-20s %s",
                         f"{r.receptor}+{r.partner}", fu, r.pae_min_AB, dp, "-", note)
        if len(P):
            log.info("  permuted negatives: median %.2f, best %.2f over %d folds",
                     P.pae_min_AB.median(), P.pae_min_AB.min(), len(P))

    # One verdict, written once. The ranking table used to recompute it from the
    # PAE columns, which meant the zone requirement existed in one place and not
    # the other -- and the table would have gone on calling a 98%-cytoplasmic
    # pose a PASS.
    verdicts = {}
    for r in best[best.arm.isin(["P_candidate", "P_both_ecd"])].itertuples():
        perm_arm = "P_permuted" if r.arm == "P_candidate" else "P_both_permuted"
        perm = best[best.arm.eq(perm_arm) & best.receptor.eq(r.receptor)]
        cyto = zone.get(r.job, float("nan"))
        if cyto >= 0.3:
            v = f"FAIL -- {100 * cyto:.0f}% cytoplasmic receptor face"
        elif not len(perm):
            v = "folded, no negative yet"
        elif r.pae_min_AB < perm.pae_min_AB.min():
            v = "PASS"
        else:
            v = "FAIL -- a permuted ectodomain docks as well"
        verdicts[r.job] = v
    best["verdict"] = best.job.map(verdicts)
    best["rec_cyto"] = best.job.map(zone)
    best.to_csv(ROOT / "results" / "ectodomain_per_job.csv", index=False)
    log.info("\nwrote results/ectodomain_per_job.csv with one verdict column")


if __name__ == "__main__":
    main()
