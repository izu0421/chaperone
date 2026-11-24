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
    best.to_csv(ROOT / "results" / "ectodomain_per_job.csv", index=False)
    log.info("wrote results/ectodomain_scores.csv and ectodomain_per_job.csv")

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
    for pref, lab in (("P_", "PARTNER ECTODOMAIN ONLY"), ("P_both_", "BOTH ECTODOMAINS")):
        arm_c = "P_candidate" if pref == "P_" else "P_both_ecd"
        arm_p = "P_permuted" if pref == "P_" else "P_both_permuted"
        C = best[best.arm.eq(arm_c)]
        P = best[best.arm.eq(arm_p)]
        if not len(C):
            continue
        log.info("\n%s (interface PAE, lower is better)", lab)
        log.info("  %-22s %8s %8s   %-22s %s", "pair", "screen", "cut",
                 "best permuted negative", "verdict")
        for r in C.sort_values("pae_min_AB").itertuples():
            perm = P[P.receptor.eq(r.receptor)]
            fu = full.get((r.receptor, r.partner), np.nan)
            if len(perm):
                b = perm.loc[perm.pae_min_AB.idxmin()]
                v = ("PASS" if r.pae_min_AB < b.pae_min_AB else
                     "FAIL -- a permuted ectodomain docks as well")
                log.info("  %-22s %8.2f %8.2f   %-22s %s",
                         f"{r.receptor}+{r.partner}", fu, r.pae_min_AB,
                         f"{b.partner} {b.pae_min_AB:.2f}", v)
            else:
                log.info("  %-22s %8.2f %8.2f   %-22s (no negative folded yet)",
                         f"{r.receptor}+{r.partner}", fu, r.pae_min_AB, "-")
        if len(P):
            log.info("  permuted negatives: median %.2f, best %.2f over %d folds",
                     P.pae_min_AB.median(), P.pae_min_AB.min(), len(P))


if __name__ == "__main__":
    main()
