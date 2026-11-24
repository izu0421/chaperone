"""Score the rescue re-folds: does a correct pose exist once AlphaFold3 is allowed
more than one draw, or once the wrong face is removed?

The screen ran modelSeeds=[1]. The topology rule then rejected 43 of 57 protein
candidates for putting the partner on the cytoplasmic face — a rule that rejects
61% of KNOWN binders on the chemokine panel, so it was discarding pose failures
and calling them biology.

This asks the question the screen could not: across five seeds, does ANY draw
place the partner on the extracellular face, and how confident is it when it
does? Three arms:

  A_multiseed      the 22 candidates with a secreted-type partner, same
                   construct, five seeds.
  B_ecd            GPR158, GPR37 and MAS1L truncated to their extracellular
                   domain, so the wrong face does not exist, plus
  B_ecd_decoy      the same truncated receptors against albumin, lysozyme and
                   RNase 1. Truncation exposes the faces the rest of the protein
                   was covering; if the decoys dock as well as the candidate, the
                   construct is making a surface rather than revealing one.
  C_control_*      chemokine pairs whose answer is known, folded the same way.
                   This is what makes arm A readable: five seeds give five
                   chances to get lucky, and without a control a rescued
                   candidate cannot be told from a rescued coincidence.

The headline is NOT "the best of five seeds", which is a maximum over draws and
rises for everything. It is whether the best-of-five SEPARATES known binders from
known non-binders in arm C by more than it did at one seed.

Usage:
    micromamba run -n structmod python -I scripts/32_score_refold_rescue.py --workers 12
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("rescue-score")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "rescue_out"

_spec = importlib.util.spec_from_file_location("sc14", ROOT / "scripts" / "14_score_orphans.py")
sc14 = importlib.util.module_from_spec(_spec)
sys.modules["sc14"] = sc14
_spec.loader.exec_module(sc14)
score_one = sc14.score_one


def seed_dirs(job_dir: Path) -> list[Path]:
    """Every per-seed model directory AF3 wrote, or the job itself if it wrote one.

    AF3's layout for multiple seeds is <job>/seed-N_sample-M/. Falling back to the
    job directory keeps this working if a run produced a single ranked model, so a
    layout change shows up as one row per job rather than as zero rows.
    """
    subs = sorted(p for p in job_dir.glob("seed-*") if p.is_dir())
    return subs or ([job_dir] if list(job_dir.glob("*_summary_confidences.json")) else [])


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "work" / "rescue_manifest.csv")
    jobs = [(r.job, r.arm) for r in man.itertuples() if (OUT / r.job).is_dir()]
    log.info("%d of %d jobs have an output directory", len(jobs), len(man))

    targets, meta = [], []
    for job, arm in jobs:
        for sd in seed_dirs(OUT / job):
            targets.append(sd)
            meta.append((job, arm, sd.name))
    if not targets:
        raise SystemExit("nothing scored yet")
    log.info("scoring %d per-seed models", len(targets))

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(score_one, targets))
    df = pd.DataFrame(rows)
    df["job"] = [m[0] for m in meta]
    df["arm"] = [m[1] for m in meta]
    df["seed"] = [m[2] for m in meta]
    df = df[df.iptm.notna()].copy()
    df["no_clash"] = df.min_sep >= 1.5
    dest = ROOT / "results" / "rescue_scores.csv"
    df.to_csv(dest, index=False)
    log.info("wrote %s (%d rows, %d jobs)", dest, len(df), df.job.nunique())

    # ---- per job: the single seed the screen would have taken, vs best of five ---
    best = (df.sort_values("pae_min_AB")
              .groupby(["job", "arm"], as_index=False)
              .first()
              .rename(columns={"pae_min_AB": "pae_best"}))
    first = (df[df.seed.str.contains("seed-1") | df.seed.eq(df.job)]
             .groupby(["job", "arm"], as_index=False).first()
             .rename(columns={"pae_min_AB": "pae_seed1"})[["job", "pae_seed1"]])
    j = best.merge(first, on="job", how="left")
    spread = (df.groupby("job").pae_min_AB.agg(["min", "max", "std"])
                .rename(columns={"min": "pae_min", "max": "pae_max", "std": "pae_sd"}))
    j = j.merge(spread, left_on="job", right_index=True, how="left")

    log.info("\nHOW MUCH ONE SEED COSTS, per arm:")
    log.info("%s", j.groupby("arm").agg(
        jobs=("job", "size"),
        median_seed1=("pae_seed1", "median"),
        median_best=("pae_best", "median"),
        median_spread=("pae_sd", "median")).round(2).to_string())

    # ---- arm C: does best-of-five separate known answers better than one seed? ---
    C = j[j.arm.str.startswith("C_control")]
    if len(C):
        pos = C[C.arm.eq("C_control_binder")]
        neg = C[C.arm.eq("C_control_negative")]
        log.info("\nTHE CONTROL -- chemokine pairs whose answer is known:")
        for col, lab in (("pae_seed1", "one seed, as the screen ran"),
                         ("pae_best", "best of five seeds")):
            au = auroc(-pos[col], -neg[col])
            log.info("  %-28s AUROC %.3f   (n = %d binders, %d non-binders)",
                     lab, au, len(pos), len(neg))
        log.info("  -> five seeds are five chances to get lucky. If AUROC does NOT rise,")
        log.info("     the extra draws help binders and non-binders equally and rescue")
        log.info("     nothing; a candidate that improves is improving by the same amount.")

    # ---- arm B: did truncation reveal a surface, or manufacture one? -------------
    B = j[j.arm.str.startswith("B_")]
    if len(B):
        log.info("\nTRUNCATION TO THE EXTRACELLULAR DOMAIN:")
        for r in B[B.arm.eq("B_ecd")].itertuples():
            rec = r.job.split("__")[0].replace("B_", "").replace("ECD", "")
            dec = B[(B.arm.eq("B_ecd_decoy")) & (B.job.str.contains(rec))]
            if len(dec):
                log.info("  %-28s candidate PAE %.2f   decoys %.2f-%.2f (n=%d)",
                         r.job, r.pae_best, dec.pae_best.min(), dec.pae_best.max(), len(dec))
                if r.pae_best >= dec.pae_best.min():
                    log.warning("     a decoy docks at least as well -- the cut face is a "
                                "surface the construct made, not one it revealed")
            else:
                log.info("  %-28s candidate PAE %.2f   (no decoy folded yet)",
                         r.job, r.pae_best)

    # ---- arm A: which candidates a correct pose now exists for -------------------
    A = j[j.arm.eq("A_multiseed")].sort_values("pae_best")
    if len(A):
        log.info("\nCANDIDATES, best of five seeds (interface PAE, lower is better):")
        for r in A.itertuples():
            moved = (r.pae_seed1 - r.pae_best) if pd.notna(r.pae_seed1) else float("nan")
            log.info("  %-30s seed1 %5.2f -> best %5.2f  (gain %.2f)",
                     r.job.replace("A_", ""), r.pae_seed1, r.pae_best, moved)
    j.to_csv(ROOT / "results" / "rescue_per_job.csv", index=False)
    log.info("\nwrote results/rescue_per_job.csv")


if __name__ == "__main__":
    main()
