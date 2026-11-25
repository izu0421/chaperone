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
    B = j[j.arm.str.startswith("B")]
    if len(B):
        # A truncated receptor that does not fold cannot support a verdict either
        # way: a chain-pair PAE between a decoy and a random coil is not evidence
        # about binding. MIN_ECD gates on annotated segment LENGTH, which is not
        # the same thing as a construct that folds, so the bar has to be applied
        # after the fold. Measured here: GPR37ECD 23.7-27.5 mean chain-A pLDDT
        # against 53.8 for the intact receptor, GPR142ECD 19.7-26.8 against 64.8,
        # GPR158ECD 55.2-60.9 against 50.0 -- only GPR158's cut construct folds.
        pl = (df[df.arm.str.startswith("B")].groupby("job").plddt_A.mean()
              if "plddt_A" in df else pd.Series(dtype=float))
        intact = (df[df.arm.eq("A_multiseed")].assign(
                      rec=lambda d: d.job.str.replace("A_", "", regex=False).str.split("__").str[0])
                  .groupby("rec").plddt_A.mean())
        log.info("\nDOES THE TRUNCATED RECEPTOR FOLD AT ALL?")
        unfolded = set()
        for job in sorted(pl.index):
            rec = job.replace("B_", "").split("ECD")[0]
            ref = intact.get(rec, float("nan"))
            bad = pd.notna(ref) and pl[job] < 0.75 * ref
            if bad:
                unfolded.add(job)
            log.info("  %-28s chain-A pLDDT %4.1f   intact receptor %4.1f%s",
                     job, pl[job], ref, "   <- does not fold, verdict withheld" if bad else "")

        log.info("\nTRUNCATION TO THE EXTRACELLULAR DOMAIN:")
        for r in B[B.arm.isin(["B_ecd", "B2_ntd"])].itertuples():
            pref, dec_arm = (("B_", "B_ecd_decoy") if r.arm == "B_ecd"
                             else ("B2_", "B2_ntd_decoy"))
            rec = r.job.split("__")[0].replace(pref, "").replace("ECD", "").replace("NTD", "")
            dec = B[(B.arm.eq(dec_arm)) & (B.job.str.contains(rec))]
            if len(dec):
                # best-of-N against best-of-N. Quoting the decoys' WORST draw as the
                # upper bound would pair a candidate's maximum with a decoy's minimum.
                log.info("  %-28s candidate PAE %.2f   decoys %.2f-%.2f (best of %d draws each, n=%d)",
                         r.job, r.pae_best, dec.pae_best.min(), dec.pae_best.max(),
                         int(df[df.job.eq(r.job)].shape[0]), len(dec))
                if r.job in unfolded or dec.job.isin(unfolded).any():
                    log.warning("     WITHHELD: the truncated receptor does not fold in this "
                                "job or its decoys, so this comparison measures nothing")
                elif r.pae_best >= dec.pae_best.min():
                    log.warning("     a decoy docks at least as well -- the cut face is a "
                                "surface the construct made, not one it revealed")
                else:
                    log.info("     candidate beats every decoy")
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
    # One verdict per ECD job, written here so the ranking table reads it instead
    # of recomputing a comparison that has a construct-folds precondition.
    verd = {}
    for r in B[B.arm.isin(["B_ecd", "B2_ntd"])].itertuples():
        if r.arm == "B_ecd":
            rec = r.job.replace("B_", "").split("ECD")[0]
            dec = B[(B.arm.eq("B_ecd_decoy")) & (B.job.str.startswith(f"B_{rec}ECD__"))]
        else:
            rec = r.job.replace("B2_", "").split("NTD")[0]
            dec = B[(B.arm.eq("B2_ntd_decoy")) & (B.job.str.startswith(f"B2_{rec}NTD__"))]
        if r.job in unfolded or dec.job.isin(unfolded).any():
            verd[r.job] = "WITHHELD -- the truncated receptor does not fold"
        elif not len(dec):
            verd[r.job] = "no decoy folded"
        elif r.pae_best < dec.pae_best.min():
            verd[r.job] = "PASS"
        else:
            verd[r.job] = "FAIL -- a decoy docks as well"
    j["verdict"] = j.job.map(verd)
    j["plddt_A_mean"] = j.job.map(df.groupby("job").plddt_A.mean())
    j.to_csv(ROOT / "results" / "rescue_per_job.csv", index=False)
    log.info("\nwrote results/rescue_per_job.csv")


if __name__ == "__main__":
    main()
