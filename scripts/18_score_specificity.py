"""Score the specificity matrix and ask whether any confidence metric knows
which receptor a small molecule belongs to.

Read-outs, in the order they should be believed:

  per-ligand rank   where a ligand's own receptor lands among the 16 it was
                    folded against. The deorphanisation question asked
                    directly, and better powered than the pooled AUROC with
                    only 16 cognate pairs. DeorphaNN switched to a per-
                    receptor statistic for the same reason: a matrix-wide ROC
                    flatters promiscuous receptors.
  AUROC             cognate against ChEMBL-clean negatives, per metric, with
                    a bootstrap interval over pairs.
  stratified        the same split within and across chemotype family.
                    Within-family is the hard case and the one a real
                    deorphanisation faces.

The `ambiguous` cells -- off-diagonal pairs with measured ChEMBL activity --
are reported on their own and never pooled into the negatives.

The detection floor is printed BEFORE any result, because a null is only
evidence with its minimum detectable effect attached: with 16 positives the
pooled AUROC cannot resolve a real 0.6 from chance, and saying so afterwards
would be an excuse instead of a design note.

Usage:
    micromamba run -n structmod python -I scripts/18_score_specificity.py --workers 24
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("spec-score")
ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("sc14", ROOT / "scripts" / "14_score_orphans.py")
sc14 = importlib.util.module_from_spec(_spec)
sys.modules["sc14"] = sc14
_spec.loader.exec_module(sc14)
score_one = sc14.score_one

# higher is better for all of these; interface PAE and min separation are
# negated at use so that every metric points the same way
METRICS = ["iptm", "iptm_AB", "ranking_score", "plddt_B", "iface_plddt_B",
           "iface_plddt", "plddt_A", "bsa", "n_contact"]


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC, ties counted as half."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def boot_auroc(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan, np.nan
    vals = [auroc(rng.choice(pos, len(pos), replace=True),
                  rng.choice(neg, len(neg), replace=True)) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def detection_floor(n_pos: int, n_neg: int) -> float:
    """Smallest AUROC this design could call apart from chance at 80% power.

    Hanley-McNeil standard error under the null, two-sided alpha 0.05.
    """
    a = 0.5
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    se = np.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a ** 2)
                  + (n_neg - 1) * (q2 - a ** 2)) / (n_pos * n_neg))
    return float(0.5 + (1.96 + 0.84) * se)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "work" / "specificity_manifest.csv")
    out_dir = ROOT / "results" / "specificity_out"
    dirs = sorted(d for d in out_dir.iterdir() if d.is_dir()) if out_dir.exists() else []
    log.info("%d of %d folds complete", len(dirs), len(man))
    if not dirs:
        log.error("nothing to score yet"); return

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(score_one, dirs))
    df = pd.DataFrame(rows).merge(man, left_on="pair", right_on="job", how="left")
    df.to_csv(ROOT / "results" / "specificity_scores.csv", index=False)
    log.info("wrote results/specificity_scores.csv")

    cog = df[df.label.eq("cognate")]
    neg = df[df.label.eq("negative")]
    amb = df[df.label.eq("ambiguous")]
    log.info("scored: %d cognate, %d clean negative, %d ambiguous",
             len(cog), len(neg), len(amb))

    floor = detection_floor(len(cog), len(neg))
    log.info("\nDETECTION FLOOR: with %d cognate against %d negatives this design "
             "can separate chance from AUROC >= %.3f at 80%% power.\n"
             "An AUROC below that is NOT evidence of specificity, and NOT evidence "
             "against it either.", len(cog), len(neg), floor)

    log.info("\n--- per-ligand rank of its own receptor among the 16 folded ---")
    rank_rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        ranks = []
        for lig, g in df.groupby("ligand"):
            g = g[np.isfinite(g[metric])]
            if g.empty or not g.label.eq("cognate").any():
                continue
            ordered = g.sort_values(metric, ascending=False).reset_index(drop=True)
            ranks.append(int(ordered.index[ordered.label.eq("cognate")][0]) + 1)
        if ranks:
            rank_rows.append({"metric": metric, "n_ligands": len(ranks),
                              "median_rank": float(np.median(ranks)),
                              "mean_rank": round(float(np.mean(ranks)), 2),
                              "top1": int(sum(r == 1 for r in ranks)),
                              "top3": int(sum(r <= 3 for r in ranks))})
    ranks_df = pd.DataFrame(rank_rows)
    log.info("chance: median rank 8.5, top1 %.1f of %d, top3 %.1f of %d\n%s",
             len(cog) / 16, len(cog), 3 * len(cog) / 16, len(cog),
             ranks_df.to_string(index=False) if len(ranks_df) else "  none")

    log.info("\n--- AUROC, cognate vs ChEMBL-clean negatives ---")
    auc_rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        p, n = cog[metric].to_numpy(float), neg[metric].to_numpy(float)
        lo, hi = boot_auroc(p, n)
        same = neg[neg.same_family]
        diff = neg[~neg.same_family]
        auc_rows.append({
            "metric": metric, "auroc": round(auroc(p, n), 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "within_family": round(auroc(p, same[metric].to_numpy(float)), 3),
            "across_family": round(auroc(p, diff[metric].to_numpy(float)), 3),
            "vs_ambiguous": round(auroc(p, amb[metric].to_numpy(float)), 3)
            if len(amb) else np.nan,
            "beats_floor": auroc(p, n) >= floor})
    auc_df = pd.DataFrame(auc_rows).sort_values("auroc", ascending=False)
    log.info("\n%s", auc_df.to_string(index=False))

    log.info("\n--- the flatness question: spread of each metric across the matrix ---")
    flat = df.groupby("label")[[m for m in METRICS if m in df.columns]].median().round(3)
    log.info("\n%s", flat.to_string())

    ranks_df.to_csv(ROOT / "results" / "specificity_ranks.csv", index=False)
    auc_df.to_csv(ROOT / "results" / "specificity_auroc.csv", index=False)
    log.info("\nwrote results/specificity_ranks.csv and results/specificity_auroc.csv")


if __name__ == "__main__":
    main()
