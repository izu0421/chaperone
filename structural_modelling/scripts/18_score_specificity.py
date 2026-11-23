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
sys.path.insert(0, str(ROOT / "scripts"))
import score_store as S  # noqa: E402

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
    ap.add_argument("--out", default="specificity_scores.csv")
    ap.add_argument("--full", action="store_true", help="re-score every fold")
    ap.add_argument("--allow-shrink", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report ranks and AUROC on an unfinished matrix. Off by "
                         "default: a partial matrix inflates every rank, because a "
                         "ligand's own receptor is ranked against however many "
                         "competitors happen to have landed")
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "work" / "specificity_manifest.csv")
    out_dir = ROOT / "results" / "specificity_out"
    dest = ROOT / "results" / a.out
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} does not exist; nothing has been folded yet")

    expected = man.job.astype(str).tolist()
    markers = list((ROOT / "run").glob("specificity_shard*.running")) if (ROOT / "run").is_dir() else []
    state = S.classify(out_dir, expected, True if markers else None)
    log.info("manifest %d folds: %d ready, %d being written, %d queued, %d absent",
             len(expected), len(state["ready"]), len(state["writing"]),
             len(state["queued"]), len(state["absent"]))
    if not state["ready"]:
        raise SystemExit("nothing to score yet")

    code_fp = S.code_fingerprint([ROOT / "scripts" / f for f in
                                  ("18_score_specificity.py", "08_score_armB.py",
                                   "score_store.py")])
    cache = pd.DataFrame() if a.full else S.load_cache(dest, code_fp)
    todo, reused = S.plan(state["ready"], cache)
    log.info("scoring %d folds; reusing %d cached rows (code %s)",
             len(todo), len(reused), code_fp)

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(score_one, [out_dir / n for n in todo]))
    fresh = pd.DataFrame(rows)
    if not fresh.empty:
        fresh[S.FP_COL] = fresh.pair.map(state["ready"])
        fresh[S.CODE_COL] = code_fp
    df = pd.concat([reused, fresh], ignore_index=True) if len(reused) else fresh
    df = df[df.pair.isin(state["ready"])].copy()
    df = df.drop(columns=[c for c in man.columns if c in df.columns and c != "pair"]).merge(
        man, left_on="pair", right_on="job", how="left")

    # A ligand's rank is only meaningful against the full receptor panel. Count how
    # many of each ligand's receptors are present and carry that with the frame, so
    # the rank step can drop the incomplete ones instead of flattering them.
    panel = man.groupby("ligand").job.size().rename("panel_n")
    have = df.groupby("ligand").pair.size().rename("have_n")
    cov = pd.concat([panel, have], axis=1).fillna(0)
    cov["complete"] = cov.have_n >= cov.panel_n
    df = df.merge(cov[["panel_n", "have_n", "complete"]], left_on="ligand",
                  right_index=True, how="left")
    n_complete = int(cov.complete.sum())
    log.info("%d of %d ligands have their full receptor panel folded",
             n_complete, len(cov))

    S.write(dest, df, state, code_fp, len(expected), allow_shrink=a.allow_shrink,
            extra={"ligands_with_full_panel": n_complete,
                   "ligands_total": int(len(cov)),
                   "reused_rows": int(len(reused)), "newly_scored": int(len(fresh))})

    matrix_complete = len(state["ready"]) == len(expected)
    if not matrix_complete and not a.allow_partial:
        log.warning("\nMATRIX INCOMPLETE (%d of %d folds). Scores are written, but "
                    "the ranks and AUROC below are restricted to the %d ligands whose "
                    "full panel has landed. Pass --allow-partial to score every "
                    "ligand regardless -- knowing that a ligand ranked against 6 "
                    "receptors instead of 16 looks far better than it is.",
                    len(state["ready"]), len(expected), n_complete)
    rank_pool = df if (matrix_complete or a.allow_partial) else df[df.complete]
    if rank_pool.empty:
        log.warning("no ligand has a complete panel yet; stopping before the "
                    "rank and AUROC sections")
        return

    cog = rank_pool[rank_pool.label.eq("cognate")]
    neg = rank_pool[rank_pool.label.eq("negative")]
    amb = rank_pool[rank_pool.label.eq("ambiguous")]
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
        ranks, panel_sizes = [], []
        for lig, g in rank_pool.groupby("ligand"):
            g = g[np.isfinite(g[metric])]
            if g.empty or not g.label.eq("cognate").any():
                continue
            ordered = g.sort_values(metric, ascending=False).reset_index(drop=True)
            ranks.append(int(ordered.index[ordered.label.eq("cognate")][0]) + 1)
            # the number actually competed against, not the number intended: a
            # metric that cannot be computed on some folds shrinks the panel too
            panel_sizes.append(len(g))
        if ranks:
            rank_rows.append({"metric": metric, "n_ligands": len(ranks),
                              "median_panel": float(np.median(panel_sizes)),
                              "chance_median_rank": round((np.mean(panel_sizes) + 1) / 2, 2),
                              "median_rank": float(np.median(ranks)),
                              "mean_rank": round(float(np.mean(ranks)), 2),
                              "top1": int(sum(r == 1 for r in ranks)),
                              "exp_top1": round(sum(1 / n for n in panel_sizes), 2),
                              "top3": int(sum(r <= 3 for r in ranks)),
                              "exp_top3": round(sum(min(3, n) / n for n in panel_sizes), 2)})
    ranks_df = pd.DataFrame(rank_rows)
    # chance is computed from the panel each ligand actually competed in, never
    # from the 16 the design intended: those differ whenever the matrix is partial
    log.info("chance expectations are per-metric (chance_median_rank, exp_top1, "
             "exp_top3), derived from the real panel sizes\n%s",
             ranks_df.to_string(index=False) if len(ranks_df) else "  none")

    log.info("\n--- AUROC, cognate vs ChEMBL-clean negatives ---")
    auc_rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        p, n = cog[metric].to_numpy(float), neg[metric].to_numpy(float)
        # `auroc()` silently drops non-finite values, so the AUROC for a metric
        # that is missing on some folds is measured on fewer pairs than the
        # design has -- while `floor` was computed once from the full cognate and
        # negative counts. Comparing the two marks a metric as beating a bar it
        # was never tested against. The floor belongs to the sample that produced
        # the number, so it is recomputed per metric, and the n goes in the table
        # so the comparison can be seen rather than trusted.
        pf, nf = p[np.isfinite(p)], n[np.isfinite(n)]
        same = neg[neg.same_family][metric].to_numpy(float)
        diff = neg[~neg.same_family][metric].to_numpy(float)
        a = auroc(p, n)
        lo, hi = boot_auroc(p, n)
        m_floor = detection_floor(len(pf), len(nf)) if len(pf) and len(nf) else np.nan
        # with very few positives no AUROC is interpretable; say so instead of
        # answering False, which reads as "tested and did not beat it"
        beats = (np.nan if len(pf) < 8 or not np.isfinite(a) or not np.isfinite(m_floor)
                 else a >= m_floor)
        auc_rows.append({
            "metric": metric, "auroc": round(a, 3),
            "n_pos": int(len(pf)), "n_neg": int(len(nf)),
            "dropped_pos": int(len(p) - len(pf)), "dropped_neg": int(len(n) - len(nf)),
            "floor": round(m_floor, 3) if np.isfinite(m_floor) else np.nan,
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "within_family": round(auroc(p, same), 3),
            "n_within": int(np.isfinite(same).sum()),
            "across_family": round(auroc(p, diff), 3),
            "n_across": int(np.isfinite(diff).sum()),
            "vs_ambiguous": round(auroc(p, amb[metric].to_numpy(float)), 3)
            if len(amb) else np.nan,
            "beats_floor": beats})
    auc_df = pd.DataFrame(auc_rows).sort_values("auroc", ascending=False)
    log.info("\n%s", auc_df.to_string(index=False))
    thin = auc_df[auc_df.n_pos.lt(len(cog))]
    if len(thin):
        log.warning("%d metrics were measured on fewer cognates than the %d available, "
                    "because the metric is missing on some folds. Their floors differ "
                    "from the design's; read the per-row floor, not the headline one:\n%s",
                    len(thin), len(cog),
                    thin[["metric", "n_pos", "n_neg", "floor", "auroc"]].to_string(index=False))

    log.info("\n--- the flatness question: spread of each metric across the matrix ---")
    flat = df.groupby("label")[[m for m in METRICS if m in df.columns]].median().round(3)
    log.info("\n%s", flat.to_string())

    ranks_df.to_csv(ROOT / "results" / "specificity_ranks.csv", index=False)
    auc_df.to_csv(ROOT / "results" / "specificity_auroc.csv", index=False)
    log.info("\nwrote results/specificity_ranks.csv and results/specificity_auroc.csv")


if __name__ == "__main__":
    main()
