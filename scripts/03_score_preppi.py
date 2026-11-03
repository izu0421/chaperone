"""Score PrePPI on the benchmark: the orthogonal structural baseline, and two sanity checks.

This runs before any GPU work, for three reasons.

1. **It tests the benchmark.** PrePPI's `str_score` is a structure-based PPI predictor. If a
   structural method cannot separate positives from negatives here, the benchmark is broken and
   no amount of folding will show otherwise.
2. **It measures the label leak.** `final_score` contains `exp_score` for 15.7% of PrePPI pairs,
   and those pairs all carry a literature reference. Against experimentally-derived positives
   that is leakage. Reporting `str` and `final` side by side turns a design assertion into a
   number.
3. **It measures the negative-set confound.** The same score is evaluated against random and
   degree-matched negatives. The gap is how much any method on this benchmark is flattered by
   negatives drawn from less-studied proteins.

Metric is AUCPR at the paper's 1:1000 prevalence, applied as sample weights so n does not change.

Ties are handled explicitly and reported, because `str_score` takes only 245 distinct values over
1.55M pairs. A score that is mostly ties does not rank, and average_precision_score will not say
so (feedback_standalone_auc_tie_mass: a 0.807 AUC here was one saturated threshold, 0.700 once
ties were stripped).

Pairs absent from PrePPI are the crux, and the first version of this script got it wrong.
PrePPI publishes only pairs reaching final >= 600, so absence looks like a prediction of
non-interaction, and imputing the minimum looks reasonable. It is not: `final` contains
`exp_score`, so MEMBERSHIP in PrePPI is itself set partly by experimental evidence. Imputing
absence as the floor injects that leak into `str_score`, and because 90-96% of rows then sit at
one imputed value, the resulting AUROC measures membership, not structure.

So the three effects are reported SEPARATELY and never merged into one number:

  coverage    is the pair in PrePPI at all. Reported as a statistic and as its own AUROC, with
              the fraction of covered pairs carrying an exp_score, so the circularity is visible.
  ranking     AUCPR/AUROC of each score among COVERED pairs only. This is the only honest
              measure of whether PrePPI's structural score ranks.
  imputed     the old all-pairs number, kept so the difference from `ranking` is on the record
              rather than quietly corrected away.

Usage:
    python -I scripts/03_score_preppi.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("preppi")

HERE = Path(__file__).resolve().parent.parent
SCORES = ["str_score", "protpep_score", "str_max_score", "total_score", "final_score"]


def tie_profile(x: np.ndarray) -> dict:
    """How much of this score is ties, and how much sits at its single commonest value."""
    finite = x[np.isfinite(x)]
    if not len(finite):
        return {"frac_distinct": np.nan, "mass_at_mode": np.nan}
    vals, counts = np.unique(finite, return_counts=True)
    return {"frac_distinct": len(vals) / len(finite),
            "mass_at_mode": counts.max() / len(finite)}


def evaluate(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> dict:
    ok = np.isfinite(s) & (w > 0)
    y, s, w = y[ok], s[ok], w[ok]
    out = {"n": int(ok.sum()), "n_pos": int((y == 1).sum()),
           "aucpr": average_precision_score(y, s, sample_weight=w),
           "auroc": roc_auc_score(y, s, sample_weight=w)}
    # The same metric with tied rows removed: what the score achieves where it actually ranks.
    vals, counts = np.unique(s, return_counts=True)
    tied = set(vals[counts > 1])
    keep = ~np.isin(s, list(tied)) if tied else np.ones(len(s), bool)
    if keep.sum() > 50 and len(np.unique(y[keep])) == 2:
        out["aucpr_untied"] = average_precision_score(y[keep], s[keep], sample_weight=w[keep])
        out["frac_untied"] = float(keep.mean())
    else:
        out["aucpr_untied"], out["frac_untied"] = np.nan, float(keep.mean())
    out.update(tie_profile(s))
    # The floor: AUCPR of a random ranker is the weighted positive prevalence.
    out["baseline_aucpr"] = float(w[y == 1].sum() / w.sum())
    return out


def precision_at_k(y: np.ndarray, s: np.ndarray, w: np.ndarray, ks) -> pd.DataFrame:
    """Weighted precision in the top k, which is what a screen actually cares about.

    AUROC is the wrong summary for PrePPI's str_score and says so loudly: positives have a
    higher MEAN (108.6 vs 20.2) and 90th percentile (240.9 vs 33.1) than negatives but a LOWER
    median (2.20 vs 3.18). The signal lives in the upper tail only, so a rank statistic over
    the whole set comes out below 0.5 while the top of the list is strongly enriched. The
    RF2-PPI paper reports precision at a recall level for the same reason.
    """
    o = np.argsort(-s, kind="stable")
    y, s, w = y[o], s[o], w[o]
    floor = w[y == 1].sum() / w.sum()
    rows = []
    for k in ks:
        k = min(k, len(y))
        wk, yk = w[:k], y[:k]
        prec = wk[yk == 1].sum() / wk.sum() if wk.sum() else np.nan
        rows.append({"k": k, "precision": prec, "enrichment": prec / floor,
                     "min_score_in_k": float(s[k - 1])})
    return pd.DataFrame(rows)


def main():
    d = pd.read_parquet(HERE / "bench" / "pairs.parquet")
    pp = pd.read_csv(HERE / "data" / "preppi_raw" / "preppi_final600.txt", sep="\t",
                     usecols=["prot1", "prot2", "exp_score"] + SCORES, low_memory=False)
    # PrePPI pairs are unordered but written in one arbitrary order; key both the same way.
    # numpy has no minimum/maximum loop for string dtypes, so sort the two columns row-wise.
    pair = np.sort(pp[["prot1", "prot2"]].astype(str).to_numpy(), axis=1)
    a, b = pair[:, 0], pair[:, 1]
    pp = pp.assign(prot_a=a, prot_b=b).drop(columns=["prot1", "prot2"])
    d = d.merge(pp, on=["prot_a", "prot_b"], how="left")

    cov = d.groupby("label").str_score.apply(lambda s: s.notna().mean())
    log.info("PrePPI coverage: positives %.3f, negatives %.3f",
             cov.get(1, np.nan), cov.get(0, np.nan))
    log.info("  (absence = below PrePPI's own final>=600 cut; scored as the minimum, not dropped)")

    rows, cov_rows = [], []
    for neg in ("random", "degmatched"):
        sub = d[d.neg_set.isin(["both", neg])].copy()
        w_all = sub[f"w_{neg}"].to_numpy()
        y_all = sub.label.to_numpy()
        covered = sub.str_score.notna().to_numpy()

        # --- coverage, as its own predictor, with the circularity measured -------------
        c = sub[covered]
        cov_rows.append({
            "negatives": neg,
            "cov_pos": float(covered[y_all == 1].mean()),
            "cov_neg": float(covered[y_all == 0].mean()),
            "auroc_membership_only": roc_auc_score(y_all, covered.astype(int),
                                                   sample_weight=w_all),
            "covered_pos_with_exp": float(c.loc[c.label == 1, "exp_score"].notna().mean())
            if "exp_score" in c else np.nan,
            "covered_neg_with_exp": float(c.loc[c.label == 0, "exp_score"].notna().mean())
            if "exp_score" in c else np.nan,
        })

        for sc in SCORES:
            # ranking: covered pairs only -- the honest measure
            r = evaluate(y_all[covered], sub[sc].to_numpy(dtype=float)[covered], w_all[covered])
            r.update({"negatives": neg, "score": sc, "arm": "ranking (covered only)"})
            rows.append(r)
            # imputed: the all-pairs number, kept visible
            s = sub[sc].to_numpy(dtype=float)
            s = np.where(np.isfinite(s), s, np.nanmin(s))
            r = evaluate(y_all, s, w_all)
            r.update({"negatives": neg, "score": sc, "arm": "imputed (all pairs)"})
            rows.append(r)
    res = pd.DataFrame(rows)
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(HERE / "results" / "preppi_coverage.csv", index=False)
    log.info("\n--- coverage, which is NOT a structural result ---\n%s",
             cov.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out = HERE / "results" / "preppi_baseline.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)

    cols = ["arm", "negatives", "score", "n", "aucpr", "baseline_aucpr", "auroc",
            "frac_distinct", "mass_at_mode"]
    log.info("\n%s", res.sort_values(["arm", "negatives", "score"])[cols]
             .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    r = res.set_index(["arm", "negatives", "score"])
    rank = "ranking (covered only)"
    log.info("\n--- what this says ---")
    for neg in ("random", "degmatched"):
        row = r.loc[(rank, neg, "str_score")]
        log.info("%-11s str_score among covered pairs: AUCPR %.4f (floor %.4f), AUROC %.4f",
                 neg, row.aucpr, row.baseline_aucpr, row.auroc)
    log.info("membership-only AUROC: %s",
             ", ".join(f"{t.negatives} {t.auroc_membership_only:.4f}"
                       for t in cov.itertuples()))
    leak = (r.loc[(rank, "degmatched", "final_score"), "aucpr"]
            - r.loc[(rank, "degmatched", "str_score"), "aucpr"])
    log.info("final - str among covered, degree-matched: %+.4f  <- the experimental-evidence "
             "leak; final_score is never the structural baseline", leak)
    conf = (r.loc[(rank, "random", "str_score"), "aucpr"]
            - r.loc[(rank, "degmatched", "str_score"), "aucpr"])
    log.info("random - degree-matched negatives, str_score: %+.4f  <- how much an easier "
             "negative set flatters a structural method", conf)
    imp = (r.loc[("imputed (all pairs)", "degmatched", "str_score"), "auroc"]
           - r.loc[(rank, "degmatched", "str_score"), "auroc"])
    log.info("imputed - covered-only AUROC for str_score: %+.4f  <- what imputing absence as the "
             "floor adds, and it is membership, not structure", imp)
    # The tail, for str_score under degree-matched negatives: the honest headline for PrePPI.
    sub = d[d.neg_set.isin(["both", "degmatched"])]
    c = sub[sub.str_score.notna()]
    pk = precision_at_k(c.label.to_numpy(), c.str_score.to_numpy(dtype=float),
                        c.w_degmatched.to_numpy(), [100, 500, 1000, 2000, 5000, 10000, 20000,
                                                     len(c)])
    pk.to_csv(HERE / "results" / "preppi_precision_at_k.csv", index=False)
    log.info("\n--- PrePPI str_score, top-k precision (degree-matched negatives) ---\n%s",
             pk.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log.info("str_score is a TAIL detector: the top ~1-2%% is enriched ~15x and the rest is at "
             "or below chance, which is why its AUROC is under 0.5 while its AUCPR beats the "
             "floor. Compare the folding arms on top-k precision, not AUROC.")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
