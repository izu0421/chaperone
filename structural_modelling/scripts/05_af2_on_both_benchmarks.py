"""Their own AF2 scores, evaluated on their benchmark and on a degree-matched one.

This needs no GPU. `AF_scores.gz` is 3,381,661 pairs with a column literally named
`AF2_interaction_probability` and a `Source` column whose `NEG` rows are "negative controls used
for accuracy estimation". So the published AF2 arm can be reproduced from the deposit, and then
re-scored against harder negatives, without rerunning anything.

Two questions, both free:

1. **Does the harness reproduce their result?** AF2 on their own positives and negatives should
   land near the published curve. Nothing downstream is interpretable until it does.

2. **How much of the AF2 + omicMSA advantage is the degree confound?** Their negatives have mean
   pair degree 4.1 against their positives' 37.5, measured on an independent graph
   (scripts/02 and the design). The SAME AF2 scores are therefore evaluated three ways:

       their_benchmark        their positives vs all their negatives     (the published setting)
       their_bench_degmatched the same pairs, with the negatives subsampled to match the
                              positives' degree distribution

   The test is done INSIDE their benchmark, and that is not a stylistic choice. Their AF2
   scores cover their own benchmark completely (1.000 of positives and 1.000 of negatives) but
   only 88% of our positives and 2-7% of our negatives, because AF2 was run on whatever their
   DCA -> RF2-PPI cascade shortlisted PLUS, per the file's own header, every pair "from PPI
   databases, BioGrid, STRING (physical), and UniProt". Whether a pair has an AF2 score is
   therefore itself a restatement of whether it is in a PPI database -- the same membership
   leak that made PrePPI look skilful in scripts/03. Scoring our benchmark with these numbers
   would measure that leak, so it is not done.

   Within their benchmark the comparison is clean: identical pairs, identical scores, the only
   change being which negatives are kept. A method that genuinely detects interaction should
   lose little. A method partly detecting "this protein is well studied" -- which is what deeper
   omicMSAs encode, since alignment depth tracks how much sequence exists for a protein's
   homologs -- should lose a lot.

Note on what this can and cannot show: AF2 here comes with omicMSAs, so this measures the
combination, not omicMSA alone. It still bounds how much of the published number survives a
harder negative set, which is the claim at issue.

Usage:
    python -I scripts/05_af2_on_both_benchmarks.py
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("af2")

HERE = Path(__file__).resolve().parent.parent
PREVALENCE = 1 / 1000


def load_af2() -> pd.DataFrame:
    f = HERE / "data" / "rf2ppi_raw" / "AF_scores.gz"
    d = pd.read_csv(f, sep="\t", comment="#")
    d.columns = ["pair", "af2", "source"]
    sp = d.pair.str.split("_", expand=True)
    # Their pair ids are written in one arbitrary order; key unordered so a join cannot miss.
    k = np.sort(sp[[0, 1]].astype(str).to_numpy(), axis=1)
    d["prot_a"], d["prot_b"] = k[:, 0], k[:, 1]
    log.info("AF2 scores: %d pairs, sources %s", len(d), d.source.value_counts().to_dict())
    return d


def topk(y, s, w, ks=(100, 500, 1000, 5000)):
    o = np.argsort(-s, kind="stable")
    y, s, w = y[o], s[o], w[o]
    floor = w[y == 1].sum() / w.sum()
    return {f"prec@{k}": (w[:k][y[:k] == 1].sum() / w[:k].sum()) for k in ks if k <= len(y)}, floor


def evaluate(y, s, w, label) -> dict:
    ok = np.isfinite(s)
    y, s, w = y[ok], s[ok], w[ok]
    tk, floor = topk(y, s, w)
    out = {"set": label, "n": int(len(y)), "n_pos": int((y == 1).sum()),
           "aucpr": average_precision_score(y, s, sample_weight=w),
           "floor": floor,
           "auroc": roc_auc_score(y, s, sample_weight=w)}
    out["lift"] = out["aucpr"] / floor
    out.update({k: v for k, v in tk.items()})
    return out


def main():
    af2 = load_af2()[["prot_a", "prot_b", "af2", "source"]]
    rows = []

    # --- 1. their benchmark, their scores -----------------------------------
    b = pd.read_csv(HERE / "data/rf2ppi_raw/benchmarks/positives_and_negatives.tsv", sep="\t")
    b.columns = ["pair", "category"]
    sp = b.pair.str.split("_", expand=True)
    k = np.sort(sp[[0, 1]].astype(str).to_numpy(), axis=1)
    b["prot_a"], b["prot_b"] = k[:, 0], k[:, 1]
    b["label"] = b.category.eq("positive").astype(int)
    b = b.merge(af2[["prot_a", "prot_b", "af2"]].drop_duplicates(["prot_a", "prot_b"]),
                on=["prot_a", "prot_b"], how="left")
    cov = b.groupby("label").af2.apply(lambda s: s.notna().mean())
    log.info("AF2 coverage of their own benchmark: positives %.3f, negatives %.3f",
             cov.get(1, np.nan), cov.get(0, np.nan))
    sub = b[b.af2.notna()]
    # their own 1:1000 weighting: positives x0.01 relative to a 1:10 set
    w = np.where(sub.label == 1,
                 PREVALENCE * (sub.label == 0).sum() / max((sub.label == 1).sum(), 1), 1.0)
    rows.append(evaluate(sub.label.to_numpy(), sub.af2.to_numpy(), w, "their_benchmark"))

    # Show the coverage that rules out scoring our own benchmark with these numbers.
    mine = pd.read_parquet(HERE / "bench" / "pairs.parquet").merge(
        af2[["prot_a", "prot_b", "af2"]].drop_duplicates(["prot_a", "prot_b"]),
        on=["prot_a", "prot_b"], how="left")
    for neg in ("random", "degmatched"):
        c = mine[mine.neg_set.isin(["both", neg])].groupby("label").af2.apply(
            lambda x: x.notna().mean())
        log.info("AF2 coverage of OUR %s arm: positives %.3f, negatives %.3f "
                 "-- too asymmetric to score on, see the module docstring",
                 neg, c.get(1, np.nan), c.get(0, np.nan))

    # --- 2. degree-matched subsample of THEIR negatives ---------------------
    # Degree comes from our independent positive graph (BioGRID MV n STRING>700), which shares
    # no construction with their benchmark, so the matching variable is not derived from the
    # thing being matched.
    ours = pd.read_parquet(HERE / "bench" / "pairs.parquet")
    op = ours[ours.label == 1]
    deg = pd.concat([op.prot_a, op.prot_b]).value_counts()
    pd_pair = lambda d: (d.prot_a.map(deg).fillna(0) + d.prot_b.map(deg).fillna(0)).to_numpy()
    sub = sub.copy()
    sub["pdeg"] = pd_pair(sub)
    pos_d, neg_d = sub[sub.label == 1], sub[sub.label == 0]
    log.info("inside their benchmark: mean pair degree positives %.2f, negatives %.2f",
             pos_d.pdeg.mean(), neg_d.pdeg.mean())

    # Match by sampling negatives within decile bins of the POSITIVES' degree distribution.
    rng = np.random.default_rng(0)
    edges = np.unique(np.quantile(pos_d.pdeg, np.linspace(0, 1, 11)))
    keep = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        npos = int(((pos_d.pdeg >= lo) & (pos_d.pdeg < hi)).sum())
        cand = neg_d.index[(neg_d.pdeg >= lo) & (neg_d.pdeg < hi)].to_numpy()
        take = min(len(cand), npos * 10)          # hold the 1:10 ratio their set uses
        if take:
            keep.extend(rng.choice(cand, take, replace=False))
    m = pd.concat([pos_d, neg_d.loc[keep]])
    log.info("degree-matched subsample of their benchmark: %d positives, %d negatives "
             "(mean pair degree %.2f vs %.2f)",
             int((m.label == 1).sum()), int((m.label == 0).sum()),
             m[m.label == 1].pdeg.mean(), m[m.label == 0].pdeg.mean())
    wm = np.where(m.label == 1,
                  PREVALENCE * (m.label == 0).sum() / max((m.label == 1).sum(), 1), 1.0)
    rows.append(evaluate(m.label.to_numpy(), m.af2.to_numpy(), wm, "their_bench_degmatched"))

    res = pd.DataFrame(rows)
    res.to_csv(HERE / "results" / "af2_both_benchmarks.csv", index=False)
    log.info("\n%s", res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    r = res.set_index("set")
    if {"their_benchmark", "their_bench_degmatched"} <= set(r.index):
        a, b_ = r.loc["their_benchmark", "aucpr"], r.loc["their_bench_degmatched", "aucpr"]
        log.info("\nAF2 AUCPR on their benchmark: %.4f -> %.4f once their negatives are "
                 "degree-matched to their positives (%+.1f%%)", a, b_, 100 * (b_ - a) / a)
        log.info("AUROC: %.4f -> %.4f", r.loc["their_benchmark", "auroc"],
                 r.loc["their_bench_degmatched", "auroc"])
        log.info("Same pairs, same scores; only which negatives are kept has changed, so the "
                 "difference is the part of the published number that comes from negatives "
                 "drawn from less-studied proteins.")
    log.info("wrote %s", HERE / "results" / "af2_both_benchmarks.csv")


if __name__ == "__main__":
    main()
