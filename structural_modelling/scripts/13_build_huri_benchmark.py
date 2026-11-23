"""An independent validation benchmark from HuRI, folded fresh.

Why a new set at all. Every number so far rests on 2,500 pairs chosen and labelled by
the RF2-PPI authors. PrePPI corroborates those labels (99.4%), but corroboration of
existing labels is not an independent test: the pairs are the same, the folds are the
same, and the literature bias that selected them is the same.

HuRI (Luck et al., Nature 2020) is the one source that breaks that. It is a systematic
all-by-all Y2H screen of the human ORFeome, so WHICH pairs were tested does not depend
on which proteins anyone has studied. Its mean pair degree in our independent literature
graph is 6.1, against 37.5 for the RF2-PPI benchmark positives.

Positives   HI-union pairs, mapped to reviewed human UniProt, heteromeric.

Negatives   the part that makes this worth doing. Both proteins must THEMSELVES appear
            in HI-union -- i.e. each one was in the ORFeome, was expressed, and scored
            at least one interaction of its own -- but the PAIR is absent. So absence
            is a screened rejection by a competent assay, not an untested gap, and the
            "negative protein is an unstudied protein" confound is gone by construction.
            Y2H sensitivity is ~25-35%, so absence is still noisy; that is a stated
            limit of the set, not a hidden one.

Matching    negatives are degree-matched to positives on HuRI degree, with the weights
            and both distributions printed (feedback_stratum_matching_needs_weights: a
            thin pool makes a "matched" background short in exactly the strata that
            carry the confound).

Usage:
    python -I scripts/13_build_huri_benchmark.py --n-pos 500 --n-neg 2000
"""
from __future__ import annotations
import argparse, collections, gzip, logging
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("huri")
HERE = Path(__file__).resolve().parent.parent
SEED = 0


def ensg_to_acc() -> dict[str, str]:
    """ENSG -> one reviewed UniProt accession. Where several accessions map to the same
    gene the lexicographically first is taken, deterministically, so the set is
    reproducible; isoform choice is not a variable this benchmark is testing."""
    m: dict[str, set[str]] = collections.defaultdict(set)
    with gzip.open(HERE / "data/uniprot_raw/acc_to_ensg.tsv.gz", "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2 or not p[1]:
                continue
            for g in p[1].replace(";", " ").split():
                m[g.split(".")[0]].add(p[0])
    return {g: sorted(a)[0] for g, a in m.items() if a}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=500)
    ap.add_argument("--n-neg", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    g2a = ensg_to_acc()
    hi = pd.read_csv(HERE / "data/huri_raw/HI-union.tsv", sep="\t", header=None,
                     names=["g1", "g2"], dtype=str)
    log.info("HI-union rows: %d", len(hi))

    hi["a"] = hi.g1.map(g2a); hi["b"] = hi.g2.map(g2a)
    hi = hi.dropna(subset=["a", "b"])
    hi = hi[hi.a != hi.b]                      # heteromeric only
    pos = {tuple(sorted((x, y))) for x, y in zip(hi.a, hi.b)}
    log.info("mapped heteromeric UniProt pairs: %d", len(pos))

    # proteins that scored at least one interaction of their own: assay-competent
    comp = sorted({p for k in pos for p in k})
    log.info("assay-competent proteins (>=1 HI-union interaction): %d", len(comp))

    msa = {f.stem for f in (HERE / "data/rf2ppi_raw/protein_omicMSAs/alns").glob("*.a3m")}
    log.info("omicMSAs available: %d  | competent proteins covered: %d (%.1f%%)",
             len(msa), sum(p in msa for p in comp),
             100 * sum(p in msa for p in comp) / len(comp))

    # restrict everything to proteins arm B can actually run
    comp = [p for p in comp if p in msa]
    compset = set(comp)
    pos = {k for k in pos if k[0] in compset and k[1] in compset}
    log.info("positives with omicMSAs for BOTH proteins: %d", len(pos))

    deg = collections.Counter()
    for x, y in pos:
        deg[x] += 1; deg[y] += 1

    # candidate negatives: competent x competent, pair absent from HI-union
    want = a.n_neg * 40
    cand: set[tuple[str, str]] = set()
    arr = np.array(comp)
    while len(cand) < want:
        x = rng.choice(arr, want); y = rng.choice(arr, want)
        for u, v in zip(x, y):
            if u == v:
                continue
            k = (u, v) if u < v else (v, u)
            if k not in pos:
                cand.add(k)
            if len(cand) >= want:
                break
    cand = sorted(cand)
    log.info("candidate screened-negative pairs: %d", len(cand))

    def pdeg(k): return (deg[k[0]] + deg[k[1]]) / 2
    P = sorted(pos); rng.shuffle(P); P = P[:a.n_pos]
    edges = np.array([0, 1.5, 2.5, 4, 7, 12, 25, 45, 70, 110, 170, 260, 1e9])
    pb = np.digitize([pdeg(k) for k in P], edges) - 1
    cb = np.digitize([pdeg(k) for k in cand], edges) - 1
    target = np.bincount(pb, minlength=len(edges) - 1) / len(P)

    log.info("\n%14s %6s %10s %8s %8s", "HuRI pair deg", "pos", "neg pool", "want", "take")
    take: list[tuple[str, str]] = []
    for i in range(len(edges) - 1):
        pool = [k for k, bb in zip(cand, cb) if bb == i]
        want_i = int(round(target[i] * a.n_neg))
        got = min(want_i, len(pool))
        if got:
            idx = rng.choice(len(pool), got, replace=False)
            take += [pool[int(t)] for t in idx]
        lab = f"{edges[i]:.1f}-{edges[i+1]:.0f}" if i < len(edges) - 2 else f">{edges[i]:.0f}"
        log.info("%14s %6d %10d %8d %8d", lab, int(target[i] * len(P)), len(pool), want_i, got)

    out = pd.DataFrame(
        [{"pair": f"{x}_{y}", "cat": "positive", "a": x, "b": y, "pdeg": pdeg((x, y))} for x, y in P]
        + [{"pair": f"{x}_{y}", "cat": "negative", "a": x, "b": y, "pdeg": pdeg((x, y))} for x, y in take])
    log.info("\nmean HuRI pair degree  positives %.2f  negatives %.2f",
             out[out.cat == "positive"].pdeg.mean(), out[out.cat == "negative"].pdeg.mean())
    log.info("built %d pairs (%d pos / %d neg)", len(out),
             (out.cat == "positive").sum(), (out.cat == "negative").sum())
    dst = HERE / "bench/huri_pairs.csv"
    out.to_csv(dst, index=False)
    log.info("wrote %s", dst)


if __name__ == "__main__":
    main()
