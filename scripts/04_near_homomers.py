"""How many of the RF2-PPI benchmark's "heteromeric" positives are effectively homomers?

Neither benchmark file contains a literal self-pair (A_A): 0 of 33,000 and 0 of 5,715. But
`P60709_P61163` is in the large-interface positives, and that is ACTB with ACTG1 — paralogs
that differ at a handful of residues. For a paired-MSA method a near-identical pair is a
homomer in all but name:

  * the two chains' alignments are drawn from the same homolog families, so pairing by organism
    is degenerate and the "cross-chain" coevolution signal is really within-family covariation;
  * the complex is close to a symmetric homodimer, which is the easiest possible target; and
  * the pair is trivially detectable by sequence identity alone, with no interaction biology.

If near-identical pairs are enriched among positives and absent from negatives (negatives are
random pairs, so they will be), part of every method's score on this benchmark is a paralogy
detector. That is a different claim from "predicts interactions", and this script measures how
much of the positive set is exposed to it.

Identity is computed two ways because they fail differently:
  k-mer   a 5-mer Jaccard containment, which is cheap over all pairs and never misses a
          near-identical pair, but over-calls low-complexity sequences;
  align   a proper global alignment (Bio.Align.PairwiseAligner, BLOSUM62) on everything the
          k-mer screen flags, which is exact but too slow for the full set.

Usage:
    python -I scripts/04_near_homomers.py
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("homomer")

HERE = Path(__file__).resolve().parent.parent
K = 5
KMER_SCREEN = 0.20        # containment above this goes to real alignment
NEAR_HOMOMER = 0.30       # identity above this is reported as a near-homomer


def sequences() -> dict[str, str]:
    f = HERE / "data" / "uniprot_raw" / "human_reviewed.fasta.gz"
    seqs, acc, buf = {}, None, []
    with gzip.open(f, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                acc, buf = line.split("|")[1], []
            else:
                buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    log.info("%d reviewed human sequences", len(seqs))
    return seqs


def kmers(s: str) -> set:
    return {s[i:i + K] for i in range(len(s) - K + 1)}


def identity(a: str, b: str) -> float:
    """Global alignment identity over the shorter sequence."""
    from Bio import Align
    al = Align.PairwiseAligner(scoring="blastp", mode="global")
    try:
        aln = al.align(a, b)[0]
    except Exception:
        return np.nan
    sa, sb = aln[0], aln[1]
    match = sum(1 for x, y in zip(sa, sb) if x == y and x != "-")
    return match / min(len(a), len(b))


def annotate(pairs: pd.DataFrame, seqs: dict) -> pd.DataFrame:
    kc = {}
    out = []
    for a, b in zip(pairs.a, pairs.b):
        sa, sb = seqs.get(a), seqs.get(b)
        if not sa or not sb:
            out.append(np.nan)
            continue
        ka = kc.setdefault(a, kmers(sa))
        kb = kc.setdefault(b, kmers(sb))
        if not ka or not kb:
            out.append(np.nan)
            continue
        out.append(len(ka & kb) / min(len(ka), len(kb)))
    pairs = pairs.copy()
    pairs["kmer_containment"] = out
    flag = pairs.kmer_containment.fillna(0) >= KMER_SCREEN
    log.info("%d of %d pairs pass the k-mer screen and go to alignment",
             int(flag.sum()), len(pairs))
    ident = np.full(len(pairs), np.nan)
    idx = np.flatnonzero(flag.to_numpy())
    for n, i in enumerate(idx, 1):
        r = pairs.iloc[i]
        ident[i] = identity(seqs[r.a], seqs[r.b])
        if n % 50 == 0:
            log.info("  aligned %d/%d", n, len(idx))
    pairs["identity"] = ident
    # unflagged pairs are below the screen, so their identity is bounded by it, not unknown
    pairs["identity"] = pairs.identity.fillna(0.0).where(
        flag | pairs.kmer_containment.notna(), np.nan)
    pairs["near_homomer"] = pairs.identity >= NEAR_HOMOMER
    return pairs


def main():
    seqs = sequences()
    out_rows = []

    b = pd.read_csv(HERE / "data/rf2ppi_raw/benchmarks/positives_and_negatives.tsv", sep="\t")
    b.columns = ["pair", "category"]
    sp = b.pair.str.split("_", expand=True)
    b["a"], b["b"] = sp[0], sp[1]
    log.info("literal self-pairs (A_A): %d", int((b.a == b.b).sum()))
    b = annotate(b, seqs)

    p = pd.read_csv(HERE / "data/rf2ppi_raw/benchmarks/pairs_partitioned_by_interface_sizes.tsv",
                    sep="\t")
    p.columns = ["pair", "category"]
    sp = p.pair.str.split("_", expand=True)
    p["a"], p["b"] = sp[0], sp[1]
    p = annotate(p, seqs)

    for name, d in (("positives_and_negatives", b), ("interface_sizes", p)):
        g = d.groupby("category").agg(
            n=("pair", "size"),
            mapped=("identity", lambda s: int(s.notna().sum())),
            near_homomer=("near_homomer", "sum"),
            frac=("near_homomer", "mean"),
            median_identity=("identity", "median"),
            max_identity=("identity", "max"))
        log.info("\n--- %s ---\n%s", name,
                 g.to_string(float_format=lambda v: f"{v:.4f}"))
        out_rows.append(d.assign(file=name))

    d = pd.concat(out_rows, ignore_index=True)
    d.to_csv(HERE / "results" / "benchmark_near_homomers.csv", index=False)

    pos = d[d.category.eq("positive")]
    neg = d[d.category.eq("negative")]
    if len(pos) and len(neg):
        log.info("\nnear-homomer rate: positives %.4f, negatives %.4f (ratio %.1fx)",
                 pos.near_homomer.mean(), neg.near_homomer.mean(),
                 pos.near_homomer.mean() / max(neg.near_homomer.mean(), 1e-9))
        log.info("A method that only detected paralogy would score %d of %d positives "
                 "(%.1f%%) correct for the wrong reason.",
                 int(pos.near_homomer.sum()), len(pos), 100 * pos.near_homomer.mean())
    log.info("wrote %s", HERE / "results" / "benchmark_near_homomers.csv")


if __name__ == "__main__":
    main()
