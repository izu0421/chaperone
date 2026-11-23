"""Build the experimental ground-truth pair sets the structural arms are scored against.

Positives follow Zhang et al. (Science 2025) so the numbers here are comparable to theirs:
the intersection of BioGRID multi-validated physical and STRING physical with combined_score
> 700, both restricted to human and mapped to UniProt accessions. (Their third source is
UniProt's own curated interactions; a 2-of-2 intersection of the two strictest sources is
reported here and the UniProt arm is added separately, so the effect of dropping it is
measurable instead of assumed.)

Negatives are the part that decides whether the benchmark means anything. Two sets are built:

  random      pairs with no evidence in either source. What the paper used.
  degmatched  the same, but matched on each protein's DEGREE in the positive set.

A hub protein appears in many positives, so a random negative pair is drawn from proteins that
are on average far less studied and less connected than a positive pair's. Every method then
looks better than it is, and the flattery is not uniform across methods -- it rewards whatever
correlates with being well studied, which for an MSA-based method is alignment depth. This
project has already paid for an unmatched control once: an abundance-confounded negative set
produced a clean, replicating, WRONG mechanism, and matching moved the headline from 22% to 5.7%
(feedback_negative_control_matching). Both arms are kept and both are reported.

Prevalence is set at 1:1000 by weight, not by subsampling, matching the paper's x0.01 positive
weighting. AUCPR is the metric; AUROC at 1:1000 is close to uninformative.

Usage:
    python -I scripts/01_build_benchmark.py
"""
from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bench")

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
BENCH = HERE / "bench"
STRING_CUT = 700
HUMAN = 9606
NEG_PER_POS = 20          # how many negatives to BUILD; prevalence is applied as a weight later
SEED = 0


def key(a: str, b: str) -> tuple[str, str]:
    """Unordered pair key. A PPI has no direction, and leaving the two orders distinct would
    let the same pair sit in positives and negatives at once."""
    return (a, b) if a <= b else (b, a)


def string_uniprot_map() -> dict[str, set[str]]:
    """STRING protein id -> UniProt accessions, from the UniProt_AC alias rows only.

    One STRING id can carry several accessions (isoforms), so this is deliberately a set: taking
    the first would silently drop interactions whose partner happens to be mapped under the
    other accession.
    """
    f = DATA / "string_raw" / "9606.protein.aliases.v12.0.txt.gz"
    m: dict[str, set[str]] = {}
    with gzip.open(f, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            sid, alias, source = line.rstrip("\n").split("\t")
            if source == "UniProt_AC":
                m.setdefault(sid, set()).add(alias.split("-")[0])
    log.info("STRING->UniProt: %d of STRING's protein ids carry a UniProt_AC", len(m))
    return m


def string_pairs() -> set[tuple[str, str]]:
    f = DATA / "string_raw" / "9606.protein.physical.links.v12.0.txt.gz"
    m = string_uniprot_map()
    out, n, unmapped = set(), 0, 0
    with gzip.open(f, "rt") as fh:
        next(fh)
        for line in fh:
            p1, p2, score = line.split()
            if int(score) <= STRING_CUT:
                continue
            n += 1
            a, b = m.get(p1), m.get(p2)
            if not a or not b:
                unmapped += 1
                continue
            for x in a:
                for y in b:
                    if x != y:
                        out.add(key(x, y))
    log.info("STRING physical > %d: %d links, %d unmapped, %d UniProt pairs",
             STRING_CUT, n, unmapped, len(out))
    return out


def biogrid_pairs() -> set[tuple[str, str]]:
    """BioGRID multi-validated physical, human-human only, via SWISS-PROT accessions.

    Human-human is enforced on BOTH interactors: the MV file is multi-organism and a
    human-yeast pair mapped on accession alone would enter the benchmark as a human PPI.
    """
    f = next((DATA / "biogrid_raw").glob("BIOGRID-MV-Physical-*.tab3.txt"))
    cols = ["Organism ID Interactor A", "Organism ID Interactor B",
            "SWISS-PROT Accessions Interactor A", "SWISS-PROT Accessions Interactor B",
            "Experimental System Type", "Publication Source"]
    d = pd.read_csv(f, sep="\t", usecols=cols, low_memory=False, na_values=["-"])
    log.info("BioGRID MV: %d rows", len(d))
    d = d[d["Experimental System Type"].eq("physical")]
    d = d[d["Organism ID Interactor A"].eq(HUMAN) & d["Organism ID Interactor B"].eq(HUMAN)]
    d = d.dropna(subset=["SWISS-PROT Accessions Interactor A",
                         "SWISS-PROT Accessions Interactor B"])
    log.info("  physical + human-human + accession on both sides: %d rows", len(d))
    out = set()
    for a, b in zip(d["SWISS-PROT Accessions Interactor A"],
                    d["SWISS-PROT Accessions Interactor B"]):
        for x in str(a).split("|"):
            for y in str(b).split("|"):
                x, y = x.strip().split("-")[0], y.strip().split("-")[0]
                if x and y and x != y:
                    out.add(key(x, y))
    log.info("  -> %d UniProt pairs", len(out))
    return out


def build_negatives(universe: list[str], positives: set, pos_degree: dict,
                    n_target: int, rng, degree_matched: bool) -> set:
    """Pairs with no evidence, either drawn uniformly or matched on positive-set degree.

    Degree matching is done by binning proteins on their degree among the positives and drawing
    each negative from the same bin pair as a real positive. That reproduces the positive set's
    joint degree distribution, which uniform sampling does not: 'how studied is this protein'
    is the strongest nuisance variable in any PPI benchmark.
    """
    out: set = set()
    if not degree_matched:
        arr = np.asarray(universe)
        while len(out) < n_target:
            a = rng.choice(arr, n_target)
            b = rng.choice(arr, n_target)
            for x, y in zip(a, b):
                if x != y:
                    k = key(str(x), str(y))
                    if k not in positives:
                        out.add(k)
                if len(out) >= n_target:
                    break
        return out

    # A configuration-model draw: each endpoint is sampled with probability proportional to its
    # degree in the positive set, which reproduces the positives' degree distribution in
    # expectation. The first version of this binned log-degree into 10 quantiles and drew
    # within bin, and it barely worked -- 12,075 of the 18,546 universe proteins appear in NO
    # positive, so log1p(degree) is 0 for 65% of them, qcut collapsed, and the "matched" arm
    # reached mean degree 20.8 against the positives' 50.5. The ceiling was never the problem:
    # 87% of top-200-hub pairs and 99% of top-2000-hub pairs are available as negatives
    # (scripts/02_degree_ceiling.py), so the shortfall was the matcher, not the data.
    prot = np.array([p for p in universe if pos_degree.get(p, 0) > 0])
    w = np.array([pos_degree[p] for p in prot], dtype=float)
    w /= w.sum()
    attempts, block = 0, max(n_target, 10_000)
    while len(out) < n_target and attempts < 200 * n_target:
        a = rng.choice(prot, block, p=w)
        b = rng.choice(prot, block, p=w)
        attempts += block
        for x, y in zip(a, b):
            if x == y:
                continue
            k = key(str(x), str(y))
            if k not in positives:
                out.add(k)
            if len(out) >= n_target:
                break
    if len(out) < n_target:
        log.warning("degree-matched negatives: only %d of %d after %d draws",
                    len(out), n_target, attempts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-per-pos", type=int, default=NEG_PER_POS)
    ap.add_argument("--prevalence", type=float, default=1 / 1000,
                    help="positive prevalence the AUCPR weights encode (paper: 1:1000)")
    a = ap.parse_args()
    BENCH.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    sp, bg = string_pairs(), biogrid_pairs()
    pos = sp & bg
    log.info("positives (STRING physical>%d AND BioGRID MV): %d  "
             "[STRING only %d, BioGRID only %d]",
             STRING_CUT, len(pos), len(sp - bg), len(bg - sp))
    if len(pos) < 1000:
        raise SystemExit(f"only {len(pos)} positives: the join is broken, not the data")

    # The universe is PrePPI's protein set, because every arm has to be scoreable on the same
    # pairs and PrePPI is the only one of the four that cannot be extended to new proteins.
    pp = pd.read_csv(DATA / "preppi_raw" / "preppi_final600.txt", sep="\t",
                     usecols=["prot1", "prot2"], low_memory=False)
    universe = sorted(set(pp.prot1) | set(pp.prot2))
    log.info("PrePPI protein universe: %d", len(universe))

    in_univ = set(universe)
    pos = {k for k in pos if k[0] in in_univ and k[1] in in_univ}
    log.info("positives inside the PrePPI universe: %d", len(pos))

    # All evidence, not just the positives, is excluded from negatives: a pair supported by one
    # source but not both is not a positive here, and it is certainly not a negative.
    evidence = sp | bg
    pos_degree: dict[str, int] = {}
    for x, y in pos:
        pos_degree[x] = pos_degree.get(x, 0) + 1
        pos_degree[y] = pos_degree.get(y, 0) + 1

    n_neg = a.neg_per_pos * len(pos)
    neg_rand = build_negatives(universe, evidence, pos_degree, n_neg, rng, degree_matched=False)
    neg_deg = build_negatives(universe, evidence, pos_degree, n_neg, rng, degree_matched=True)
    log.info("negatives: %d random, %d degree-matched", len(neg_rand), len(neg_deg))

    rows = []
    for k in pos:
        rows.append({"prot_a": k[0], "prot_b": k[1], "label": 1, "neg_set": "both"})
    for name, s in (("random", neg_rand), ("degmatched", neg_deg)):
        for k in s:
            rows.append({"prot_a": k[0], "prot_b": k[1], "label": 0, "neg_set": name})
    d = pd.DataFrame(rows)
    # One weight column per negative arm, so a downstream AUCPR is computed at the paper's
    # 1:1000 prevalence without resampling (and without silently changing n).
    for name in ("random", "degmatched"):
        sub = d.neg_set.isin(["both", name])
        w = np.zeros(len(d))
        n_pos, n_neg_a = int((d.label == 1).sum()), int((sub & (d.label == 0)).sum())
        # positives down-weighted so that sum(w_pos)/sum(w_neg) == prevalence
        w[(d.label == 1).to_numpy()] = a.prevalence * n_neg_a / max(n_pos, 1)
        w[(sub & (d.label == 0)).to_numpy()] = 1.0
        d[f"w_{name}"] = w
    d.to_parquet(BENCH / "pairs.parquet", index=False)

    log.info("\n%s", d.groupby(["label", "neg_set"]).size().to_string())
    log.info("degree check -- mean positive-set degree of the proteins in each arm:")
    for name, s in (("positives", pos), ("random", neg_rand), ("degmatched", neg_deg)):
        md = np.mean([pos_degree.get(x, 0) + pos_degree.get(y, 0) for x, y in s])
        log.info("  %-11s %6.2f", name, md)
    log.info("wrote %s", BENCH / "pairs.parquet")


if __name__ == "__main__":
    main()
