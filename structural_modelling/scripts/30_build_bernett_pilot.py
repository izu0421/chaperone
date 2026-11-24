"""AF3 inputs for a pilot of the Bernett gold-standard PPI benchmark.

Why a third benchmark. Every protein-side result in this project is fitted on the
2,500-pair RoseTTAFold2-PPI set, whose positives come from proteins 9.57x better
studied than its negatives (measured against BioGRID-multivalidated intersected
with STRING physical >700, which shares no construction with it). On a
degree-balanced set, HuRI, AlphaFold3 scores at chance: interface PAE AUROC
0.519 [0.466, 0.574] against 0.840 on the published benchmark, with a detection
floor of 0.558, comparable fold quality and 8x deeper alignments. Two readings
survive that and we cannot separate them: the first benchmark's signal is study
bias, or HuRI's yeast-two-hybrid labels are too noisy to score against.

Bernett et al. (figshare 10.6084/m9.figshare.21591618.v4, "Cracking the black
box of deep sequence-based PPI prediction") is the set that separates them. It
is degree-balanced like HuRI -- 1.09x on the same independent reference -- but
its positives are curated, from HIPPIE v2.3, rather than one assay's calls. If
AlphaFold3 also scores at chance here, the label-noise reading is out.

Design:

  balanced       250 positive and 250 negative. The detection floor at that size
                 is 0.551, so a true AUROC of 0.60 resolves from chance with a
                 95% half-width of 0.049. That is the smallest effect worth
                 acting on; 0.840 would be unmissable.
  stratified     on literature degree AND complex size, because those are the two
                 axes on which a benchmark of this kind goes wrong, and because
                 matching on one while the other drifts is how a confound
                 survives a correction.
  deduplicated   pairs appearing in the RF2-PPI benchmark or the HuRI set are
                 excluded, so the three numbers come from disjoint pairs.
  same code      built by 07_build_af3_inputs.build_pair, folded by the same
                 command as the screen and both other benchmarks. A second copy
                 of the pairing logic is where a silent mistake would look like
                 a result.

Usage:
    micromamba run -n structmod python -I scripts/30_build_bernett_pilot.py \
        --n-pos 250 --n-neg 250
"""
from __future__ import annotations

import argparse
import collections
import importlib.util as iu
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bernett")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "bernett_raw"
ALNS = ROOT / "data" / "rf2ppi_raw" / "protein_omicMSAs" / "alns"
OUT = ROOT / "work" / "bernett_pilot_inputs"
SEED = 0

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)

DEG_EDGES = np.array([-0.1, 0.5, 2, 5, 10, 20, 40, 1e9])
SIZE_EDGES = np.array([0, 600, 900, 1200, 1600, 2200, 1e9])


def literature_degree() -> collections.Counter:
    """The reference Extended Data Figure 10 uses, and not one built from any
    benchmark being audited: BioGRID multi-validated physical intersected with
    STRING physical above 700."""
    own = pd.read_parquet(ROOT / "bench" / "pairs.parquet")
    lit = own[own.label == 1]
    deg = collections.Counter()
    for a, b in zip(lit.prot_a, lit.prot_b):
        deg[a] += 1
        deg[b] += 1
    log.info("degree reference: %d literature positives over %d proteins", len(lit), len(deg))
    return deg


def load_pairs() -> pd.DataFrame:
    frames = []
    for split in (0, 1, 2):
        for cat in ("pos", "neg"):
            f = RAW / f"Intra{split}_{cat}_rr.txt"
            d = pd.read_csv(f, sep=r"\s+", header=None, names=["a", "b"])
            frames.append(d.assign(split=split,
                                   cat="positive" if cat == "pos" else "negative"))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=250)
    ap.add_argument("--n-neg", type=int, default=250)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="drop complexes above this; 0 keeps all. A cap applies to both "
                         "classes, whose sizes are balanced, so it does not skew labels")
    ap.add_argument("--dry-run", action="store_true", help="select and report, build nothing")
    a = ap.parse_args()

    pairs = load_pairs()
    log.info("%d pairs in the gold standard", len(pairs))

    # sequence lengths from the fasta they ship, so complex size is theirs
    seq, name = {}, None
    for line in (RAW / "human_swissprot_oneliner.fasta").read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
        elif name:
            seq[name] = len(line)
            name = None
    pairs["size"] = pairs.a.map(seq).fillna(0) + pairs.b.map(seq).fillna(0)

    have = {p.stem for p in ALNS.glob("*.a3m")}
    pairs["foldable"] = pairs.a.isin(have) & pairs.b.isin(have) & pairs["size"].gt(0)

    # disjoint from the other two benchmarks, so three numbers do not share pairs
    rf2 = pd.read_csv(ROOT / "results" / "armB_scores_labelled.csv")
    huri = pd.read_csv(ROOT / "bench" / "huri_pairs.csv")
    seen = set(map(frozenset, zip(rf2.prot_a, rf2.prot_b)))
    seen |= set(map(frozenset, zip(huri.a, huri.b)))
    pairs["overlaps"] = [frozenset((x, y)) in seen for x, y in zip(pairs.a, pairs.b)]

    usable = pairs[pairs.foldable & ~pairs.overlaps].copy()
    if a.max_tokens:
        usable = usable[usable["size"] <= a.max_tokens]
    log.info("usable: %d of %d (foldable %d, dropped %d overlapping, %d over the cap)",
             len(usable), len(pairs), int(pairs.foldable.sum()),
             int((pairs.foldable & pairs.overlaps).sum()),
             int(pairs.foldable.sum() - pairs.overlaps.sum() - len(usable)))

    deg = literature_degree()
    usable["pdeg"] = [(deg[x] + deg[y]) / 2 for x, y in zip(usable.a, usable.b)]
    usable["dstratum"] = np.digitize(usable.pdeg, DEG_EDGES) - 1
    usable["sstratum"] = np.digitize(usable["size"], SIZE_EDGES) - 1

    # Sample the NEGATIVES to the positives' joint (degree, size) profile, rather
    # than sampling each class to its own. Matching class-wise leaves the joint
    # distribution free, which is how a set can be matched on two margins and
    # confounded on their interaction.
    rng = np.random.default_rng(SEED)
    pos_pool = usable[usable.cat.eq("positive")]
    neg_pool = usable[usable.cat.eq("negative")]
    pos = pos_pool.sample(min(a.n_pos, len(pos_pool)), random_state=SEED)

    want = pos.groupby(["dstratum", "sstratum"]).size()
    picked, short = [], []
    for (d, s), k in want.items():
        cell = neg_pool[(neg_pool.dstratum == d) & (neg_pool.sstratum == s)]
        take = min(k, len(cell))
        if take < k:
            short.append(((d, s), int(k), int(len(cell))))
        if take:
            picked.append(cell.sample(take, random_state=int(rng.integers(1 << 30))))
    neg = pd.concat(picked) if picked else neg_pool.head(0)

    if short:
        log.warning("%d (degree, size) cells could not be filled from the negative pool:",
                    len(short))
        for cell, k, got in short[:8]:
            log.warning("   cell %s wanted %d, pool had %d", cell, k, got)
        log.warning("a thin pool makes a matched background short in exactly the strata "
                    "that carry the confound; %d of %d negatives selected",
                    len(neg), len(pos))

    sel = pd.concat([pos, neg], ignore_index=True)
    log.info("\nselected %d positive, %d negative", int(sel.cat.eq('positive').sum()),
             int(sel.cat.eq('negative').sum()))
    for col in ("pdeg", "size"):
        g = sel.groupby("cat")[col].agg(["mean", "median"])
        ratio = g.loc["positive", "mean"] / max(g.loc["negative", "mean"], 1e-9)
        log.info("  %-5s positives %8.1f  negatives %8.1f   ratio %.2fx",
                 col, g.loc["positive", "mean"], g.loc["negative", "mean"], ratio)
        if col == "pdeg" and not 0.8 <= ratio <= 1.25:
            raise SystemExit(f"the selected pilot is degree-skewed at {ratio:.2f}x, which "
                             f"is the defect it exists to avoid. Not building.")

    if a.dry_run:
        log.info("dry run: nothing built")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for r in sel.itertuples():
        name = f"{r.a}_{r.b}"
        try:
            qA, qB, rowsA, rowsB = b7.build_pair(r.a, r.b)
            keptA, keptB = b7.hhfilter_pairs(qA, qB, rowsA, rowsB)
            js = b7.af3_json(name, qA, qB, keptA, keptB)
            (OUT / f"{name}.json").write_text(json.dumps(js))
        except Exception as exc:                          # noqa: BLE001
            failed.append((name, str(exc)[:90]))
            continue
        rows.append({"job": name, "a": r.a, "b": r.b, "cat": r.cat, "split": r.split,
                     "pdeg": r.pdeg, "dstratum": r.dstratum, "sstratum": r.sstratum,
                     "size": len(qA) + len(qB), "paired_rows": len(keptA)})

    if failed:
        log.warning("%d inputs failed to build:", len(failed))
        for n, why in failed[:10]:
            log.warning("   %s: %s", n, why)
    if len(rows) < 0.9 * len(sel):
        raise SystemExit(f"only {len(rows)} of {len(sel)} built; a pilot missing that many "
                         f"is a different sample from the one that was selected")

    man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    dest = ROOT / "work" / "bernett_pilot_manifest.csv"
    man.to_csv(dest, index=False)
    log.info("\nbuilt %d inputs -> %s", len(man), OUT)
    log.info("complex size: median %d, max %d tokens", man["size"].median(), man["size"].max())
    log.info("paired MSA depth: median %d rows (HuRI pilot 4754, RF2-PPI benchmark 598)",
             man.paired_rows.median())
    log.info("manifest -> %s", dest)


if __name__ == "__main__":
    main()
