"""Depth of the alignment each AlphaFold3 arm was actually given.

The comparison between arm A and arm B is a comparison between two alignments, so how deep
each one is has to be measured from the inputs the model received, not taken from how the
alignments are described. Depth is the number of sequences per chain in the `unpairedMsa`
and `pairedMsa` fields of the AlphaFold3 input.

An omicMSA alignment is paired by construction: 06_omicmsa_to_af3.py splits one paired
alignment into the two chains, so both fields hold the same rows.

Usage:
    python -I scripts/15_msa_depth.py
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("msa_depth")
ROOT = Path(__file__).resolve().parent.parent


def depth(path: str) -> tuple[float, float]:
    """Mean number of unpaired and of paired sequences per protein chain."""
    unpaired, paired = [], []
    for entry in json.load(open(path))["sequences"]:
        protein = entry.get("protein")
        if protein:
            unpaired.append((protein.get("unpairedMsa") or "").count(">"))
            paired.append((protein.get("pairedMsa") or "").count(">"))
    return float(np.mean(unpaired)), float(np.mean(paired))


def main() -> None:
    arm_a = {p.parent.name: str(p) for p in (ROOT / "results/armA_msa").glob("*/*_data.json")}
    arm_b = {p.stem: str(p) for p in (ROOT / "work/armB_inputs").glob("*.json")}
    pairs = sorted(set(arm_a) & set(arm_b))
    log.info("arm A %d, arm B %d, in both %d", len(arm_a), len(arm_b), len(pairs))
    with ProcessPoolExecutor(24) as pool:
        da = list(pool.map(depth, [arm_a[k] for k in pairs], chunksize=40))
        db = list(pool.map(depth, [arm_b[k] for k in pairs], chunksize=40))
    out = pd.DataFrame({"pair": pairs,
                        "alphafast_unpaired": [d[0] for d in da], "alphafast_paired": [d[1] for d in da],
                        "omicmsa_unpaired": [d[0] for d in db], "omicmsa_paired": [d[1] for d in db]})
    out.to_csv(ROOT / "results/msa_depth.csv", index=False)
    log.info("median unpaired per chain: AlphaFast %.0f, omicMSA %.0f, ratio %.2f",
             out.alphafast_unpaired.median(), out.omicmsa_unpaired.median(),
             out.alphafast_unpaired.median() / out.omicmsa_unpaired.median())


if __name__ == "__main__":
    main()
