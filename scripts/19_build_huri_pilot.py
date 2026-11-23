"""AF3 inputs for a sized pilot of the HuRI benchmark.

HuRI is the independent replacement for the published folding benchmark, whose
positives are 9.6-fold better studied than its negatives. Its own positives
and negatives are balanced to 1.00-fold on that axis, so a difference measured
on it is not a difference in how well studied the proteins are.

A PILOT rather than the whole set, for the reason arm B was piloted: the
comparison is PAIRED -- the same pairs through both arms -- so the quantity
with the small variance is the arm-to-arm difference on one subsample, not
either arm's absolute AUROC. A few hundred pairs settle whether the engine
gap survives on an unbiased set; the full 2,479 only narrow an interval that
the pilot will already show is narrow or wide.

Built exactly like the benchmark arms: both chains carry their omicMSA, paired
by genome accession, with the quality mask applied as a GAP so the query stays
full length. That logic is imported from 07_build_af3_inputs.py, which carries
a validate() reproducing the authors' own paired MSAs. A second copy here is
where a silent mistake would look like a result.

Usage:
    micromamba run -n structmod python -I scripts/19_build_huri_pilot.py --n-pos 120 --n-neg 480
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("huri-pilot")

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "bench" / "huri_pairs.csv"
ALNS = ROOT / "data/rf2ppi_raw/protein_omicMSAs/alns"
OUT = ROOT / "work" / "huri_pilot_inputs"
SEED = 0

_spec = iu.spec_from_file_location("b7", ROOT / "scripts" / "07_build_af3_inputs.py")
b7 = iu.module_from_spec(_spec)
_spec.loader.exec_module(b7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=120)
    ap.add_argument("--n-neg", type=int, default=480,
                    help="kept at 4 negatives per positive, the native ratio of the full set")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIRS)
    have = {p.stem for p in ALNS.glob("*.a3m")}
    usable = pairs[pairs.a.isin(have) & pairs.b.isin(have)]
    log.info("%d of %d pairs have an omicMSA for both proteins", len(usable), len(pairs))

    # Sample within degree stratum, so the pilot keeps the degree matching the
    # full set was built to have. Taking the first N rows instead would inherit
    # whatever order the file happens to be in.
    rng = np.random.default_rng(SEED)
    edges = np.array([0, 1.5, 2.5, 4, 7, 12, 25, 45, 70, 110, 170, 260, 1e9])
    usable = usable.assign(stratum=np.digitize(usable.pdeg, edges) - 1)
    picked = []
    for cat, want in (("positive", a.n_pos), ("negative", a.n_neg)):
        sub = usable[usable.cat == cat]
        frac = sub.stratum.value_counts(normalize=True)
        for stratum, share in frac.items():
            pool = sub[sub.stratum == stratum]
            take = min(int(round(share * want)), len(pool))
            if take:
                picked.append(pool.sample(take, random_state=int(rng.integers(1 << 30))))
    pilot = pd.concat(picked).drop_duplicates("pair").reset_index(drop=True)
    log.info("pilot: %s", dict(pilot.cat.value_counts()))

    rows, n_fail = [], 0
    for r in pilot.itertuples():
        name = r.pair
        try:
            qA, qB, rowsA, rowsB = b7.build_pair(r.a, r.b)
            js = b7.af3_json(name, qA, qB, rowsA, rowsB)
        except Exception as exc:                       # noqa: BLE001
            n_fail += 1
            log.warning("%s: %s %s", name, type(exc).__name__, str(exc)[:70])
            continue
        (OUT / f"{name}.json").write_text(json.dumps(js))
        rows.append({"job": name, "a": r.a, "b": r.b, "cat": r.cat,
                     "pdeg": r.pdeg, "stratum": int(r.stratum),
                     "size": len(qA) + len(qB)})

    man = pd.DataFrame(rows).sort_values("size").reset_index(drop=True)
    man["fold_order"] = man.index + 1
    man.to_csv(ROOT / "work" / "huri_pilot_manifest.csv", index=False)
    log.info("wrote %d inputs, %d failed", len(man), n_fail)
    log.info("by class:\n%s", man.cat.value_counts().to_string())
    log.info("degree balance: positives %.1f, negatives %.1f (ratio %.2f)",
             man[man.cat.eq("positive")].pdeg.median(),
             man[man.cat.eq("negative")].pdeg.median(),
             man[man.cat.eq("positive")].pdeg.median()
             / max(man[man.cat.eq("negative")].pdeg.median(), 1e-9))
    log.info("tokens per fold: median %d, max %d", int(man["size"].median()), int(man["size"].max()))


if __name__ == "__main__":
    main()
