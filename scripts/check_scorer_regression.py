"""Guard: re-score pinned folds and refuse any drift in an existing column.

`one()` in 08_score_armB.py is shared by every arm and every screen, which is the
point -- a benchmark pair and a candidate pair must pass through identical code,
or a difference between them is about the code and not the biology. The cost of
that sharing is that an edit made for one stage silently changes every number
already published from another.

So the stored values for 24 folds are pinned in tests/golden_armB_sample.csv, and
this re-scores those folds and compares. Adding a column is fine and expected.
Moving an existing one is not, and this exits non-zero when it happens.

Run it after any edit to the scoring code:
    micromamba run -n structmod python -I scripts/check_scorer_regression.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-9

_spec = importlib.util.spec_from_file_location("score_arm", ROOT / "scripts" / "08_score_armB.py")
_m = importlib.util.module_from_spec(_spec)
sys.modules["score_arm"] = _m
_spec.loader.exec_module(_m)


def main() -> int:
    golden = pd.read_csv(ROOT / "tests" / "golden_armB_sample.csv")
    out = ROOT / "results" / "armB_out"
    cols = [c for c in golden.columns if c != "pair"]

    missing = [n for n in golden.pair if not (out / n).is_dir()]
    if missing:
        print(f"SKIP: {len(missing)} of {len(golden)} pinned folds are no longer on disk "
              f"({missing[:3]}). The guard cannot run; re-pin against folds that exist.")
        return 0

    drift, checked = [], 0
    for _, row in golden.iterrows():
        new = _m.one(out / row.pair)
        for c in cols:
            o, w = row[c], new.get(c)
            checked += 1
            if pd.isna(o) and (w is None or pd.isna(w)):
                continue
            if o is None or w is None or pd.isna(o) or pd.isna(w) or abs(float(o) - float(w)) > TOL:
                drift.append((row.pair, c, o, w))

    print(f"checked {checked} values across {len(golden)} folds and {len(cols)} columns")
    if drift:
        print(f"\nFAIL: {len(drift)} values moved. Every number already reported from this "
              f"scorer is now a different number.\n")
        for pair, c, o, w in drift[:20]:
            print(f"  {pair:34s} {c:14s} {o} -> {w}")
        if len(drift) > 20:
            print(f"  ... and {len(drift) - 20} more")
        print("\nIf the change is intended, re-score every arm and screen that this "
              "scorer feeds, then re-pin the golden sample.")
        return 1
    print("PASS: no existing column moved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
