"""Arm B discrimination, with the labels taken from the pilot's OWN manifest.

08_score_armB.py joined labels from bench/pairs.parquet (the 970k-pair universe) and recovered
only 504 of 2,500 -- 497 positive against 7 negative. The sampled negatives are not in that file
under the same key, so the join silently kept the positives and dropped almost every negative,
and the AUROC it printed (0.955) was measured against SEVEN negatives. That number is an
artefact of the join and must not be quoted.

bench/pilot_pairs.csv carries `cat` (positive/negative) for all 2,500 rows by construction, so
it is the authoritative label. Coverage is asserted here: anything below 100% stops the script
instead of being averaged over.

Metrics reported per feature:
  AUROC      rank discrimination
  AUPRC      with the 1:4 prevalence of this pilot stated, since AUPRC is prevalence-dependent
  distinct   fraction of distinct values. A score saturated at one threshold is not a ranker even
             when its AUROC looks high (TEMPO's 0.807 was one threshold with 35% tied).

Usage:
    python -I scripts/09_armB_metrics.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("m")
ROOT = Path(__file__).resolve().parent.parent

import argparse
# B - A is a paired difference, so both arms run through THIS file, not a copy.
_ap = argparse.ArgumentParser(); _ap.add_argument("--arm", default="B", choices=["A", "B"])
_ARM = _ap.parse_args().arm

d = pd.read_csv(ROOT / "results" / f"arm{_ARM}_scores.csv")
pil = pd.read_csv(ROOT / "bench" / "pilot_pairs.csv")
log.info("%d scored, %d in the pilot manifest", len(d), len(pil))

pil["key"] = [tuple(sorted((a, b))) for a, b in zip(pil.a, pil.b)]
d["key"] = [tuple(sorted((a, b))) for a, b in zip(d.prot_a, d.prot_b)]
lab = dict(zip(pil.key, pil.cat))
d["cat"] = d.key.map(lab)
cov = d.cat.notna().mean()
log.info("label coverage %.4f (%d of %d)", cov, int(d.cat.notna().sum()), len(d))
if cov < 1.0:
    miss = d[d.cat.isna()].pair.head(5).tolist()
    raise SystemExit(f"coverage {cov:.3f} < 1.0 -- the join is wrong, not the data. {miss}")

d["y"] = (d.cat == "positive").astype(int)
prev = d.y.mean()
log.info("\n%s", d.cat.value_counts().to_string())
log.info("prevalence %.3f (AUPRC baseline)", prev)

log.info("\n--- arm %s: %s, standalone discrimination (n=%d) ---", _ARM, {"A": "AF3 + AlphaFast MSA", "B": "AF3 + omicMSA"}[_ARM], len(d))
rows = []
for col in ["iptm", "iptm_AB", "ptm", "ranking_score", "pae_min_AB",
            "n_contact", "n_res_iface", "bsa", "iface_plddt", "mean_plddt",
            "fraction_disordered"]:
    if col not in d or d[col].isna().all():
        continue
    v = d.dropna(subset=[col])
    s = v[col].to_numpy(float)
    if col in ("pae_min_AB", "fraction_disordered"):
        s = -s                                   # lower is better
    auroc = roc_auc_score(v.y, s)
    rows.append({"feature": col, "AUROC": auroc,
                 "AUPRC": average_precision_score(v.y, s),
                 "n": len(v), "distinct": v[col].nunique() / len(v)})
m = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
log.info("\n%s", m.round(3).to_string(index=False))

log.info("\nclash check: %d of %d models have min_sep < 1.5 A "
         "(%.1f%% of positives, %.1f%% of negatives)",
         int((d.min_sep < 1.5).sum()), int(d.min_sep.notna().sum()),
         100 * (d.loc[d.y == 1, "min_sep"] < 1.5).mean(),
         100 * (d.loc[d.y == 0, "min_sep"] < 1.5).mean())

d.drop(columns=["key"]).to_csv(ROOT / "results" / f"arm{_ARM}_scores_labelled.csv", index=False)
m.to_csv(ROOT / "results" / f"arm{_ARM}_feature_auc.csv", index=False)
log.info("wrote %s", ROOT / "results" / f"arm{_ARM}_scores_labelled.csv")
