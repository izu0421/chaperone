"""Operating point for a PROTEIN partner, from the 2,500-pair benchmark.

The protein arm has been gated on the thresholds the RoseTTAFold2-PPI paper
published -- interface pTM >= 0.5, buried area >= 800 A^2, interface pLDDT >= 70,
no clash -- and that gate has three problems, all measurable:

 1. It gates on a metric this project never validated. Figure 3a's AUROC of 0.840
    for AlphaFold3 is computed on interface PAE. The screen then selects on
    interface pTM and interface pLDDT, which score 0.821 and 0.752 on the same
    pairs.

 2. Interface pLDDT is the binding constraint (151 folds -> 25) and is the most
    alignment-depth-dependent term there is. Among benchmark NEGATIVES, where
    nothing is true and any trend is artefact, its median runs 57.3 to 65.7 across
    depth quartiles (rho +0.312) against a cut at 70. Interface PAE runs 16.8 to
    18.3 (rho +0.086).

 3. Interface pTM and buried area both rise with complex size on the orphan arm
    (rho +0.362 and +0.420), so a gate built on them ranks partly on size.
    Interface PAE does not (rho -0.071).

So the gate is refitted on interface PAE. The anchor is the PUBLISHED gate's own
false-positive rate on the benchmark, not a round number: holding the decoy rate
fixed is what makes "more candidates" mean "more recall" and not "a looser bar".

Buried area is dropped because it is redundant once PAE gates -- the benchmark
and the orphan arm both return the identical candidate set with and without it,
so it contributed nothing but its size dependence. The clash check stays: it is a
statement about physical validity, not about confidence.

Usage:
    micromamba run -n structmod python -I scripts/27_protein_threshold.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("prot-threshold")
ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 2000
# The clash floor. Measured 2026-11-24 on the 2,500-pair benchmark: minimum
# inter-chain separation is ANTI-predictive as a score (AUROC 0.402), because a
# real interface is TIGHT -- a floor removes the closest contacts first, and at
# 1.5 A it removes 12.0% of positives against 8.4% of negatives.
#
#   floor   removes pos   removes neg   recall   precision
#     0.0          0.0%          0.0%    56.0%       85.4%
#     0.5          1.2%          0.6%    55.4%       85.2%
#     1.5         12.0%          8.4%    50.2%       83.9%   <- was in use
#
# It is kept, because a model with atoms closer than half an angstrom is two
# atoms in the same place and this project has shipped a design at 0.40 A before.
# But it is a check on physical possibility, not a confidence term, so it sits
# where impossible geometry is and not where tight interfaces are. 18 of 2,500
# benchmark models fall below 0.5 A.
MIN_SEP = 0.5
PUBLISHED = dict(iptm=0.5, bsa=800, iface_plddt=70)


def gate_stats(mask: np.ndarray, y: np.ndarray) -> dict:
    return {"n": int(mask.sum()),
            "recall": float(mask[y].mean()),
            "decoy_rate": float(mask[~y].mean()),
            "precision": float(y[mask].mean()) if mask.sum() else float("nan")}


def main() -> None:
    b = pd.read_csv(ROOT / "results" / "armB_scores_labelled.csv")
    b["no_clash"] = b.min_sep >= MIN_SEP
    y = b.cat.eq("positive").to_numpy()
    log.info("benchmark arm B: %d pairs, %d positive, %d negative",
             len(b), int(y.sum()), int((~y).sum()))

    pub = (b.iptm.ge(PUBLISHED["iptm"]) & b.bsa.ge(PUBLISHED["bsa"])
           & b.iface_plddt.ge(PUBLISHED["iface_plddt"]) & b.no_clash).to_numpy()
    pub_stats = gate_stats(pub, y)
    log.info("published gate: %s", pub_stats)
    target = pub_stats["decoy_rate"]

    # the cut on interface PAE that admits the same fraction of known decoys
    score = np.where(b.no_clash, -b.pae_min_AB.to_numpy(float), -np.inf)
    cut = float(np.quantile(score[~y], 1 - target))
    new = score >= cut
    new_stats = gate_stats(new, y)
    log.info("interface PAE gate at the SAME decoy rate (cut %.2f A): %s",
             -cut, new_stats)

    # the cut is a statistic of 2,000 negatives, so report its spread
    rng = np.random.default_rng(0)
    neg_scores = score[~y]
    boots = [np.quantile(rng.choice(neg_scores, len(neg_scores), True), 1 - target)
             for _ in range(N_BOOT)]
    lo, hi = -np.percentile(boots, 97.5), -np.percentile(boots, 2.5)
    log.info("cut %.2f A, 95%% bootstrap interval [%.2f, %.2f]", -cut, lo, hi)

    # buried area, checked rather than assumed to be redundant
    with_bsa = (score >= cut) & b.bsa.ge(PUBLISHED["bsa"]).to_numpy()
    log.info("adding buried area >= %d back: %d of %d candidates change",
             PUBLISHED["bsa"], int((with_bsa != new).sum()), len(b))

    # ---- why this metric and not the published ones ------------------------
    # Measured here rather than quoted, because the manuscript cites all three
    # sets and a number typed into a script drifts from the data silently.
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    feats = {"pae_min_AB": False, "iptm": True, "iface_plddt": True, "bsa": True}
    why = {"benchmark_auroc": {}, "depth_rho_in_negatives": {},
           "size_rho_orphan_arm": {}}
    for f, hb in feats.items():
        v = b[f].to_numpy(float)
        ok = ~np.isnan(v)
        why["benchmark_auroc"][f] = round(
            float(roc_auc_score(y[ok], v[ok] if hb else -v[ok])), 3)

    # depth dependence is measured among NEGATIVES only: there nothing is true,
    # so any trend with alignment depth is artefact and not signal
    depth = pd.read_csv(ROOT / "results" / "msa_depth.csv")
    bn = b.merge(depth, on="pair", how="inner")
    bn = bn[~bn.cat.eq("positive")]
    for f, hb in feats.items():
        r, _ = spearmanr(bn.omicmsa_paired, bn[f] if hb else -bn[f], nan_policy="omit")
        why["depth_rho_in_negatives"][f] = round(float(r), 3)
    q = pd.qcut(bn.omicmsa_paired, 4)
    why["iface_plddt_by_depth_quartile"] = [
        round(float(x), 1) for x in bn.groupby(q, observed=True).iface_plddt.median()]
    why["depth_quartile_medians"] = [
        int(x) for x in bn.groupby(q, observed=True).omicmsa_paired.median()]

    # ---- transfer to the orphan protein arm --------------------------------
    s = pd.read_csv(ROOT / "results" / "orphan_scores.csv")
    s = s[s.iptm.notna()].copy()
    s["no_clash"] = s.min_sep >= MIN_SEP
    man = pd.read_csv(ROOT / "work" / "orphan_manifest.csv")
    p = (s[s.kind == "protein"]
         .merge(man[["job", "size"]].rename(columns={"job": "pair"}), on="pair", how="left"))

    old_sel = (p.iptm.ge(PUBLISHED["iptm"]) & p.bsa.ge(PUBLISHED["bsa"])
               & p.iface_plddt.ge(PUBLISHED["iface_plddt"]) & p.no_clash)
    new_sel = p.no_clash & p.pae_min_AB.le(-cut)
    log.info("\norphan protein arm (%d folds):", len(p))
    log.info("  published gate      %3d candidates, %2d receptors",
             int(old_sel.sum()), p[old_sel].receptor.nunique())
    log.info("  interface PAE gate  %3d candidates, %2d receptors",
             int(new_sel.sum()), p[new_sel].receptor.nunique())
    log.info("  overlap %d; PAE-only %d; pLDDT-only %d",
             int((old_sel & new_sel).sum()), int((new_sel & ~old_sel).sum()),
             int((old_sel & ~new_sel).sum()))

    for f, hb in feats.items():
        r, _ = spearmanr(p["size"], p[f] if hb else -p[f], nan_policy="omit")
        why["size_rho_orphan_arm"][f] = round(float(r), 3)
    log.info("  benchmark AUROC      %s", why["benchmark_auroc"])
    log.info("  depth rho (negatives) %s", why["depth_rho_in_negatives"])
    log.info("  size rho (orphan arm) %s", why["size_rho_orphan_arm"])
    log.info("  interface pLDDT by depth quartile %s at depths %s",
             why["iface_plddt_by_depth_quartile"], why["depth_quartile_medians"])

    from scipy.stats import mannwhitneyu
    for nm, m in (("published", old_sel), ("interface PAE", new_sel)):
        _, pv = mannwhitneyu(p[m]["size"].dropna(), p[~m]["size"].dropna())
        log.info("  %-14s median complex %5.0f vs %5.0f tokens for the rest, p = %.2g",
                 nm, p[m]["size"].median(), p[~m]["size"].median(), pv)

    out = p[new_sel].sort_values("pae_min_AB")[
        ["pair", "receptor", "partner", "pae_min_AB", "iptm", "bsa", "iface_plddt",
         "n_contact", "min_sep", "size", "p_twoway"]]
    dest = ROOT / "results" / "protein_calibrated_hits.csv"
    out.to_csv(dest, index=False)

    (ROOT / "results" / "protein_thresholds.json").write_text(json.dumps({
        "fitted_on": "results/armB_scores_labelled.csv",
        "anchor": "the published gate's own false-positive rate on this benchmark",
        "published_gate": {**PUBLISHED, "min_sep": MIN_SEP, **pub_stats},
        "gate": {"pae_min_AB_max": round(-cut, 2), "min_sep": MIN_SEP,
                 "cut_ci": [round(lo, 2), round(hi, 2)], **new_stats},
        "why_not_iface_plddt": why,
        "orphan_candidates": int(new_sel.sum()),
        "orphan_candidates_published_gate": int(old_sel.sum()),
    }, indent=1) + "\n")
    log.info("\nwrote %d candidates to %s", len(out), dest)
    log.info("top 10 by interface PAE:\n%s", out.head(10).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
