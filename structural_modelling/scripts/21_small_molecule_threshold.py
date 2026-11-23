"""Operating points for a SMALL MOLECULE in a pocket, from the 256-fold benchmark.

Figure 3's thresholds come from the protein-protein benchmark and do not transfer
here. On that scale the small-molecule NEGATIVES of this benchmark sit at a median
interface pTM of 0.83, which is a top-decile protein score, and the PPI buried-area
cut of ~4,480 A^2 exceeds what any small molecule can bury (200-700 A^2) -- a gate
built from it rejects the entire arm by construction. A metabolite candidate needs
its own numbers, measured on receptors whose cognate ligand is known.

The benchmark: 16 GPCRs x 16 panel ligands, every cell folded on the same omicMSA
pipeline and scored by the same code as the screen.

  cognate    16   the ligand's own receptor
  negative  189   no measured ChEMBL activity at that receptor
  ambiguous  51   activity recorded but not the cognate pair

Ambiguous cells are NOT negatives and are excluded from the fit. 21% of off-diagonal
cells have measured activity, and calling those non-binders would set the bar using
51 pairs that may well bind. They are reported separately, because performance
against them is the honest measure of how fine a distinction this makes.

Two operating points, because they answer different questions:

  comparable   the cut where precision reaches 50%, the same criterion Figure 3's
               protein operating points use. Precision depends on prevalence and the
               two benchmarks differ (20% positives there, 7.8% here), so this is
               comparable in CRITERION, not in difficulty.
  sensitive    the cut keeping 90% of known cognate ligands, for a screen that would
               rather look at a few extra candidates than miss one.

With 16 positives a threshold is not a precise quantity, so each is bootstrapped
over the cognate set and reported with the spread. A cut quoted to two decimals off
16 points would imply a precision the data cannot support.

Usage:
    micromamba run -n structmod python -I scripts/21_small_molecule_threshold.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sm-threshold")
ROOT = Path(__file__).resolve().parent.parent

# higher is better for all of these except the PAE, where lower is
METRICS = {"iptm": True, "plddt_B": True, "iface_plddt_B": True,
           "pae_min_AB": False, "bsa": True, "n_contact": True}
N_BOOT = 2000


def _sweep(pos: np.ndarray, neg: np.ndarray, hi: bool):
    """Every distinct cut, with the recall and precision it gives."""
    vals = np.unique(np.concatenate([pos, neg]))
    out = []
    for c in vals:
        sp = pos >= c if hi else pos <= c
        sn = neg >= c if hi else neg <= c
        k = sp.sum() + sn.sum()
        if k == 0:
            continue
        out.append((float(c), sp.sum() / len(pos), sp.sum() / k, int(sn.sum())))
    return out


def _pick(sweep, mode: str):
    """The cut for one operating point, or None if the metric cannot reach it."""
    if mode == "comparable":                       # loosest cut at >=50% precision
        ok = [s for s in sweep if s[2] >= 0.50]
        return max(ok, key=lambda s: s[1]) if ok else None
    ok = [s for s in sweep if s[1] >= 0.90]        # tightest cut keeping >=90% recall
    return max(ok, key=lambda s: s[2]) if ok else None


def main() -> int:
    s = pd.read_csv(ROOT / "results" / "specificity_scores.csv")
    cog = s[s.label.eq("cognate")]
    neg = s[s.label.eq("negative")]
    amb = s[s.label.eq("ambiguous")]
    prev = len(cog) / (len(cog) + len(neg))
    log.info("benchmark: %d cognate, %d clean negative, %d ambiguous (excluded); "
             "prevalence %.1f%%", len(cog), len(neg), len(amb), 100 * prev)

    rng = np.random.default_rng(0)
    rows, chosen = [], {}
    for metric, hi in METRICS.items():
        p = cog[metric].to_numpy(float); p = p[np.isfinite(p)]
        n = neg[metric].to_numpy(float); n = n[np.isfinite(n)]
        a = amb[metric].to_numpy(float); a = a[np.isfinite(a)]
        if len(p) < 8:
            continue
        sweep = _sweep(p, n, hi)
        for mode in ("comparable", "sensitive"):
            hit = _pick(sweep, mode)
            if hit is None:
                rows.append({"metric": metric, "point": mode, "cut": np.nan,
                             "note": "cannot reach this operating point"})
                continue
            cut, rec, prec, n_fp = hit
            # how stable is the cut itself, resampling the 16 cognates?
            boots = []
            for _ in range(N_BOOT):
                pb = rng.choice(p, len(p), replace=True)
                h2 = _pick(_sweep(pb, n, hi), mode)
                if h2:
                    boots.append(h2[0])
            lo, hi_b = (np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan))
            amb_kept = float(np.mean(a >= cut) if hi else np.mean(a <= cut)) if len(a) else np.nan
            rows.append({
                "metric": metric, "point": mode, "cut": round(cut, 3),
                "cut_p5": round(lo, 3), "cut_p95": round(hi_b, 3),
                "cognate_recall": round(rec, 3), "precision": round(prec, 3),
                "negatives_admitted": n_fp,
                "frac_negatives": round(n_fp / len(n), 3),
                "frac_ambiguous_kept": round(amb_kept, 3) if amb_kept == amb_kept else None,
                "n_pos": len(p), "n_neg": len(n)})
            chosen.setdefault(mode, {})[metric] = {
                "cut": round(cut, 3), "higher_is_better": hi,
                "cognate_recall": round(rec, 3), "precision": round(prec, 3),
                "cut_p5": round(lo, 3), "cut_p95": round(hi_b, 3)}

    t = pd.DataFrame(rows)
    log.info("\n%s", t.to_string(index=False))

    # what each point does to the orphan metabolite arm
    o = pd.read_csv(ROOT / "results" / "orphan_scores.csv")
    o = o[o.iptm.notna() & o.kind.eq("metabolite")]
    log.info("\napplied to the %d orphan metabolite folds:", len(o))
    summary = {}
    for mode, mets in chosen.items():
        # the two metrics the benchmark ranks best, which are also the two the
        # co-folding literature names: the confidence and the ligand's own pLDDT
        use = [m for m in ("iptm", "plddt_B") if m in mets]
        mask = pd.Series(True, index=o.index)
        for m in use:
            c = mets[m]["cut"]
            mask &= (o[m] >= c) if METRICS[m] else (o[m] <= c)
        summary[mode] = int(mask.sum())
        log.info("  %-11s %s -> %d of %d folds survive", mode,
                 " and ".join(f"{m} {'>=' if METRICS[m] else '<='} {mets[m]['cut']}"
                              for m in use), int(mask.sum()), len(o))

    dest = ROOT / "results" / "small_molecule_thresholds.json"
    dest.write_text(json.dumps({
        "built_from": "results/specificity_scores.csv",
        "benchmark": {"cognate": len(cog), "negative": len(neg), "ambiguous_excluded": len(amb),
                      "prevalence": round(prev, 4)},
        "why_not_figure_3": ("Figure 3's operating points are fitted on protein-protein "
                             "pairs. On that scale these small-molecule negatives sit at a "
                             "median interface pTM of 0.83 and its buried-area cut of "
                             "~4,480 A^2 exceeds what any small molecule can bury, so a "
                             "protein gate rejects this arm by construction."),
        "points": chosen,
        "orphan_metabolite_folds_surviving": summary}, indent=2) + "\n")
    t.to_csv(ROOT / "results" / "small_molecule_thresholds.csv", index=False)
    log.info("\nwrote %s and small_molecule_thresholds.csv", dest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
