"""Did the step that chose what to fold enrich anything? No -- but not the way I first said.

The screen folded 5,000 of about 78,900 eligible receptor x partner combinations,
chosen by a spatial recurrence rank. That rank is the screen's only enrichment
step.

A FIRST VERSION OF THIS SCRIPT CLAIMED THE RANK WAS ANTI-CORRELATED WITH THE
STRUCTURAL GATE, on a Cochran-Armitage trend of z = +3.34, p = 0.0008, and
described it as graded across all five quintiles rather than an extreme split.
That was wrong in three ways, and the script now tests each of them:

  - DROP Q5 AND THERE IS NO TREND. Q1-Q4 run 0.79 / 0.79 / 1.18 / 0.92 %,
    z = +0.36, p = 0.72. The whole signal is the worst-ranked fifth against the
    rest, which is exactly the single lucky split the old docstring claimed to
    have ruled out by testing five quintiles.
  - THE BEST-RANKED QUINTILE IS NOT LOW. Q1 against Q2-Q4 gives OR 0.62,
    p = 0.33. The headline "0.63% against 2.31%" paired a non-significant low
    quintile with the one significant high one.
  - A THIRD OF THE GAP IS RECEPTOR COMPOSITION. The within-receptor permutation
    null is centred at +0.55 pp, not at zero, and the analytic receptor term is
    +0.56 pp of the observed +1.68 pp. Stratified by receptor the trend is
    z = +2.02, p = 0.044, not p = 0.0008.

What survives is weaker and more specific: the rank carries no information about
the gate QUANTITY at all (Spearman -0.011, p = 0.43), and the only
threshold-level signal is a tail excess in the worst-ranked fifth. On interface
pLDDT the relationship even runs the other way. The screen's selection step
neither enriches nor depletes structurally plausible pairs; it is unrelated to
them.

The gate is applied here exactly as the triaged list was selected -- interface
PAE <= 3.76 A AND min separation >= 1.5 A, which reproduces the 57 protein pairs
that actually received verdicts. The first version applied the PAE cut alone and
scored a 62-pair population that no verdict came from.

Usage:
    micromamba run -n structmod python -I scripts/40_selection_vs_gate.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("selection")
ROOT = Path(__file__).resolve().parent.parent
SHEET = Path("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
N_PERM = 3000
MIN_SEP_AS_SELECTED = 1.5


def ca_trend(df: pd.DataFrame, q: int = 5) -> tuple[float, float, list]:
    df = df.copy()
    df["q"] = pd.qcut(df["rank"], q, labels=False)
    t = df.groupby("q")["pass"].agg(["size", "sum"])
    x, n, r = np.arange(q), t["size"].to_numpy(), t["sum"].to_numpy()
    pb, xb = r.sum() / n.sum(), (n * x).sum() / n.sum()
    z = ((r - n * pb) * (x - xb)).sum() / np.sqrt(pb * (1 - pb) * (n * (x - xb) ** 2).sum())
    return float(z), float(2 * stats.norm.sf(abs(z))), (100 * t["sum"] / t["size"]).round(2).tolist()


def main() -> None:
    gate = json.loads((ROOT / "results" / "protein_thresholds.json").read_text()
                      )["gate"]["pae_min_AB_max"]
    sheet = pd.read_csv(SHEET)
    sc = pd.read_csv(ROOT / "results" / "orphan_scores.csv")
    sc["key"] = sc.receptor.astype(str) + "__" + sc.partner.astype(str)
    sheet["key"] = sheet.receptor_gene.astype(str) + "__" + sheet.partner_gene.astype(str)
    d = sheet.merge(sc[["key", "pae_min_AB", "min_sep", "iptm", "bsa", "iface_plddt", "kind"]],
                    on="key", how="inner")
    d = d[d.kind.eq("protein")].copy()
    d["pass"] = ((d.pae_min_AB <= gate) & (d.min_sep >= MIN_SEP_AS_SELECTED)).astype(int)

    eligible = sheet.receptor_gene.nunique() * sheet.partner_gene.nunique()
    log.info("%d protein pairs scored; %d clear the gate as the triaged list was selected "
             "(PAE <= %.2f A and min separation >= %.1f A)", len(d), d["pass"].sum(),
             gate, MIN_SEP_AS_SELECTED)
    log.info("chosen from about %d eligible combinations (%d receptors x %d partners): "
             "%.1f%% coverage\n", eligible, sheet.receptor_gene.nunique(),
             sheet.partner_gene.nunique(), 100 * len(sheet) / eligible)

    z, p, rates = ca_trend(d)
    log.info("GATE PASS RATE BY SPATIAL RANK QUINTILE (Q1 = strongest spatial evidence)")
    log.info("   %s", "  ".join(f"Q{i+1} {r:.2f}%" for i, r in enumerate(rates)))
    log.info("   Cochran-Armitage over all five: z = %+.2f, p = %.4f", z, p)

    cut = d["rank"].quantile(0.8)
    z4, p4, r4 = ca_trend(d[d["rank"] <= cut], q=5)
    log.info("\nIS IT A TREND, OR ONE QUINTILE?")
    log.info("   drop Q5 and re-test Q1-Q4: z = %+.2f, p = %.4f   rates %s",
             z4, p4, "  ".join(f"{r:.2f}%" for r in r4))
    lo, hi = d["rank"].quantile(0.2), d["rank"].quantile(0.8)
    q1, mid, q5 = d[d["rank"] <= lo], d[(d["rank"] > lo) & (d["rank"] <= hi)], d[d["rank"] > hi]
    rest = pd.concat([q1, mid])
    o5, p5 = stats.fisher_exact([[q5["pass"].sum(), len(q5) - q5["pass"].sum()],
                                 [rest["pass"].sum(), len(rest) - rest["pass"].sum()]])
    o1, p1 = stats.fisher_exact([[q1["pass"].sum(), len(q1) - q1["pass"].sum()],
                                 [mid["pass"].sum(), len(mid) - mid["pass"].sum()]])
    log.info("   Q5 against Q1-Q4: OR = %.2f, p = %.5f   <- the whole signal", o5, p5)
    log.info("   Q1 against Q2-Q4: OR = %.2f, p = %.4f   <- the 'best' quintile is not low",
             o1, p1)

    log.info("\nDOES THE RANK TRACK THE GATE QUANTITY AT ALL?")
    for c in ("pae_min_AB", "iptm", "bsa", "iface_plddt"):
        r, pp = stats.spearmanr(d["rank"], d[c], nan_policy="omit")
        note = ""
        if c == "pae_min_AB":
            note = "   <- the gate's own quantity: no signal"
        elif c == "iface_plddt" and r < 0 and pp < 0.05:
            note = "   <- and this one runs the OTHER way"
        log.info("   rank vs %-12s rho = %+.4f  p = %.3f%s", c, r, pp, note)

    log.info("\nHOW MUCH OF THE GAP IS RECEPTOR COMPOSITION?")
    def gap(df: pd.DataFrame) -> float:
        q = pd.qcut(df["rank"], 5, labels=False)
        m = df.groupby(q)["pass"].mean()
        return m.iloc[-1] - m.iloc[0]
    obs = gap(d)
    rng = np.random.default_rng(0)
    null = np.array([gap(d.assign(**{"rank": d.groupby("receptor_gene")["rank"].transform(
        lambda s: rng.permutation(s.values))})) for _ in range(N_PERM)])
    d["q"] = pd.qcut(d["rank"], 5, labels=False)
    pr = d.groupby("receptor_gene")["pass"].mean()
    w5 = d[d.q == 4].receptor_gene.value_counts(normalize=True)
    w1 = d[d.q == 0].receptor_gene.value_counts(normalize=True)
    analytic = sum((w5.get(r, 0) - w1.get(r, 0)) * pr[r] for r in pr.index)
    log.info("   observed Q5-Q1 gap               %+.2f pp", 100 * obs)
    log.info("   within-receptor null             mean %+.2f pp (sd %.2f) -- NOT zero",
             100 * null.mean(), 100 * null.std())
    log.info("   analytic receptor-composition    %+.2f pp, i.e. %.0f%% of the gap",
             100 * analytic, 100 * analytic / obs)
    log.info("   p, excess over that null         %.4f", float((null >= obs).mean()))

    num = den = 0.0
    for _, g in d.groupby("receptor_gene"):
        if g["pass"].sum() == 0 or len(g) < 10:
            continue
        x, y = g["rank"].rank(pct=True).to_numpy(), g["pass"].to_numpy()
        num += ((x - x.mean()) * (y - y.mean())).sum()
        den += y.mean() * (1 - y.mean()) * ((x - x.mean()) ** 2).sum()
    zs = num / np.sqrt(den) if den > 0 else float("nan")
    log.info("   receptor-stratified trend        z = %+.2f, p = %.4f  (unadjusted was "
             "z = %+.2f, p = %.4f)", zs, 2 * stats.norm.sf(abs(zs)), z, p)

    out = {"gate_pae": gate, "gate_min_sep": MIN_SEP_AS_SELECTED, "n_pairs": int(len(d)),
           "n_pass": int(d["pass"].sum()), "eligible_combinations": int(eligible),
           "coverage_pct": round(100 * len(sheet) / eligible, 2),
           "rate_by_rank_quintile_pct": rates,
           "trend_all5_z": round(z, 3), "trend_all5_p": p,
           "trend_without_Q5_z": round(z4, 3), "trend_without_Q5_p": p4,
           "Q5_vs_rest_OR": round(o5, 3), "Q5_vs_rest_p": p5,
           "Q1_vs_mid_OR": round(o1, 3), "Q1_vs_mid_p": p1,
           "spearman_rank_vs_pae": round(float(stats.spearmanr(d["rank"], d.pae_min_AB)[0]), 4),
           "receptor_composition_share": round(float(analytic / obs), 3),
           "receptor_stratified_z": round(float(zs), 3),
           "receptor_stratified_p": float(2 * stats.norm.sf(abs(zs)))}
    (ROOT / "results" / "selection_vs_gate.json").write_text(json.dumps(out, indent=1) + "\n")

    log.info("\nREADING: the spatial rank carries no information about the gate quantity")
    log.info("(rho %.3f, p = %.2f). There is no graded trend. What is left is a tail excess",
             out["spearman_rank_vs_pae"], stats.spearmanr(d["rank"], d.pae_min_AB)[1])
    log.info("in the worst-ranked fifth, a third of it receptor composition, significant at")
    log.info("p = %.3f once receptors are held fixed. The 5,000 folded pairs are not an",
             2 * stats.norm.sf(abs(zs)))
    log.info("enriched sample -- and they are not a depleted one either. A random draw from")
    log.info("the remaining ~74,000 is still the only thing that measures the selection's")
    log.info("effect directly.")


if __name__ == "__main__":
    main()
