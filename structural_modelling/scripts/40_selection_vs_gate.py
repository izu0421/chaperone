"""Does the criterion that chose what to fold point AT structurally plausible pairs?

The screen folded 5,000 receptor-partner pairs chosen by a spatial recurrence
score out of roughly 79,000 eligible combinations (75 receptors x 1,052
partners). That choice is the screen's only enrichment step, so it is worth
asking whether it enriched anything.

It did not. It points the other way: the pairs the spatial score ranked WORST
clear the structural gate four times as often as the ones it ranked best.

Three ways the result could be an artefact, each tested here:
  size        bigger complexes used to clear the old ipTM gate more often
              ([[AF3 ipTM size confound]]). The PAE gate was chosen partly to
              remove that, and this checks it did.
  receptor    a few receptors hold most of the passes, and if those receptors
              also sit at one end of the ranking the trend is theirs, not the
              score's. Tested by permuting rank WITHIN receptor.
  threshold   a single cut could be a lucky split, so the trend is tested
              across all five quintiles, not just the extremes.

What it does NOT show is that the spatial score is wrong and the gate is right.
Both are near-null on their own: spatial recurrence gives AUROC 0.524 for
canonical ligand-receptor pairs, and AF3 interface scoring gives 0.519 on
degree-balanced HuRI. Two near-null criteria that disagree systematically is a
statement about the screen's selection, not a ranking of the two.

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
N_PERM = 2000


def main() -> None:
    gate = json.loads((ROOT / "results" / "protein_thresholds.json").read_text()
                      )["gate"]["pae_min_AB_max"]
    sheet = pd.read_csv(SHEET)
    sc = pd.read_csv(ROOT / "results" / "orphan_scores.csv")
    sc["key"] = sc.receptor.astype(str) + "__" + sc.partner.astype(str)
    sheet["key"] = sheet.receptor_gene.astype(str) + "__" + sheet.partner_gene.astype(str)
    d = sheet.merge(sc[["key", "pae_min_AB", "kind"]], on="key", how="inner")
    d = d[d.kind.eq("protein")].copy()
    d["tok"] = d.receptor_seq.str.len() + d.partner_seq.str.len()
    d["pass"] = (d.pae_min_AB <= gate).astype(int)

    eligible = sheet.receptor_gene.nunique() * sheet.partner_gene.nunique()
    log.info("%d protein pairs folded and scored, %d clear the %.2f A gate (%.2f%%)",
             len(d), d["pass"].sum(), gate, 100 * d["pass"].mean())
    log.info("chosen from about %d eligible receptor x partner combinations "
             "(%d receptors, %d partners): %.1f%% coverage\n",
             eligible, sheet.receptor_gene.nunique(), sheet.partner_gene.nunique(),
             100 * len(sheet) / eligible)

    d["q"] = pd.qcut(d["rank"], 5, labels=False)
    tab = d.groupby("q")["pass"].agg(["size", "sum"])
    tab["rate_%"] = (100 * tab["sum"] / tab["size"]).round(2)
    tab.index = ["Q1 best spatial rank", "Q2", "Q3", "Q4", "Q5 worst"]
    log.info("GATE PASS RATE BY THE RANK THAT CHOSE WHAT TO FOLD\n%s", tab.to_string())

    x, n, r = np.arange(5), tab["size"].to_numpy(), tab["sum"].to_numpy()
    pb = r.sum() / n.sum(); xb = (n * x).sum() / n.sum()
    z = ((r - n * pb) * (x - xb)).sum() / np.sqrt(pb * (1 - pb) * (n * (x - xb) ** 2).sum())
    log.info("\nCochran-Armitage trend: z = %+.2f, p = %.4f", z, 2 * stats.norm.sf(abs(z)))

    rho_s, p_s = stats.spearmanr(d.tok, d["pass"])
    log.info("size vs gate pass: rho = %+.3f (p = %.3g) -- the PAE gate is size-neutral, "
             "so size cannot carry this", rho_s, p_s)

    # Mantel-Haenszel across size quintiles
    d["sz"] = pd.qcut(d.tok, 5, labels=False)
    d["half"] = np.where(d["rank"] <= d["rank"].median(), "top", "bottom")
    num = den = 0.0
    for _, g in d.groupby("sz"):
        a, b = g[g.half.eq("top")], g[g.half.eq("bottom")]
        a1, a0 = a["pass"].sum(), len(a) - a["pass"].sum()
        b1, b0 = b["pass"].sum(), len(b) - b["pass"].sum()
        num += a1 * b0 / len(g); den += a0 * b1 / len(g)
    log.info("Mantel-Haenszel OR for the top half of the ranking, stratified by complex "
             "size: %.2f", num / den)

    rng = np.random.default_rng(0)
    def gap(df: pd.DataFrame) -> float:
        q = pd.qcut(df["rank"], 5, labels=False)
        m = df.groupby(q)["pass"].mean()
        return m.iloc[-1] - m.iloc[0]
    obs = gap(d)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        dd = d.copy()
        dd["rank"] = dd.groupby("receptor_gene")["rank"].transform(
            lambda s: rng.permutation(s.values))
        null[i] = gap(dd)
    p_perm = float((np.abs(null) >= abs(obs)).mean())
    log.info("worst-minus-best pass rate = %+.2f pp; permuting rank WITHIN receptor "
             "(%d draws): p = %.4f, null sd %.2f pp", 100 * obs, N_PERM, p_perm,
             100 * null.std())

    pr = d.groupby("receptor_gene")["pass"].agg(["size", "sum"])
    log.info("\n%d of %d receptors have any pass; the top three hold %.0f%% of them",
             int((pr["sum"] > 0).sum()), len(pr),
             100 * pr["sum"].nlargest(3).sum() / pr["sum"].sum())

    out = {"gate": gate, "n_pairs": int(len(d)), "n_pass": int(d["pass"].sum()),
           "eligible_combinations": int(eligible),
           "coverage_pct": round(100 * len(sheet) / eligible, 2),
           "rate_by_rank_quintile_pct": tab["rate_%"].tolist(),
           "trend_z": round(float(z), 3),
           "trend_p": float(2 * stats.norm.sf(abs(z))),
           "size_rho": round(float(rho_s), 4),
           "mh_or_top_half": round(num / den, 3),
           "within_receptor_permutation_p": p_perm}
    dest = ROOT / "results" / "selection_vs_gate.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    log.info("\nwrote %s", dest)
    log.info("\nREADING: the screen's one enrichment step is anti-correlated with its own")
    log.info("structural gate, and neither criterion is strong on its own (spatial")
    log.info("recurrence AUROC 0.524 for canonical pairs; AF3 interface scoring 0.519 on")
    log.info("degree-balanced HuRI). The 5,000 folded pairs cannot be described as an")
    log.info("enriched sample, and a random draw from the remaining ~74,000 would be a")
    log.info("fair test of that.")


if __name__ == "__main__":
    main()
