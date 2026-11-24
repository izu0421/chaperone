"""Score the protein counter-screen and derive the protein arm's operating point.

Written BEFORE the folds landed, so the read-outs are not chosen after seeing
them. The question is the one the screen cannot answer about itself: what does
a co-folding score look like for a GPCR complex whose answer is known, and does
the threshold the screen uses transfer?

Three read-outs, in the order they should be believed:

  per-receptor rank   where a receptor's own chemokines land among the 16 it was
                      folded against. The deorphanisation question asked
                      directly, and better powered than a pooled AUROC: a
                      matrix-wide ROC flatters the promiscuous receptors, and
                      CXCR3 alone binds 7 of the 16.
  operating point     for each metric, the threshold at which precision on this
                      panel reaches 50%, with its bootstrap interval. This is
                      what the orphan protein arm is missing; the metabolite arm
                      has had one since the 256-fold panel.
  bundle block        do known heterodimers outscore receptor pairs with no
                      dimerisation evidence? GPR62 + ADORA1 passed every rule
                      the project has, including the face-matching topology
                      rule, and nothing in the screen tests whether AlphaFold3
                      packs any two seven-helix bundles that way.

The detection floor is printed BEFORE any result, because a null is only
evidence with its minimum detectable effect attached.

Usage:
    micromamba run -n structmod python -I scripts/26_score_protein_counterscreen.py --workers 24
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pcs-score")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import score_store as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("sc14", ROOT / "scripts" / "14_score_orphans.py")
sc14 = importlib.util.module_from_spec(_spec)
sys.modules["sc14"] = sc14
_spec.loader.exec_module(sc14)
score_one = sc14.score_one

# higher is better for all of these; interface PAE is negated at use so every
# metric points the same way
METRICS = ["iptm", "iptm_AB", "ranking_score", "iface_plddt", "plddt_B",
           "iface_plddt_B", "plddt_A", "bsa", "n_contact"]
LOWER_BETTER = {"pae_min_AB"}
N_BOOT = 2000


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC, ties counted as half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_auroc(pos, neg, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = [auroc(rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True))
            for _ in range(n)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def detection_floor(n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil: the smallest AUROC separable from 0.5 at this sample size."""
    a = 0.5
    se = np.sqrt((a * (1 - a) + (n_pos - 1) * (a / (2 - a) - a ** 2)
                  + (n_neg - 1) * (2 * a ** 2 / (1 + a) - a ** 2)) / (n_pos * n_neg))
    return float(0.5 + 1.96 * se)


def operating_point(values: np.ndarray, y: np.ndarray, higher_better: bool,
                    target_precision: float = 0.5):
    """The threshold at which precision on this panel first reaches the target.

    Returns None when no threshold reaches it, which is itself a result: buried
    area and contact count reach no such threshold on the metabolite arm, which
    is why neither is in that gate.
    """
    v = values if higher_better else -values
    ok = ~np.isnan(v)
    v, yy = v[ok], y[ok]
    best = None
    for cut in np.unique(v):
        sel = v >= cut
        if sel.sum() == 0:
            continue
        prec = yy[sel].mean()
        if prec >= target_precision:
            best = {"cut": float(cut if higher_better else -cut),
                    "precision": float(prec),
                    "recall": float(yy[sel].sum() / yy.sum()),
                    "neg_rate": float((~yy.astype(bool))[sel].sum() / (~yy.astype(bool)).sum())}
            break                      # unique() ascends: the first is the loosest
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="protein_counterscreen_scores.csv")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--allow-shrink", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report ranks on an unfinished matrix. Off by default: a "
                         "partial matrix inflates every rank, because a receptor's "
                         "own chemokine is ranked against however many competitors "
                         "happen to have landed")
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "work" / "protein_counterscreen_manifest.csv")
    out_dir = ROOT / "results" / "protein_counterscreen_out"
    dest = ROOT / "results" / a.out
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} does not exist; nothing has been folded yet")

    expected = man.job.astype(str).tolist()
    markers = list((ROOT / "run").glob("pcs_shard*.running")) if (ROOT / "run").is_dir() else []
    state = S.classify(out_dir, expected, True if markers else None)
    log.info("manifest %d folds: %d ready, %d being written, %d queued, %d absent",
             len(expected), len(state["ready"]), len(state["writing"]),
             len(state["queued"]), len(state["absent"]))
    if not state["ready"]:
        raise SystemExit("nothing to score yet")

    code_fp = S.code_fingerprint([ROOT / "scripts" / f for f in
                                  ("26_score_protein_counterscreen.py",
                                   "14_score_orphans.py", "score_store.py")])
    cache = pd.DataFrame() if a.full else S.load_cache(dest, code_fp)
    todo, reused = S.plan(state["ready"], cache)
    log.info("scoring %d folds; reusing %d cached rows (code %s)",
             len(todo), len(reused), code_fp)

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(score_one, [out_dir / n for n in todo]))
    fresh = pd.DataFrame(rows)
    if not fresh.empty:
        fresh[S.FP_COL] = fresh.pair.map(state["ready"])
        fresh[S.CODE_COL] = code_fp
    df = pd.concat([reused, fresh], ignore_index=True) if len(reused) else fresh
    df = df[df.pair.isin(state["ready"])].copy()
    df = df.drop(columns=[c for c in man.columns if c in df.columns and c != "pair"]).merge(
        man, left_on="pair", right_on="job", how="left")

    panel = man[man.block.eq("A_chemokine")].groupby("chain_a").job.size().rename("panel_n")
    have = df[df.block.eq("A_chemokine")].groupby("chain_a").pair.size().rename("have_n")
    cov = pd.concat([panel, have], axis=1).fillna(0)
    cov["complete"] = cov.have_n >= cov.panel_n
    df = df.merge(cov[["panel_n", "have_n", "complete"]], left_on="chain_a",
                  right_index=True, how="left")
    n_complete = int(cov.complete.sum())
    log.info("%d of %d receptors have their full chemokine panel folded",
             n_complete, len(cov))

    S.write(dest, df, state, code_fp, len(expected), allow_shrink=a.allow_shrink,
            extra={"receptors_with_full_panel": n_complete,
                   "receptors_total": int(len(cov)),
                   "reused_rows": int(len(reused)), "newly_scored": int(len(fresh))})

    complete = len(state["ready"]) == len(expected)
    if not complete and not a.allow_partial:
        log.warning("\nMATRIX INCOMPLETE (%d of %d). Scores are written, but the "
                    "sections below are restricted to receptors whose full panel "
                    "has landed.", len(state["ready"]), len(expected))

    A = df[df.block.eq("A_chemokine")]
    if not (complete or a.allow_partial):
        A = A[A.complete.fillna(False)]
    if A.empty:
        log.warning("no receptor has a complete panel yet; stopping before the "
                    "rank and threshold sections")
        return

    pos = A[A.label.eq("binder")]
    neg = A[A.label.eq("negative")]
    log.info("\nblock A: %d binder, %d negative cells over %d receptors",
             len(pos), len(neg), A.chain_a.nunique())
    floor = detection_floor(len(pos), len(neg))
    log.info("detection floor at this sample size: AUROC %.3f "
             "(anything below is indistinguishable from chance)", floor)

    # ---- per-receptor rank -------------------------------------------------
    log.info("\nper-receptor rank of the known binders (1 = top of its own panel):")
    rank_rows = []
    for metric in ("iptm", "iface_plddt", "bsa"):
        ranks = []
        for rec, grp in A.groupby("chain_a"):
            order = grp.sort_values(metric, ascending=False).reset_index(drop=True)
            for i, r in order.iterrows():
                if r.label == "binder":
                    ranks.append(i + 1)
        if ranks:
            n_lig = A.chain_b.nunique()
            rank_rows.append({"metric": metric, "n_binders": len(ranks),
                              "median_rank": float(np.median(ranks)),
                              "chance_median": (n_lig + 1) / 2,
                              "top1": int(sum(r == 1 for r in ranks)),
                              "top3": int(sum(r <= 3 for r in ranks))})
    ranks_df = pd.DataFrame(rank_rows)
    log.info("\n%s", ranks_df.to_string(index=False))
    ranks_df.to_csv(ROOT / "results" / "protein_counterscreen_ranks.csv", index=False)

    # ---- AUROC and the operating point -------------------------------------
    y = A.label.eq("binder").to_numpy().astype(int)
    out = []
    for metric in METRICS + list(LOWER_BETTER):
        if metric not in A:
            continue
        hb = metric not in LOWER_BETTER
        v = A[metric].to_numpy(float)
        au = auroc(v[y == 1], v[y == 0]) if hb else auroc(-v[y == 1], -v[y == 0])
        lo, hi = boot_auroc(v[y == 1] if hb else -v[y == 1],
                            v[y == 0] if hb else -v[y == 0])
        op = operating_point(v, y, hb)
        out.append({"metric": metric, "auroc": round(au, 3), "ci_lo": round(lo, 3),
                    "ci_hi": round(hi, 3), "floor": round(floor, 3),
                    "beats_floor": bool(au > floor),
                    "cut": None if op is None else round(op["cut"], 3),
                    "precision": None if op is None else round(op["precision"], 3),
                    "recall": None if op is None else round(op["recall"], 3),
                    "neg_rate": None if op is None else round(op["neg_rate"], 3)})
    res = pd.DataFrame(out).sort_values("auroc", ascending=False)
    log.info("\nAUROC and the 50%%-precision operating point on this panel:\n%s",
             res.to_string(index=False))
    res.to_csv(ROOT / "results" / "protein_counterscreen_thresholds.csv", index=False)
    no_point = res[res.cut.isna()].metric.tolist()
    if no_point:
        log.info("no threshold reaches 50%% precision for: %s — so none of these can "
                 "gate this arm at any value", ", ".join(no_point))

    # ---- block B: does AF3 pack any two bundles? ---------------------------
    B = df[df.block.eq("B_bundle")]
    if len(B):
        log.info("\nblock B, seven-helix bundle against seven-helix bundle:")
        for lab in ("binder", "negative", "test"):
            sub = B[B.label.eq(lab)]
            if len(sub):
                log.info("  %-9s n = %2d   median interface pTM %.2f   median BSA %.0f",
                         lab, len(sub), sub.iptm.median(), sub.bsa.median())
        d_pos, d_neg = B[B.label.eq("binder")], B[B.label.eq("negative")]
        if len(d_pos) and len(d_neg):
            au = auroc(d_pos.iptm.to_numpy(float), d_neg.iptm.to_numpy(float))
            fl = detection_floor(len(d_pos), len(d_neg))
            log.info("  known heterodimers vs non-dimer receptor pairs: AUROC %.3f "
                     "(floor %.3f at n = %d / %d)", au, fl, len(d_pos), len(d_neg))
        for r in B[B.label.eq("test")].itertuples():
            above = int((d_neg.iptm > r.iptm).sum()) if len(d_neg) else -1
            log.info("  %s: interface pTM %.2f, above %d of %d non-dimer pairs",
                     r.job, r.iptm, len(d_neg) - above, len(d_neg))
    log.info("\nwrote %s and the two summary tables", dest)


if __name__ == "__main__":
    main()
