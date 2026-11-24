"""Score the HuRI pilot and ask whether the protein gate survives a benchmark
built to remove the defect the first one has.

The 2,500-pair RoseTTAFold2-PPI benchmark carries one defect that cannot be
corrected inside it: its positives are drawn from proteins 9.6-fold better
studied than its negatives, and re-weighting the negatives to match leaves an
effective sample of 57 of 2,000. Every threshold in this project's protein arm
is fitted on that benchmark, including the interface PAE cut.

HuRI is the replacement: a systematic all-by-all yeast two-hybrid screen in
which the negatives are proteins that each scored at least one interaction in
the SAME assay, so both classes have been looked at. The pilot is 600 of those
pairs, 120 positive and 480 negative, at the same 1:4 prevalence, folded by the
same code.

The question is narrow and worth asking on its own: does the operating point
fitted on the study-biased benchmark hold here? Three outcomes are informative.
If the cut lands in the same place, the gate transfers and the study bias did
not set it. If it lands elsewhere, the gate is partly an artefact of who has
been studied. If nothing separates at all, the first benchmark's discrimination
was the study bias.

Usage:
    micromamba run -n structmod python -I scripts/28_score_huri_pilot.py --workers 16
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("huri")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import score_store as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("sc14", ROOT / "scripts" / "14_score_orphans.py")
sc14 = importlib.util.module_from_spec(_spec)
sys.modules["sc14"] = sc14
_spec.loader.exec_module(sc14)
score_one = sc14.score_one

MIN_SEP = 1.5
PUBLISHED = dict(iptm=0.5, bsa=800, iface_plddt=70)


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_auroc(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    v = [auroc(rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True))
         for _ in range(n)]
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="huri_pilot_scores.csv")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--allow-shrink", action="store_true")
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "work" / "huri_pilot_manifest.csv")
    out_dir = ROOT / "results" / "huri_pilot_out"
    dest = ROOT / "results" / a.out
    expected = man.job.astype(str).tolist()
    markers = list((ROOT / "run").glob("huri*.running")) if (ROOT / "run").is_dir() else []
    state = S.classify(out_dir, expected, True if markers else None)
    log.info("manifest %d folds: %d ready, %d queued, %d absent",
             len(expected), len(state["ready"]), len(state["queued"]), len(state["absent"]))
    if not state["ready"]:
        raise SystemExit("nothing folded yet")

    code_fp = S.code_fingerprint([ROOT / "scripts" / f for f in
                                  ("28_score_huri_pilot.py", "14_score_orphans.py",
                                   "score_store.py")])
    cache = pd.DataFrame() if a.full else S.load_cache(dest, code_fp)
    todo, reused = S.plan(state["ready"], cache)
    log.info("scoring %d folds, reusing %d", len(todo), len(reused))
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
    df["no_clash"] = df.min_sep >= MIN_SEP
    S.write(dest, df, state, code_fp, len(expected), allow_shrink=a.allow_shrink)

    y = df.cat.eq("positive").to_numpy()
    log.info("\n%d scored: %d positive, %d negative", len(df), int(y.sum()), int((~y).sum()))

    # ---- discrimination, per metric -----------------------------------------
    log.info("\nAUROC on the HuRI pilot, beside the same metric on the first benchmark:")
    bench = pd.read_csv(ROOT / "results" / "armB_scores_labelled.csv")
    yb = bench.cat.eq("positive").to_numpy()
    for f, hb in (("pae_min_AB", False), ("iptm", True), ("iface_plddt", True), ("bsa", True)):
        v = df[f].to_numpy(float)
        au = auroc(v[y] if hb else -v[y], v[~y] if hb else -v[~y])
        lo, hi = boot_auroc(v[y] if hb else -v[y], v[~y] if hb else -v[~y])
        vb = bench[f].to_numpy(float)
        aub = auroc(vb[yb] if hb else -vb[yb], vb[~yb] if hb else -vb[~yb])
        log.info("  %-14s HuRI %.3f [%.3f, %.3f]   first benchmark %.3f   delta %+.3f",
                 f, au, lo, hi, aub, au - aub)

    # ---- does the fitted cut transfer? --------------------------------------
    cfg = json.loads((ROOT / "results" / "protein_thresholds.json").read_text())
    cut = cfg["gate"]["pae_min_AB_max"]
    target = cfg["published_gate"]["decoy_rate"]

    sel = (df.pae_min_AB.le(cut) & df.no_clash).to_numpy()
    log.info("\nthe cut fitted on the first benchmark (interface PAE <= %.2f A), applied here:",
             cut)
    log.info("  recall %.1f%%, decoys %.2f%%, precision %.1f%%",
             100 * sel[y].mean(), 100 * sel[~y].mean(),
             100 * y[sel].mean() if sel.sum() else float("nan"))

    # what the cut WOULD be if fitted here, at the same decoy rate
    score = np.where(df.no_clash, -df.pae_min_AB.to_numpy(float), -np.inf)
    here = float(np.quantile(score[~y], 1 - target))
    rng = np.random.default_rng(0)
    neg = score[~y]
    boots = [np.quantile(rng.choice(neg, len(neg), True), 1 - target) for _ in range(2000)]
    lo, hi = -np.percentile(boots, 97.5), -np.percentile(boots, 2.5)
    log.info("  refitted on HuRI at the same %.2f%% decoy rate: %.2f A "
             "(95%% bootstrap %.2f to %.2f)", 100 * target, -here, lo, hi)
    log.info("  the fitted cut %s inside that interval",
             "falls" if lo <= cut <= hi else "FALLS OUTSIDE")

    pub = (df.iptm.ge(PUBLISHED["iptm"]) & df.bsa.ge(PUBLISHED["bsa"])
           & df.iface_plddt.ge(PUBLISHED["iface_plddt"]) & df.no_clash).to_numpy()
    log.info("\nthe published gate here: recall %.1f%%, decoys %.2f%%, precision %.1f%%",
             100 * pub[y].mean(), 100 * pub[~y].mean(),
             100 * y[pub].mean() if pub.sum() else float("nan"))

    (ROOT / "results" / "huri_pilot_transfer.json").write_text(json.dumps({
        "n": int(len(df)), "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
        "fitted_cut": cut, "refitted_here": round(-here, 2),
        "refit_ci": [round(lo, 2), round(hi, 2)],
        "transfers": bool(lo <= cut <= hi),
        "pae_gate_here": {"recall": round(float(sel[y].mean()), 3),
                          "decoy_rate": round(float(sel[~y].mean()), 4),
                          "precision": round(float(y[sel].mean()), 3) if sel.sum() else None},
        "published_gate_here": {"recall": round(float(pub[y].mean()), 3),
                                "decoy_rate": round(float(pub[~y].mean()), 4),
                                "precision": round(float(y[pub].mean()), 3) if pub.sum() else None},
    }, indent=1) + "\n")
    log.info("\nwrote %s and huri_pilot_transfer.json", dest)


if __name__ == "__main__":
    main()
