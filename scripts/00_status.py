"""One honest view of every stage in the pipeline.

Written because "why is the screen not moving forward?" took a dozen ad-hoc
commands to answer, and because the answer turned out to be that it was moving
at full speed -- the folds were simply getting bigger, so the counter crawled
while the GPU-seconds were unchanged. A completion count alone cannot show that.
This prints the count, the instantaneous rate, and the size of what is left,
which together can.

Adding a stage means adding one entry to STAGES. Everything below is generic, so
a new benchmark is visible here the moment its manifest exists, instead of being
invisible until someone writes a bespoke progress command for it.

Usage:
    micromamba run -n structmod python -I scripts/00_status.py
    micromamba run -n structmod python -I scripts/00_status.py --stage orphan --window-h 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import score_store as S  # noqa: E402


@dataclass
class Stage:
    name: str
    manifest: str          # under work/
    out: str               # under results/
    scores: str | None     # under results/, None if the stage has no scorer yet
    marker: str            # glob under run/ that a live shard touches
    code: tuple[str, ...]  # scripts whose edit invalidates the scores
    size_col: str = "size"
    job_col: str = "job"


STAGES = [
    Stage("orphan", "orphan_manifest.csv", "orphan_out", "orphan_scores.csv",
          "orphan_shard*.running",
          ("14_score_orphans.py", "08_score_armB.py", "score_store.py")),
    Stage("specificity", "specificity_manifest.csv", "specificity_out",
          "specificity_scores.csv", "specificity_shard*.running",
          ("18_score_specificity.py", "08_score_armB.py", "score_store.py")),
    Stage("huri_pilot", "huri_pilot_manifest.csv", "huri_pilot_out",
          "huri_pilot_scores.csv", "huri_shard*.running",
          ("08_score_armB.py", "score_store.py")),
]


def rate_and_eta(out_dir: Path, ready: list[str], queued_sizes: pd.Series,
                 done_sizes: pd.Series, window_h: float) -> dict:
    """Folds per hour over a recent window, and an ETA that respects size.

    A rate in folds/h is misleading on its own when the queue is size-ordered: a
    stage can be running at constant GPU throughput while folds/h collapses,
    because each remaining fold is bigger. So the ETA is computed in cost units
    (AF3 cost grows steeply with token count; the exponent used in this project's
    own sizing is 2.79) and only then converted back to hours.
    """
    now = time.time()
    stamps = []
    for name in ready:
        f = out_dir / name / f"{name}_summary_confidences.json"
        try:
            stamps.append(f.stat().st_mtime)
        except OSError:
            continue
    if not stamps:
        return {}
    stamps = np.array(sorted(stamps))
    recent = stamps[stamps > now - window_h * 3600]
    out = {"folds_last_window": int(len(recent)),
           "rate_per_h": round(len(recent) / window_h, 1),
           "last_fold_min_ago": round((now - stamps[-1]) / 60, 1)}

    if len(queued_sizes) == 0 or len(recent) < 3 or done_sizes.empty:
        return out
    # cost-weighted: what fraction of the remaining work the recent rate represents
    EXP = 2.79
    recent_names = [n for n in ready
                    if (out_dir / n / f"{n}_summary_confidences.json").exists()
                    and (out_dir / n / f"{n}_summary_confidences.json").stat().st_mtime
                    > now - window_h * 3600]
    done_recent = done_sizes.reindex(recent_names).dropna()
    if done_recent.empty:
        return out
    cost_done = float((done_recent ** EXP).sum())
    cost_left = float((queued_sizes ** EXP).sum())
    cost_per_h = cost_done / window_h
    if cost_per_h > 0:
        out["eta_h_cost_weighted"] = round(cost_left / cost_per_h, 1)
        out["eta_h_naive_count"] = round(len(queued_sizes) / max(out["rate_per_h"], 1e-9), 1)
    return out


def report(st: Stage, window_h: float) -> dict:
    man_path = ROOT / "work" / st.manifest
    out_dir = ROOT / "results" / st.out
    line: dict = {"stage": st.name}
    if not man_path.exists():
        return {**line, "state": "no manifest -- stage not built yet"}
    man = pd.read_csv(man_path)
    expected = man[st.job_col].astype(str).tolist()
    if not out_dir.exists():
        return {**line, "state": f"manifest has {len(expected)} folds, nothing folded yet"}

    markers = sorted((ROOT / "run").glob(st.marker)) if (ROOT / "run").is_dir() else []
    live = True if markers else None
    state = S.classify(out_dir, expected, live)

    # Work that cannot be done is not work that is pending. 85 orphan jobs sat in
    # the queue for four days because their inputs had no embedded MSA, which
    # --norun_data_pipeline cannot supply; the screen reported "96 remaining" when
    # 85 of those were impossible, and an ETA built on that number was meaningless.
    unfoldable, retired = set(), set()
    in_dir = {"orphan": "orphan_inputs_omic", "specificity": "specificity_inputs",
              "huri_pilot": "huri_pilot_inputs"}.get(st.name)
    if in_dir and (ROOT / "work" / in_dir).is_dir():
        try:
            import importlib.util as _iu
            _s = _iu.spec_from_file_location("chk", ROOT / "scripts" / "check_af3_inputs.py")
            _chk = _iu.module_from_spec(_s)
            _s.loader.exec_module(_chk)
            # only the jobs that have not produced output: validating the ones
            # already folded would read tens of GB to learn nothing
            pending = set(state["queued"]) | set(state["writing"])
            _, bad = _chk.check(ROOT / "work" / in_dir, only=pending)
            unfoldable = set(bad)
        except Exception:  # noqa: BLE001  -- the check is a courtesy, not a gate
            pass
    rf = ROOT / "run" / f"{st.name}_retired.txt"
    if rf.exists():
        retired = set(rf.read_text().split())

    sizes = man.set_index(st.job_col)[st.size_col] if st.size_col in man.columns else pd.Series(dtype=float)

    # Two different things, and conflating them misleads in opposite directions.
    # An unfoldable input is a BUG to fix. A retired job is a DECISION already
    # taken. The first should be fixed and folded; the second should be left
    # alone and reported as a gap in coverage.
    bad_input = (unfoldable - retired) - set(state["ready"])
    dropped = retired - set(state["ready"])
    blocked = bad_input | dropped
    foldable_total = len(expected) - len(blocked)
    really_queued = [n for n in state["queued"] if n not in blocked]
    line.update({
        "manifest": len(expected),
        "ready": len(state["ready"]),
        "pct_of_manifest": round(100 * len(state["ready"]) / len(expected), 1),
        "writing": len(state["writing"]),
        "queued": len(really_queued),
        "dropped_by_decision": len(dropped),
        "blocked_bad_input": len(bad_input),
        "still_in_scope": foldable_total,
        "pct_of_in_scope": round(100 * len(state["ready"]) / max(foldable_total, 1), 1),
        "shards_live": len(markers) if markers else ("unknown" if live is None else 0),
    })
    if len(sizes):
        q = sizes.reindex(really_queued).dropna()
        dn = sizes.reindex(list(state["ready"])).dropna()
        if len(q):
            line["queued_median_tokens"] = int(q.median())
        if len(dn):
            line["done_median_tokens"] = int(dn.median())
        line.update(rate_and_eta(out_dir, list(state["ready"]), q, dn, window_h))

    if st.scores:
        sc = ROOT / "results" / st.scores
        if not sc.exists():
            line["scores"] = "not written yet"
        else:
            cv = sc.with_suffix(".coverage.json")
            if cv.exists():
                c = json.loads(cv.read_text())
                line["scores"] = f"{c['scored']} folds, complete={c['complete']}"
                cur = S.code_fingerprint([ROOT / "scripts" / f for f in st.code])
                line["scores_stale_code"] = c.get("code_fingerprint") != cur
            else:
                # the condition this project has been burnt by: a results file with
                # no record of its own coverage, older than the code that made it
                n = len(pd.read_csv(sc, usecols=lambda c: c == "pair"))
                newest = max((f.stat().st_mtime for f in
                              (ROOT / "scripts" / x for x in st.code) if f.exists()),
                             default=0)
                line["scores"] = f"{n} folds, NO coverage record"
                line["scores_stale_code"] = sc.stat().st_mtime < newest
            if line.get("scores_stale_code"):
                line["scores"] += "  <-- OLDER THAN ITS SCORING CODE"
            if isinstance(line.get("scores"), str) and line["ready"] > 0:
                try:
                    n_scored = int(line["scores"].split()[0])
                    if n_scored < line["ready"]:
                        line["scores"] += f"  <-- {line['ready'] - n_scored} folds unscored"
                except ValueError:
                    pass
    return line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="just this one")
    ap.add_argument("--window-h", type=float, default=1.0,
                    help="window for the fold rate, in hours")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    rows = [report(s, a.window_h) for s in STAGES
            if a.stage is None or s.name == a.stage]
    if a.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"pipeline status  {time.strftime('%F %T')}   (rate over the last "
          f"{a.window_h:g} h)\n")
    for r in rows:
        print(f"  {r['stage']}")
        for k, v in r.items():
            if k != "stage":
                print(f"      {k:24s} {v}")
        print()

    notes = []
    for r in rows:
        if r.get("scores_stale_code"):
            notes.append(f"{r['stage']}: the scores predate the scoring code -- re-score "
                         f"before quoting any number from them")
        if r.get("queued") and r.get("shards_live") == 0:
            notes.append(f"{r['stage']}: {r['queued']} folds queued and no shard alive -- "
                         f"nothing is going to finish them")
        if r.get("blocked_bad_input"):
            notes.append(f"{r['stage']}: {r['blocked_bad_input']} jobs have inputs AlphaFold3 "
                         f"would reject (run check_af3_inputs.py, then "
                         f"13_repair_unfoldable_inputs.py). That is a bug to fix, not "
                         f"pending work, and an ETA counting them is wrong")
        if r.get("dropped_by_decision"):
            notes.append(f"{r['stage']}: {r['dropped_by_decision']} folds were dropped on "
                         f"purpose (run/{r['stage']}_retired.txt; see "
                         f"results/orphan_dropped.csv for what was given up and why). "
                         f"Coverage against the manifest is "
                         f"{r.get('pct_of_manifest')}%, not {r.get('pct_of_in_scope')}% -- "
                         f"quote the first")
        if (r.get("queued_median_tokens") and r.get("done_median_tokens")
                and r["queued_median_tokens"] > 2 * r["done_median_tokens"]):
            notes.append(f"{r['stage']}: the queue's median fold is "
                         f"{r['queued_median_tokens'] / r['done_median_tokens']:.1f}x the size "
                         f"of what is done, so folds/h will keep falling at constant "
                         f"GPU throughput")
    if notes:
        print("  notes")
        for n in notes:
            print(f"    - {n}")


if __name__ == "__main__":
    main()
