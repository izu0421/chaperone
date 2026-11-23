"""Score the completed orphan co-folds on the same interface terms as the benchmark arms.

The per-fold work is imported from `08_score_armB.py` rather than re-implemented, so a
candidate pair and a benchmark pair are measured by exactly the same code. Only two
things differ here: the output directory, and that there are no labels to join -- these
are the candidates the screen exists to rank.

The screen is still running, so this scores whatever has finished and records how many
that was. Re-run it as more folds land.

Usage:
    micromamba run -n structmod python -I scripts/14_score_orphans.py --workers 24
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("orphan-score")
ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "scripts"))
import score_store as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("score_arm", ROOT / "scripts" / "08_score_armB.py")
_score_arm = importlib.util.module_from_spec(_spec)
# registered before exec so that worker processes can re-import it by name: a module
# loaded only by path cannot be pickled across the process boundary
sys.modules["score_arm"] = _score_arm
_spec.loader.exec_module(_score_arm)
one = _score_arm.one



def score_one(d: Path) -> dict:
    """The shared benchmark scoring, with the fingerprint of what was scored.

    The per-chain confidences used to be computed here, which parsed the CIF a
    second time and gave the candidates a metric the benchmark arms did not get.
    They now come out of `one()` itself, so there is one parse and one definition.

    The fingerprint travels with the row, so a later run can tell a fold whose
    output has not changed (reuse the row) from one that was re-folded under the
    same job name (re-score it).
    """
    return {**one(d), S.FP_COL: S.fold_fingerprint(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="orphan_scores.csv",
                    help="filename under results/; use a different one to score "
                         "without touching the file the figures read")
    ap.add_argument("--seed-from", default=None,
                    help="another scores file under results/ whose still-valid rows "
                         "should be reused; lets a new output inherit a finished run "
                         "instead of re-scoring thousands of unchanged folds")
    ap.add_argument("--full", action="store_true",
                    help="ignore the cache and re-score every fold")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="permit writing fewer folds than the existing result")
    ap.add_argument("--stage-live", choices=("auto", "yes", "no"), default="auto",
                    help="is a fold runner still going? 'auto' reads run/orphan_shard*."
                         "running markers and leaves it unknown if there are none")
    ap.add_argument("--checkpoint-every", type=int, default=400,
                    help="write the partial result every N folds, so an interrupted "
                         "run resumes instead of starting over")
    a = ap.parse_args()

    out_dir = ROOT / "results" / "orphan_out"
    dest = ROOT / "results" / a.out
    manifest = pd.read_csv(ROOT / "work" / "orphan_manifest.csv")
    expected = manifest.job.astype(str).tolist()

    # the denominator is the manifest, never "whatever is on disk": the latter grows
    # as the screen runs, which makes every fraction computed against it unreadable
    if a.stage_live == "auto":
        markers = list((ROOT / "run").glob("orphan_shard*.running")) if (ROOT / "run").is_dir() else []
        # no marker is NOT evidence that nothing is running: the orphan shards were
        # started before the marker convention existed. Leave it unknown.
        live = True if markers else None
    else:
        live = a.stage_live == "yes"
    state = S.classify(out_dir, expected, live)
    log.info("manifest %d folds: %d ready, %d being written, %d queued, %d absent "
             "(runner live: %s)", len(expected), len(state["ready"]),
             len(state["writing"]), len(state["queued"]), len(state["absent"]), live)

    code_fp = S.code_fingerprint([ROOT / "scripts" / f for f in
                                  ("14_score_orphans.py", "08_score_armB.py",
                                   "score_store.py")])
    cache = pd.DataFrame() if a.full else S.load_cache(dest, code_fp)
    if not a.full and a.seed_from:
        seed = ROOT / "results" / a.seed_from
        inherited = S.load_cache(seed, code_fp) if seed.exists() else pd.DataFrame()
        if len(inherited):
            log.info("inherited %d valid rows from %s", len(inherited), seed.name)
            cache = pd.concat([cache, inherited], ignore_index=True).drop_duplicates(
                subset=["pair"], keep="first")
        elif seed.exists():
            log.warning("%s has no rows valid for code %s; scoring from scratch",
                        seed.name, code_fp)
        else:
            log.warning("--seed-from %s does not exist", seed)

    partial = dest.with_suffix(".partial.csv")
    if not a.full and partial.exists():
        # an earlier run of this same code was interrupted; its rows are as good
        # as any, and recovering them is the whole point of checkpointing
        recovered = S.load_cache(partial, code_fp)
        if len(recovered):
            log.info("recovered %d rows from an interrupted run (%s)",
                     len(recovered), partial.name)
            cache = pd.concat([cache, recovered], ignore_index=True).drop_duplicates(
                subset=["pair"], keep="last")
    todo, reused = S.plan(state["ready"], cache)
    log.info("scoring %d folds; reusing %d cached rows (code %s)",
             len(todo), len(reused), code_fp)

    def checkpoint(rows: list[dict]) -> None:
        """Persist what is scored so far, so an interrupted run is not lost.

        Scoring 5,000 folds takes long enough that it gets interrupted -- by a
        kill, a reboot, a machine needed for something else. Writing only at the
        end means every interruption starts over, and a stage that cannot be
        resumed is a stage that silently never gets re-run after the code
        changes. The checkpoint carries the fingerprints, so the next run reuses
        it through the ordinary cache path.
        """
        part = pd.DataFrame(rows)
        if part.empty:
            return
        part[S.CODE_COL] = code_fp
        combined = pd.concat([reused, part], ignore_index=True) if len(reused) else part
        tmp = dest.with_suffix(".partial.csv")
        combined.to_csv(tmp.with_suffix(".csv.tmp"), index=False)
        os.replace(tmp.with_suffix(".csv.tmp"), tmp)

    rows, done = [], 0
    if todo:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futures = [ex.submit(score_one, out_dir / n) for n in todo]
            for f in as_completed(futures):
                rows.append(f.result())
                done += 1
                if done % 250 == 0:
                    log.info("  scored %d/%d", done, len(todo))
                if a.checkpoint_every and done % a.checkpoint_every == 0:
                    checkpoint(rows)

    fresh = pd.DataFrame(rows)
    if not fresh.empty:
        fresh[S.CODE_COL] = code_fp
    scored = pd.concat([reused, fresh], ignore_index=True) if len(reused) else fresh
    if scored.empty:
        raise SystemExit("no folds scored and no cache to carry over; nothing to write")

    # a fold that has not finished must not appear as a row at all. A row with NaN
    # terms is what a genuinely failed fold looks like, and mixing the two is how
    # "not measured" gets read as "measured absent"
    scored = scored[scored.pair.isin(state["ready"])].copy()

    meta = ["receptor", "partner", "kind", "msa_source", "p_twoway", "rank"]
    scored = scored.drop(columns=[c for c in meta if c in scored.columns]).merge(
        manifest.rename(columns={"job": "pair"})[["pair", *meta]], on="pair", how="left")

    missing = int(scored.receptor.isna().sum())
    if missing:
        log.warning("%d scored folds are not in the manifest", missing)

    by_kind = scored.kind.value_counts().to_dict()
    S.write(dest, scored, state, code_fp, len(expected),
            allow_shrink=a.allow_shrink,
            extra={"scored_by_kind": by_kind,
                   "reused_rows": int(len(reused)), "newly_scored": int(len(fresh))})

    partial.unlink(missing_ok=True)   # the full result supersedes the checkpoint

    noted = scored[scored.note.fillna("").ne("")]
    log.info("\n%d folds with a note:\n%s", len(noted),
             noted.note.value_counts().head(6).to_string() if len(noted) else "  none")

    # a contact count means nothing without a clash check: an un-relaxed trajectory can
    # report hundreds of contacts at a separation no real interface could adopt
    usable = scored[scored.min_sep.notna()]
    log.info("\nclash check: %d of %d folds have min_sep < 1.5 A",
             int((usable.min_sep < 1.5).sum()), len(usable))
    cols = [c for c in ["iptm", "bsa", "n_contact", "iface_plddt", "plddt_B",
                        "iface_plddt_B"] if c in scored.columns]
    log.info("\n%s", scored[cols].describe().round(2).to_string())

    # the comparison the literature says to make: does the partner chain's own
    # confidence separate the two candidate kinds differently from ipTM?
    if {"kind", "plddt_B"} <= set(scored.columns):
        log.info("\nby candidate kind (n = %s):\n%s", by_kind,
                 scored.groupby("kind")[[c for c in ("iptm", "iface_plddt", "plddt_B",
                                                     "iface_plddt_B") if c in scored.columns]]
                 .median().round(3).to_string())


if __name__ == "__main__":
    main()
