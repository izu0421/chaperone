"""Incremental, coverage-recording storage for fold scores.

Every scoring stage in this project reads a directory of AlphaFold3 output folds
and writes one CSV. Three things about that loop have burnt this project before,
and all three are handled here once instead of in each scorer:

1. **A partial run silently replacing a fuller one.** The screens are scored
   while they are still folding, so a scoring pass is normally incomplete. With a
   single hardcoded output path and no record of coverage, the resulting CSV is
   indistinguishable from a finished one. `write()` refuses to shrink an existing
   result unless the caller says so, and always writes a `.coverage.json`
   sidecar carrying the counts that a reader needs to know what they are holding.

2. **A results file older than its own scoring code.** When the scoring function
   changes, the rows already on disk were computed by code that no longer exists.
   `load_cache()` fingerprints the scoring source and drops every cached row that
   was produced by a different version, so a changed scorer re-scores rather than
   leaving a mixture of two definitions in one file.

3. **"Not measured" read as "measured absent".** A fold still being written has
   no summary JSON, which is exactly what a crashed fold looks like. Scoring it
   produces a row that reads as a failure. `classify()` separates a fold that has
   not finished yet from one that has stopped without finishing, using an age
   cutoff, and reports the two separately.

Incremental re-scoring keys on the output files' own size and mtime, so a fold
that is re-run is re-scored while the rest are reused.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger("score-store")

FP_COL = "_fingerprint"   # per-fold: identifies the bytes that were scored
CODE_COL = "_code"        # per-fold: identifies the code that scored them


def code_fingerprint(paths: list[Path]) -> str:
    """A short hash of the scoring source files.

    Any edit to the code that computes a row changes this, which invalidates
    every cached row. Cheap insurance against a file holding two definitions of
    the same column.
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:12]


def fold_fingerprint(d: Path) -> str | None:
    """Size and mtime of the files a score is computed from, or None if absent.

    Returning None is the signal that this fold has nothing to score yet. It is
    deliberately not an empty string: a caller testing truthiness cannot confuse
    the two.
    """
    parts = []
    for suffix in ("_summary_confidences.json", "_model.cif"):
        f = d / f"{d.name}{suffix}"
        if not f.exists():
            return None
        st = f.stat()
        parts.append(f"{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def fold_stage(d: Path) -> str:
    """How far this fold has got, read from what is in its directory.

    Directory mtime is NOT a usable liveness signal for this pipeline: the fold
    runners create every output directory and drop the AF3 input JSON in it when
    the shard starts, so a fold queued behind 4,000 others has a directory days
    old and looks abandoned. Reading the contents instead distinguishes the three
    states that matter without guessing:

      ready    the summary JSON and the model CIF are both present
      writing  a model CIF exists but the summary does not -- AF3 is mid-write
      queued   neither exists; only the input JSON the runner left
    """
    name = d.name
    if (d / f"{name}_summary_confidences.json").exists() and (d / f"{name}_model.cif").exists():
        return "ready"
    if any(d.glob("*_model.cif")) or any(d.glob("seed-*")):
        return "writing"
    return "queued"


def classify(out_dir: Path, expected: list[str], stage_live: bool | None = None) -> dict:
    """Split every expected fold into ready / writing / queued / absent.

    `expected` is the job names from the manifest, so a fold that was never
    started is counted too. Without this the denominator is whatever happens to
    be on disk, which grows as the screen runs and makes every fraction computed
    against it unreadable.

    `stage_live` says whether a runner for this stage is still going. It only
    renames the queued folds -- if nothing is running, a queued fold is one that
    will never arrive, and calling that "stalled" is the difference between
    "still coming" and "lost". When it is None the distinction is left open,
    because inventing one is how "not measured" becomes "measured absent".
    """
    buckets: dict[str, list[str]] = {"ready": [], "writing": [], "queued": [], "absent": []}
    fps: dict[str, str] = {}
    for name in expected:
        d = out_dir / name
        if not d.is_dir():
            buckets["absent"].append(name)
            continue
        stage = fold_stage(d)
        if stage == "ready":
            fp = fold_fingerprint(d)
            if fp is None:                 # raced with a write; treat as in flight
                buckets["writing"].append(name)
                continue
            fps[name] = fp
        buckets[stage].append(name)
    unexpected = sorted(d.name for d in out_dir.iterdir()
                        if d.is_dir() and d.name not in set(expected))
    return {"ready": fps, "writing": buckets["writing"], "queued": buckets["queued"],
            "absent": buckets["absent"], "unexpected": unexpected,
            "stage_live": stage_live,
            "stalled": [] if stage_live is not False else buckets["queued"]}


def load_cache(dest: Path, code_fp: str) -> pd.DataFrame:
    """Rows from a previous run that are still valid, as a frame indexed by pair.

    A row survives only if it carries both fingerprint columns and its code
    fingerprint matches the current one. An older file written before this module
    existed has neither column, so nothing is reused and the stage recomputes --
    the safe direction.
    """
    if not dest.exists():
        return pd.DataFrame()
    try:
        prev = pd.read_csv(dest)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s (%s); scoring from scratch", dest.name, exc)
        return pd.DataFrame()
    if FP_COL not in prev.columns or CODE_COL not in prev.columns:
        log.info("%s predates fingerprinting; scoring all folds", dest.name)
        return pd.DataFrame()
    keep = prev[prev[CODE_COL].eq(code_fp) & prev[FP_COL].notna()]
    if len(keep) < len(prev):
        log.info("dropped %d of %d cached rows: scored by different code",
                 len(prev) - len(keep), len(prev))
    return keep


def plan(ready: dict[str, str], cache: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Which folds must be scored now, and which cached rows carry over.

    A fold is re-scored when its bytes changed, which is what happens when it is
    re-folded. Reuse is keyed on the fingerprint, never on the name alone: a
    re-run under the same job name is the case a name-keyed cache gets wrong.
    """
    if cache.empty:
        return sorted(ready), cache
    valid = cache[cache.pair.map(ready.get).eq(cache[FP_COL])]
    todo = sorted(set(ready) - set(valid.pair))
    return todo, valid


def write(dest: Path, scored: pd.DataFrame, state: dict, code_fp: str,
          manifest_n: int, allow_shrink: bool = False, extra: dict | None = None) -> None:
    """Write the result atomically, with a coverage sidecar, refusing to shrink.

    The shrink guard is the important part. Scoring runs while folding runs, so
    the natural failure is a pass that sees fewer folds than the last one -- a
    crashed worker pool, a half-mounted directory, a manifest rebuilt smaller --
    quietly overwriting a more complete result with a thinner one that still
    looks complete. Refusing needs an explicit override, and the override is
    recorded in the sidecar.
    """
    cov_path = dest.with_suffix(".coverage.json")
    if dest.exists() and not allow_shrink:
        try:
            prev_n = len(pd.read_csv(dest, usecols=["pair"]))
        except Exception:  # noqa: BLE001
            prev_n = 0
        if len(scored) < prev_n:
            raise SystemExit(
                f"refusing to overwrite {dest.name}: it holds {prev_n} folds and this "
                f"run scored {len(scored)}. A scoring pass that sees fewer folds than "
                f"the last one is usually a broken run, not progress. Check "
                f"{cov_path.name} for what the previous pass covered, then pass "
                f"--allow-shrink if the shrink is genuinely intended.")

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    scored.to_csv(tmp, index=False)
    os.replace(tmp, dest)      # atomic: a reader never sees a half-written CSV

    coverage = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_fingerprint": code_fp,
        "manifest_folds": manifest_n,
        "scored": int(len(scored)),
        "ready_on_disk": len(state["ready"]),
        "writing_now": len(state["writing"]),
        "queued_not_started": len(state["queued"]),
        "stalled_no_runner": len(state["stalled"]),
        "stage_live": state.get("stage_live"),
        "never_started": len(state["absent"]),
        "unexpected_on_disk": len(state["unexpected"]),
        "complete": len(state["ready"]) == manifest_n,
        "allow_shrink": bool(allow_shrink),
        **(extra or {}),
    }
    cov_tmp = cov_path.with_suffix(".json.tmp")
    cov_tmp.write_text(json.dumps(coverage, indent=2) + "\n")
    os.replace(cov_tmp, cov_path)

    log.info("wrote %s (%d folds) and %s", dest.name, len(scored), cov_path.name)
    log.info("coverage: %d/%d of the manifest scored; %d being written, %d queued, "
             "%d never started%s", coverage["scored"], manifest_n,
             coverage["writing_now"], coverage["queued_not_started"],
             coverage["never_started"],
             "" if coverage["complete"] else "  <-- SCREEN IS INCOMPLETE")
    if state["stalled"]:
        log.warning("%d folds have no output and no runner is alive, so they are not "
                    "coming: %s", len(state["stalled"]), ", ".join(state["stalled"][:6]))
    if state["unexpected"]:
        log.warning("%d fold directories are not in the manifest: %s",
                    len(state["unexpected"]), ", ".join(state["unexpected"][:6]))


def require_complete(dest: Path, what: str = "this analysis") -> dict:
    """Read the sidecar and refuse to proceed on a partial screen.

    For a notebook or report that must not quote a number from a half-finished
    screen. Raises if the sidecar is missing, because an unlabelled CSV is
    exactly the artefact this module exists to stop being trusted.
    """
    cov_path = dest.with_suffix(".coverage.json")
    if not cov_path.exists():
        raise SystemExit(f"{dest.name} has no coverage sidecar, so there is no record "
                         f"of how much of the screen it covers; {what} cannot rely on it")
    cov = json.loads(cov_path.read_text())
    if not cov.get("complete"):
        raise SystemExit(
            f"{dest.name} covers {cov['scored']} of {cov['manifest_folds']} folds "
            f"({cov.get('queued_not_started', '?')} still queued); {what} needs the "
            f"full screen")
    return cov
