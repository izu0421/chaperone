"""What score_store must guarantee, exercised on synthetic fold directories.

These are the four failures the module exists to prevent. Each one has bitten
this project in some form, so each gets a test rather than a comment.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("score_store", ROOT / "scripts" / "score_store.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["score_store"] = S
_spec.loader.exec_module(S)


def make_fold(out: Path, name: str, stage: str = "ready", payload: str = "x") -> Path:
    """A fold directory at a chosen stage of completion."""
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}_data.json").write_text("{}")          # the runner's input copy
    if stage in ("writing", "ready"):
        (d / f"{name}_model.cif").write_text(f"# {payload}\n")
    if stage == "ready":
        (d / f"{name}_summary_confidences.json").write_text(json.dumps({"iptm": 0.5}))
    return d


def test_stage_read_from_contents_not_mtime(tmp_path):
    """A queued fold is not a stalled one, however old its directory is.

    The runners create every output directory and drop the input JSON in it when
    the shard starts, so a fold queued behind thousands of others has a
    directory days old. Judging liveness by mtime called 95 queued folds stalled.
    """
    out = tmp_path / "out"
    make_fold(out, "ready_one", "ready")
    make_fold(out, "writing_one", "writing")
    queued = make_fold(out, "queued_one", "queued")
    import os
    os.utime(queued, (0, 0))          # a directory from 1970: still only queued
    assert S.fold_stage(out / "ready_one") == "ready"
    assert S.fold_stage(out / "writing_one") == "writing"
    assert S.fold_stage(queued) == "queued"


def test_classify_counts_against_the_manifest(tmp_path):
    """The denominator is the manifest, including folds never started."""
    out = tmp_path / "out"
    make_fold(out, "a", "ready")
    make_fold(out, "b", "queued")
    make_fold(out, "rogue", "ready")           # on disk, not in the manifest
    st = S.classify(out, ["a", "b", "c"], stage_live=True)
    assert list(st["ready"]) == ["a"]
    assert st["queued"] == ["b"]
    assert st["absent"] == ["c"]                # never started, still counted
    assert st["unexpected"] == ["rogue"]
    assert st["stalled"] == []                  # a runner is alive, so b is coming


def test_queued_is_only_stalled_when_nothing_runs(tmp_path):
    out = tmp_path / "out"
    make_fold(out, "b", "queued")
    assert S.classify(out, ["b"], stage_live=True)["stalled"] == []
    assert S.classify(out, ["b"], stage_live=None)["stalled"] == []   # unknown, not asserted
    assert S.classify(out, ["b"], stage_live=False)["stalled"] == ["b"]


def test_changed_scoring_code_invalidates_cached_rows(tmp_path):
    """A results file must never hold two definitions of the same column."""
    dest = tmp_path / "scores.csv"
    pd.DataFrame({"pair": ["a"], "iptm": [0.5],
                  S.FP_COL: ["fp1"], S.CODE_COL: ["codeA"]}).to_csv(dest, index=False)
    assert len(S.load_cache(dest, "codeA")) == 1
    assert len(S.load_cache(dest, "codeB")) == 0      # scorer changed: recompute


def test_prefingerprint_file_is_not_reused(tmp_path):
    """An older file with no fingerprints is recomputed, not trusted."""
    dest = tmp_path / "scores.csv"
    pd.DataFrame({"pair": ["a"], "iptm": [0.5]}).to_csv(dest, index=False)
    assert S.load_cache(dest, "codeA").empty


def test_refold_under_same_name_is_rescored(tmp_path):
    """Reuse keys on the bytes, never on the job name alone."""
    out = tmp_path / "out"
    make_fold(out, "a", "ready", payload="first")
    fp_before = S.fold_fingerprint(out / "a")
    cache = pd.DataFrame({"pair": ["a"], S.FP_COL: [fp_before], S.CODE_COL: ["c"]})

    todo, reused = S.plan({"a": fp_before}, cache)
    assert todo == [] and len(reused) == 1            # unchanged: reuse

    make_fold(out, "a", "ready", payload="a much longer second fold")
    fp_after = S.fold_fingerprint(out / "a")
    assert fp_after != fp_before
    todo, reused = S.plan({"a": fp_after}, cache)
    assert todo == ["a"] and len(reused) == 0         # re-folded: re-score


def test_write_refuses_to_shrink(tmp_path):
    """A pass that sees fewer folds than the last one is usually a broken run."""
    dest = tmp_path / "scores.csv"
    st = {"ready": {"a": "f", "b": "f"}, "writing": [], "queued": [],
          "absent": [], "unexpected": [], "stalled": [], "stage_live": True}
    S.write(dest, pd.DataFrame({"pair": ["a", "b"]}), st, "c", 2)
    assert len(pd.read_csv(dest)) == 2

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        S.write(dest, pd.DataFrame({"pair": ["a"]}), st, "c", 2)
    assert len(pd.read_csv(dest)) == 2                # the fuller result survived

    S.write(dest, pd.DataFrame({"pair": ["a"]}), st, "c", 2, allow_shrink=True)
    assert len(pd.read_csv(dest)) == 1                # explicit override honoured
    assert json.loads(dest.with_suffix(".coverage.json").read_text())["allow_shrink"]


def test_coverage_sidecar_records_incompleteness(tmp_path):
    dest = tmp_path / "scores.csv"
    st = {"ready": {"a": "f"}, "writing": ["b"], "queued": ["c"], "absent": ["d"],
          "unexpected": [], "stalled": [], "stage_live": True}
    S.write(dest, pd.DataFrame({"pair": ["a"]}), st, "code1", 4)
    cov = json.loads(dest.with_suffix(".coverage.json").read_text())
    assert cov["complete"] is False
    assert (cov["scored"], cov["manifest_folds"]) == (1, 4)
    assert (cov["writing_now"], cov["queued_not_started"], cov["never_started"]) == (1, 1, 1)
    assert cov["code_fingerprint"] == "code1"


def test_require_complete_blocks_a_partial_screen(tmp_path):
    dest = tmp_path / "scores.csv"
    st = {"ready": {"a": "f"}, "writing": [], "queued": ["b"], "absent": [],
          "unexpected": [], "stalled": [], "stage_live": True}
    S.write(dest, pd.DataFrame({"pair": ["a"]}), st, "c", 2)
    with pytest.raises(SystemExit, match="needs the full screen"):
        S.require_complete(dest, "the figure")

    st2 = {**st, "ready": {"a": "f", "b": "f"}, "queued": []}
    S.write(dest, pd.DataFrame({"pair": ["a", "b"]}), st2, "c", 2)
    assert S.require_complete(dest, "the figure")["complete"] is True


def test_require_complete_refuses_an_unlabelled_file(tmp_path):
    """A CSV with no sidecar has no record of its coverage, so it is not trusted."""
    dest = tmp_path / "scores.csv"
    pd.DataFrame({"pair": ["a"]}).to_csv(dest, index=False)
    with pytest.raises(SystemExit, match="no coverage sidecar"):
        S.require_complete(dest)


def test_code_fingerprint_tracks_the_source(tmp_path):
    f = tmp_path / "s.py"
    f.write_text("a = 1")
    first = S.code_fingerprint([f])
    f.write_text("a = 2")
    assert S.code_fingerprint([f]) != first
    assert S.code_fingerprint([tmp_path / "absent.py"])      # missing file is not a crash
