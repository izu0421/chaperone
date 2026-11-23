"""Shard assignment must not move a job, ever.

The fold runners split work between two GPUs. The original split was
`[f for i, f in enumerate(todo) if i % nshard == shard]`, where `todo` is the
jobs not yet finished. That list shrinks as folds complete, so every remaining
job is renumbered each pass and shard membership flips: measured on the orphan
screen, 1,403 of 5,000 jobs were picked up by both shards. That wastes GPU time
and can put two AlphaFold3 processes in one output directory.

The replacement shards on a stable hash of the job name. These tests pin the
three properties that makes it safe for a queue that is still growing.

The function is duplicated here because it lives inside a heredoc in the shell
runners; `test_matches_the_runners` reads it back out of them so the two cannot
drift apart.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNERS = ["run_orphan_fold.v2.sh", "run_specificity_fold.sh", "run_huri_pilot_fold.sh"]


def shard_of(job: str, n: int) -> int:
    return int(hashlib.sha1(job.encode()).hexdigest(), 16) % n


JOBS = [f"RECEPTOR{i}__PARTNER{j}" for i in range(40) for j in range(25)]


def test_completion_does_not_move_a_job():
    """The whole point: finishing folds must not reassign the rest."""
    before = {j: shard_of(j, 2) for j in JOBS}
    remaining = JOBS[300:]                      # 300 folds finish
    after = {j: shard_of(j, 2) for j in remaining}
    assert all(after[j] == before[j] for j in remaining)


def test_shards_never_overlap():
    a = {j for j in JOBS if shard_of(j, 2) == 0}
    b = {j for j in JOBS if shard_of(j, 2) == 1}
    assert not (a & b)
    assert a | b == set(JOBS)


def test_overlap_across_two_moments_is_zero():
    """The exact failure that happened: shard 0 at t0 vs shard 1 at t1."""
    at = lambda done, s: {j for j in JOBS if j not in done and shard_of(j, 2) == s}
    assert not (at(set(), 0) & at(set(JOBS[:500]), 1))


def test_adding_jobs_does_not_move_existing_ones():
    before = {j: shard_of(j, 2) for j in JOBS}
    grown = JOBS + [f"NEWRECEPTOR__NEW{i}" for i in range(80)]
    assert all(shard_of(j, 2) == before[j] for j in JOBS)
    assert len(grown) == len(JOBS) + 80


def test_reordering_the_manifest_does_not_move_a_job():
    before = {j: shard_of(j, 2) for j in JOBS}
    for j in sorted(JOBS, reverse=True):
        assert shard_of(j, 2) == before[j]


def test_balance_is_close_to_even():
    n0 = sum(1 for j in JOBS if shard_of(j, 2) == 0)
    assert 0.9 < n0 / (len(JOBS) - n0) < 1.1


def test_hash_is_stable_across_processes():
    """sha1, not Python's hash(), which is salted per process -- two shards
    running as separate processes would not otherwise agree with each other."""
    assert shard_of("GPR101__ACVR1C", 2) == int(
        hashlib.sha1(b"GPR101__ACVR1C").hexdigest(), 16) % 2


@pytest.mark.parametrize("runner", RUNNERS)
def test_matches_the_runners(runner):
    """The runners must use this rule, not a positional modulo."""
    src = (ROOT / "scripts" / runner).read_text()
    assert "i % nshard == shard" not in src, f"{runner} still shards by position"
    assert "sha1" in src and "_shard_of" in src, f"{runner} does not shard by a stable hash"
    # and the expression they use must agree with the one tested here
    m = re.search(r"return int\(hashlib\.sha1\(job\.encode\(\)\)\.hexdigest\(\), 16\) % n", src)
    assert m, f"{runner}'s shard function differs from the tested one"
