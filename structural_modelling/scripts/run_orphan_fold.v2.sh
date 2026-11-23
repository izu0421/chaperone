#!/bin/bash
# Orphan folds: 5,000 receptor-partner pairs on the omicMSAs.
#
# SMALLEST FIRST, deliberately. AF3 cost goes as bucket^2.79, and for this set
# the 87 largest pairs (1.7%) carry 27% of the total ~512 GPU-h while the 85%
# smallest carry 28%. Size order means most of the set is answered in the first
# day or so and the expensive tail can be abandoned without losing coverage --
# better than picking an arbitrary size cap up front. The two-way-null rank is
# kept in work/orphan_manifest.csv, not in the queue order.
#
# --norun_data_pipeline: the MSAs are already embedded by
# 12_build_orphan_inputs_omic.py. 85 of the 5,000 have no partner omicMSA and
# fold single-sequence; they are flagged msa_source=none_single_sequence in the
# manifest so they can be read apart rather than quietly diluting the set.
#
# Same two rules as every other arm here:
#   XLA_PYTHON_CLIENT_PREALLOCATE=false, or JAX takes ~75% of a shared card.
#   Progress is the check, never the exit code.
# v2, 2026-11-23. Written as a separate file because bash reads a script
# incrementally as it runs, so editing the copy a live shard is executing moves
# the byte offset under it. Two changes:
#
# 1. Progress is COMPLETED FOLDS, not directories. AF3 creates <job>/ and writes
#    <job>_data.json when a batch starts, so a directory count saturates at the
#    job total and then reads as "no progress" forever even while folds succeed
#    -- and on a fresh stage a batch that OOMs on every job still creates every
#    directory, so a total GPU failure reads as healthy progress. v1 hit the
#    first of those at 5,000 directories against 4,909 finished folds.
#
# 2. A job that keeps failing is retired instead of blocking the queue. 85 jobs
#    were written without an embedded MSA, which --norun_data_pipeline cannot
#    supply, so AF3 rejected each one every pass. They never left the todo list,
#    so `n` never reached 0 and the clean exit was unreachable: the shard could
#    only ever leave through the failure path. Those 85 are repaired now
#    (13_repair_unfoldable_inputs.py), but the loop must not depend on that.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
SHARD=$1
GPU=$2
NSHARD=${3:-2}
IN=$ROOT/work/orphan_inputs_omic
STAGE=$ROOT/work/orphan_stage_$SHARD
OUT=$ROOT/results/orphan_out
mkdir -p "$STAGE" "$OUT" "$ROOT/logs"
# A marker a watcher can poll. Pattern-matching the process list cannot be
# trusted here: a shell whose command line merely CONTAINS this script's name
# matches it, which already fired two watchers early.
mkdir -p "$ROOT/run"
RUNMARK="$ROOT/run/orphan_shard$SHARD.running"
echo $$ > "$RUNMARK"
trap 'rm -f "$RUNMARK"' EXIT

while true; do
  find "$STAGE" -maxdepth 1 -name '*.json' -delete 2>/dev/null
  n=$(/data/yzy21/micromamba/envs/structmod/bin/python -I - "$IN" "$OUT" "$STAGE" "$SHARD" "$NSHARD" "$ROOT" <<'PY'
import shutil, sys, csv
from pathlib import Path
inp, out, stage = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shard, nshard, root = int(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6])
# fold order comes from the manifest (already size-sorted), so the queue never
# re-parses the 37 GB of embedded MSAs just to recover a sequence length.
order = []
with open(root / "work" / "orphan_manifest.csv") as fh:
    for row in csv.DictReader(fh):
        order.append(row["job"])
retired_f = root / "run" / "orphan_retired.txt"
retired = set(retired_f.read_text().split()) if retired_f.exists() else set()
todo = []
for job in order:
    f = inp / f"{job}.json"
    if not f.exists():
        continue
    if list((out / job).glob("*_summary_confidences.json")) or \
       list((out / job.lower()).glob("*_summary_confidences.json")):
        continue
    if job in retired:
        continue
    todo.append(f)
# Shard by a stable hash of the job name, NOT by position in `todo`.
# `todo` shrinks as folds complete, so a positional modulo reassigns every
# remaining job each time the list changes: a job in shard 0's set at one
# iteration lands in shard 1's at the next. Measured on this screen, 1,403 of
# 5,000 jobs were picked up by both shards, which wastes the GPU and can put two
# AF3 processes in one output directory. Adding candidates reshuffles it the same
# way. A hash of the name is invariant to completion, to additions, and to the
# manifest's order. sha1 because Python's own hash() is salted per process, so
# two shards would not even agree with each other.
import hashlib
def _shard_of(job: str, n: int) -> int:
    return int(hashlib.sha1(job.encode()).hexdigest(), 16) % n
mine = [f for f in todo if _shard_of(f.stem, nshard) == shard]
# stage a bounded slice: the inputs are ~7 MB each, so copying all 2,500 at once
# would write 18 GB before a single fold starts.
for f in mine[:400]:
    shutil.copy(f, stage / f.name)
print(len(mine))
PY
)
  echo "[orphan shard $SHARD gpu $GPU] $(date +%H:%M:%S) $n pairs remaining"
  [ "$n" -eq 0 ] && break

  # Record what AF3 rejected, and retire a job that has failed the same way three
  # times. Without this a permanently unfoldable job is retried forever, which
  # both wastes a featurisation pass per iteration and makes the shard unable to
  # finish. The retired list is a file, so the scorer and the status command can
  # report those jobs as explicitly not foldable instead of merely absent.
  RETIRED=$ROOT/run/orphan_retired.txt
  touch "$RETIRED"
  grep -oE 'Fold job [A-Za-z0-9_.-]+ failed' "$ROOT/logs/orphan_shard${SHARD}.log" 2>/dev/null \
    | awk '{print $3}' | sort | uniq -c \
    | awk '$1 >= 3 {print $2}' | sort -u >> "$RETIRED".new 2>/dev/null || true
  if [ -s "$RETIRED".new ]; then
    sort -u "$RETIRED" "$RETIRED".new > "$RETIRED".tmp && mv "$RETIRED".tmp "$RETIRED"
    rm -f "$RETIRED".new
    echo "[orphan shard $SHARD] $(wc -l < "$RETIRED") jobs retired after repeated failures"
  fi

  before=$(find "$OUT" -maxdepth 2 -name "*_summary_confidences.json" 2>/dev/null | wc -l)
  env CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    singularity exec --nv \
      --bind /data/yzy21/yy/af/alphafast_db:/data/public_databases \
      --bind /data/yzy21/yy/af:/data/models \
      --bind "$STAGE":/data/af_input \
      --bind "$OUT":/data/af_output \
      /data/yzy21/yy/af/alphafast/alphafast.sif \
      python /app/alphafold/run_alphafold.py --input_dir=/data/af_input --output_dir=/data/af_output \
        --model_dir=/data/models --db_dir=/data/public_databases --norun_data_pipeline \
    >> "$ROOT/logs/orphan_shard${SHARD}.log" 2>&1
  rc=$?
  after=$(find "$OUT" -maxdepth 2 -name "*_summary_confidences.json" 2>/dev/null | wc -l)
  if [ "$after" -le "$before" ]; then
    echo "[orphan shard $SHARD] NO PROGRESS (rc=$rc, $before -> $after). Stopping." \
      | tee -a "$ROOT/logs/orphan_shard${SHARD}.log"
    exit 1
  fi
  echo "[orphan shard $SHARD] rc=$rc, outputs $before -> $after"
done
echo "[orphan shard $SHARD] complete $(date +%F\ %H:%M)"
