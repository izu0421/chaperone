#!/bin/bash
# Fold the subcomplexes whole, and the secreted partners as their mature chain.
# Five seeds per job.
#
# Usage: run_subcomplex_fold.sh <shard> <gpu> [nshard]
#
# Shards by fold order (size-sorted). The inputs carry their own omicMSA, so
# --norun_data_pipeline applies here too. Progress is the check, never the
# exit code: a CUDA OOM can leave a clean exit with nothing written.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
SHARD=$1
GPU=$2
NSHARD=${3:-2}
IN=$ROOT/work/ectodomain_inputs
STAGE=$ROOT/work/ecd_stage_$SHARD
OUT=$ROOT/results/ectodomain_out
mkdir -p "$STAGE" "$OUT" "$ROOT/logs"
# A marker a watcher can poll. Pattern-matching the process list cannot be
# trusted here: a shell whose command line merely CONTAINS this script's name
# matches it, which already fired two watchers early.
mkdir -p "$ROOT/run"
RUNMARK="$ROOT/run/ecd_shard$SHARD.running"
echo $$ > "$RUNMARK"
trap 'rm -f "$RUNMARK"' EXIT

while true; do
  find "$STAGE" -maxdepth 1 -name '*.json' -delete 2>/dev/null
  n=$(/data/yzy21/micromamba/envs/structmod/bin/python -I - "$IN" "$OUT" "$STAGE" "$SHARD" "$NSHARD" "$ROOT" <<'PY'
import csv, shutil, sys
from pathlib import Path
inp, out, stage = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shard, nshard, root = int(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6])
order = []
with open(root / "work" / "ectodomain_manifest.csv") as fh:
    for row in csv.DictReader(fh):
        order.append(row["job"])
todo = []
for job in order:
    f = inp / f"{job}.json"
    if not f.exists():
        continue
    if list((out / job).glob("*_summary_confidences.json")) or \
       list((out / job.lower()).glob("*_summary_confidences.json")):
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
for f in mine[:300]:
    shutil.copy(f, stage / f.name)
print(len(mine))
PY
)
  echo "[ecd shard $SHARD gpu $GPU] $(date +%H:%M:%S) $n pairs remaining"
  [ "$n" -eq 0 ] && break

  # Progress is COMPLETED FOLDS, not directories. AlphaFold3 creates <job>/ and
  # writes <job>_data.json the moment a batch starts, so a directory count is
  # wrong in both directions: it saturates at the job total and then reads as
  # "no progress" forever even while folds succeed, and on a fresh stage a batch
  # that runs out of memory on every single job still creates every directory,
  # so a total GPU failure reads as healthy progress. The orphan screen hit the
  # first of those: 5,000 directories against 4,909 finished folds.
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
    >> "$ROOT/logs/ecd_shard${SHARD}.log" 2>&1
  rc=$?
  after=$(find "$OUT" -maxdepth 2 -name "*_summary_confidences.json" 2>/dev/null | wc -l)
  if [ "$after" -le "$before" ]; then
    echo "[ecd shard $SHARD] NO PROGRESS (rc=$rc, $before -> $after). Stopping." \
      | tee -a "$ROOT/logs/ecd_shard${SHARD}.log"
    exit 1
  fi
  echo "[ecd shard $SHARD] rc=$rc, outputs $before -> $after"
done
echo "[ecd shard $SHARD] complete $(date +%F\ %H:%M)"
