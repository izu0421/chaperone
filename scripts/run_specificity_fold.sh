#!/bin/bash
# Fold the small-molecule specificity matrix: every panel ligand against every
# panel receptor. Same structure as run_orphan_fold.sh, which is deliberate --
# the benchmark has to be measured by the code path the screen uses.
#
# Usage: run_specificity_fold.sh <shard> <gpu> [nshard]
#
# Shards by fold order (size-sorted). The inputs carry their own omicMSA, so
# --norun_data_pipeline applies here too. Progress is the check, never the
# exit code: a CUDA OOM can leave a clean exit with nothing written.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
SHARD=$1
GPU=$2
NSHARD=${3:-2}
IN=$ROOT/work/specificity_inputs
STAGE=$ROOT/work/specificity_stage_$SHARD
OUT=$ROOT/results/specificity_out
mkdir -p "$STAGE" "$OUT" "$ROOT/logs"

while true; do
  find "$STAGE" -maxdepth 1 -name '*.json' -delete 2>/dev/null
  n=$(/data/yzy21/micromamba/envs/structmod/bin/python -I - "$IN" "$OUT" "$STAGE" "$SHARD" "$NSHARD" "$ROOT" <<'PY'
import csv, shutil, sys
from pathlib import Path
inp, out, stage = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shard, nshard, root = int(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6])
order = []
with open(root / "work" / "specificity_manifest.csv") as fh:
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
mine = [f for i, f in enumerate(todo) if i % nshard == shard]
for f in mine[:300]:
    shutil.copy(f, stage / f.name)
print(len(mine))
PY
)
  echo "[spec shard $SHARD gpu $GPU] $(date +%H:%M:%S) $n pairs remaining"
  [ "$n" -eq 0 ] && break

  before=$(ls -d "$OUT"/*/ 2>/dev/null | wc -l)
  env CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    singularity exec --nv \
      --bind /data/yzy21/yy/af/alphafast_db:/data/public_databases \
      --bind /data/yzy21/yy/af:/data/models \
      --bind "$STAGE":/data/af_input \
      --bind "$OUT":/data/af_output \
      /data/yzy21/yy/af/alphafast/alphafast.sif \
      python /app/alphafold/run_alphafold.py --input_dir=/data/af_input --output_dir=/data/af_output \
        --model_dir=/data/models --db_dir=/data/public_databases --norun_data_pipeline \
    >> "$ROOT/logs/spec_shard${SHARD}.log" 2>&1
  rc=$?
  after=$(ls -d "$OUT"/*/ 2>/dev/null | wc -l)
  if [ "$after" -le "$before" ]; then
    echo "[spec shard $SHARD] NO PROGRESS (rc=$rc, $before -> $after). Stopping." \
      | tee -a "$ROOT/logs/spec_shard${SHARD}.log"
    exit 1
  fi
  echo "[spec shard $SHARD] rc=$rc, outputs $before -> $after"
done
echo "[spec shard $SHARD] complete $(date +%F\ %H:%M)"
