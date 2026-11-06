#!/bin/bash
# Arm B: fold the pilot pairs with AF3 on the omicMSAs, sharded across GPUs.
#
# Resumable by construction. Each invocation stages only the inputs that have no
# *_summary_confidences.json yet, so a crash, an OOM caused by another user taking memory, or a
# deliberate stop costs at most the pair in flight. That matters here because the box is shared:
# five other accounts were running when this started.
#
# XLA_PYTHON_CLIENT_PREALLOCATE=false is required, not optional. JAX otherwise grabs ~75% of the
# card (66.6 GB observed for a 419-token pair) and AF3 will not start alongside other jobs.
# With preallocation off, the largest pilot pair (1,634 tokens, bucket 2048) ran in 44 GB.
#
# Cost is strongly super-linear in sequence length: fitting the two measured points gives
# t ~ bucket^2.79, so a 2048-bucket pair costs 618 s against 40 s at 768. Shards are therefore
# balanced by round-robin over the SIZE-SORTED list, not by file name -- splitting alphabetically
# would leave one GPU with most of the large pairs and idle the others for a day.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
SHARD=$1                      # 0, 1 or 2
GPU=$2
IN=$ROOT/work/armB_inputs
STAGE=$ROOT/work/armB_stage_$SHARD
OUT=$ROOT/results/armB_out
mkdir -p "$STAGE" "$OUT"

while true; do
  rm -f "$STAGE"/*.json
  n=$(/data/yzy21/micromamba/envs/structmod/bin/python -I - "$IN" "$OUT" "$STAGE" "$SHARD" <<'PY'
import json, shutil, sys
from pathlib import Path
inp, out, stage, shard = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
todo = []
for f in inp.glob("*.json"):
    if not list((out / f.stem).glob("*_summary_confidences.json")):
        d = json.loads(f.read_text())
        todo.append((sum(len(s["protein"]["sequence"]) for s in d["sequences"]), f))
todo.sort()                                  # size-sorted, then dealt round-robin
mine = [f for i, (_, f) in enumerate(todo) if i % 3 == shard]
for f in mine:
    shutil.copy(f, stage / f.name)
print(len(mine))
PY
)
  echo "[shard $SHARD gpu $GPU] $(date +%H:%M:%S) $n pairs remaining"
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
    >> "$ROOT/logs/armB_shard${SHARD}.log" 2>&1
  rc=$?
  after=$(ls -d "$OUT"/*/ 2>/dev/null | wc -l)

  # Exit code is not the check. A GPU failure can return 0 having produced nothing, which is how
  # the arm D build wasted ten minutes earlier today and how a previous project logged 480/480
  # NaN on an exit-0 run. Progress is the check.
  if [ "$after" -le "$before" ]; then
    echo "[shard $SHARD] NO PROGRESS (rc=$rc, $before -> $after). Stopping this shard." \
      | tee -a "$ROOT/logs/armB_shard${SHARD}.log"
    exit 1
  fi
  echo "[shard $SHARD] rc=$rc, outputs $before -> $after"
done
echo "[shard $SHARD] complete $(date +%F\ %H:%M)"
