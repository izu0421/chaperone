#!/bin/bash
# Arm A: the same 2,500 pilot pairs as arm B, folded by AF3 on AlphaFast's OWN
# MMseqs2-GPU MSAs. B - A isolates the alignment at fixed engine.
#
# Differs from run_armB.sh in exactly one flag: --norun_data_pipeline is ABSENT,
# so AlphaFast builds the MSAs instead of reading omicMSAs out of the input JSON.
# Everything else -- pair set, templates, seeds, sharding, GPU hygiene -- is held
# identical, because the read-out is a paired difference and any other change
# would be attributed to the alignment.
#
# Same two hard-won rules as arm B:
#   XLA_PYTHON_CLIENT_PREALLOCATE=false, or JAX grabs ~75% of a shared card.
#   Progress is the check, never the exit code: a GPU failure can return 0 having
#   produced nothing (arm D wasted ten minutes that way, and a previous project
#   logged 480/480 NaN on an exit-0 run).
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
SHARD=$1
GPU=$2
NSHARD=${3:-3}
IN=$ROOT/work/armA_inputs
STAGE=$ROOT/work/armA_stage_$SHARD
OUT=$ROOT/results/armA_out
mkdir -p "$STAGE" "$OUT" "$ROOT/logs"

while true; do
  rm -f "$STAGE"/*.json
  n=$(/data/yzy21/micromamba/envs/structmod/bin/python -I - "$IN" "$OUT" "$STAGE" "$SHARD" "$NSHARD" <<'PY'
import json, shutil, sys
from pathlib import Path
inp, out, stage = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shard, nshard = int(sys.argv[4]), int(sys.argv[5])
todo = []
for f in inp.glob("*.json"):
    if not list((out / f.stem.lower()).glob("*_summary_confidences.json")) \
       and not list((out / f.stem).glob("*_summary_confidences.json")):
        d = json.loads(f.read_text())
        todo.append((sum(len(s["protein"]["sequence"]) for s in d["sequences"]), f))
todo.sort()                        # size-sorted, dealt round-robin: cost ~ bucket^2.79,
mine = [f for i, (_, f) in enumerate(todo) if i % nshard == shard]   # so alphabetical
for f in mine:                                                        # would idle 2 GPUs
    shutil.copy(f, stage / f.name)
print(len(mine))
PY
)
  echo "[armA shard $SHARD gpu $GPU] $(date +%H:%M:%S) $n pairs remaining"
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
        --model_dir=/data/models --db_dir=/data/public_databases \
        --mmseqs_db_dir=/data/public_databases/mmseqs \
    >> "$ROOT/logs/armA_shard${SHARD}.log" 2>&1
  rc=$?
  after=$(ls -d "$OUT"/*/ 2>/dev/null | wc -l)

  if [ "$after" -le "$before" ]; then
    echo "[armA shard $SHARD] NO PROGRESS (rc=$rc, $before -> $after). Stopping." \
      | tee -a "$ROOT/logs/armA_shard${SHARD}.log"
    exit 1
  fi
  echo "[armA shard $SHARD] rc=$rc, outputs $before -> $after"
done
echo "[armA shard $SHARD] complete $(date +%F\ %H:%M)"
