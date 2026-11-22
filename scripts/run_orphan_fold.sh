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
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
SHARD=$1
GPU=$2
NSHARD=${3:-2}
IN=$ROOT/work/orphan_inputs_omic
STAGE=$ROOT/work/orphan_stage_$SHARD
OUT=$ROOT/results/orphan_out
mkdir -p "$STAGE" "$OUT" "$ROOT/logs"

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
# stage a bounded slice: the inputs are ~7 MB each, so copying all 2,500 at once
# would write 18 GB before a single fold starts.
for f in mine[:400]:
    shutil.copy(f, stage / f.name)
print(len(mine))
PY
)
  echo "[orphan shard $SHARD gpu $GPU] $(date +%H:%M:%S) $n pairs remaining"
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
    >> "$ROOT/logs/orphan_shard${SHARD}.log" 2>&1
  rc=$?
  after=$(ls -d "$OUT"/*/ 2>/dev/null | wc -l)
  if [ "$after" -le "$before" ]; then
    echo "[orphan shard $SHARD] NO PROGRESS (rc=$rc, $before -> $after). Stopping." \
      | tee -a "$ROOT/logs/orphan_shard${SHARD}.log"
    exit 1
  fi
  echo "[orphan shard $SHARD] rc=$rc, outputs $before -> $after"
done
echo "[orphan shard $SHARD] complete $(date +%F\ %H:%M)"
