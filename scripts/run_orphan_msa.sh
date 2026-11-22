#!/bin/bash
# Orphan folds, stage 1: MSAs for the 5,000 pairs (1,093 unique chains).
#
# CPU path by default, for the reason stage 1 of arm A found: MMseqs2-GPU dies in
# the ungapped prefilter (libmarv dbbatching.cuh:69) on the 3rd search of a
# pipelined run, whichever database sits in that slot, even with 82 GB free. CPU
# is ~10x slower per search but it finishes, and the MSA CONTENT is identical --
# CPU and GPU MMseqs2 run the same search. Arm A's 4,181 chains took 3.07 h this
# way, so 1,093 chains should be well under an hour.
#
# --batch_size IS NOT OPTIONAL: without it AF3 searches all four databases PER
# PAIR, which here would be 20,000 searches instead of 4.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
IN=$ROOT/work/orphan_inputs
OUT=$ROOT/results/orphan_msa
mkdir -p "$OUT" "$ROOT/logs"

before=$(ls "$OUT"/*/*_data.json 2>/dev/null | wc -l)
env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  singularity exec --nv \
    --bind /data/yzy21/yy/af/alphafast_db:/data/public_databases \
    --bind /data/yzy21/yy/af:/data/models \
    --bind "$IN":/data/af_input \
    --bind "$OUT":/data/af_output \
    /data/yzy21/yy/af/alphafast/alphafast.sif \
    python /app/alphafold/run_alphafold.py \
      --input_dir=/data/af_input --output_dir=/data/af_output \
      --model_dir=/data/models --db_dir=/data/public_databases \
      --mmseqs_db_dir=/data/public_databases/mmseqs \
      --mmseqs_n_threads=32 --batch_size=8192 \
      --norun_inference --nouse_mmseqs_gpu \
  >> "$ROOT/logs/orphan_msa.log" 2>&1
rc=$?
after=$(ls "$OUT"/*/*_data.json 2>/dev/null | wc -l)
echo "$(date +%T) rc=$rc, data.json $before -> $after (target 5000)"
[ "$after" -le "$before" ] && { echo "NO PROGRESS -- stopping"; exit 1; }
echo "orphan MSA stage done: $after of 5000"
