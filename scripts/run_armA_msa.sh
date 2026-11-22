#!/bin/bash
# Arm A, stage 1: build the MSAs. Folding is stage 2 (run_armA_fold.sh).
#
# WHY TWO STAGES. Arm B separated them too, but here it is forced. MMseqs2-GPU
# needs a large amount of free GPU memory and fails with a CUDA "illegal memory
# access" at libmarv dbbatching.cuh:69 when it does not have it -- an OOM wearing
# a misleading error message. Measured on this box:
#
#   ~60 GB free  all four databases search on GPU (small_bfd in 10s)
#    29 GB free  fails
#   8.6 GB free  fails
#
# None of it is database-specific, which cost me two wrong diagnoses. A rebuilt
# small_bfd from the official AF3 FASTA failed identically (so: not corruption),
# and uniprot searches fine standalone but died as the 4th search in the pipeline
# (so: not that database either). Thread count correlates only because more
# threads means more concurrent GPU memory. This box is shared with ~8 other
# jobs and swings between 8 GB and 60 GB free, so the MSA stage cannot simply
# assume a GPU.
#
# So: wait for a card with enough free memory and use it; fall back to CPU after
# a bounded wait. The MSA CONTENT is identical either way -- CPU and GPU MMseqs2
# run the same search -- so arm A is scientifically the same experiment on either
# path. Only the wall-clock speedup differs, and that is a separate measurement.
#
# Folding, which is where arm B spent ~31 GPU-hours, is stage 2 and always GPU.
#
# --batch_size IS NOT OPTIONAL. Without it AF3 "processes each fold input
# sequentially": 4 database searches PER PAIR, so 2,500 pairs would run 10,000
# searches instead of 4. With it, protein chains are deduplicated across all
# fold inputs first -- 5,000 chain entries collapse to 4,181 unique sequences --
# and each database is searched once for the whole arm. 8192 is set above that
# count so the batch is not split.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
NEED_MIB=${NEED_MIB:-61440}        # 60 GB; the lowest observed success
WAIT_MIN=${WAIT_MIN:-45}           # how long to hold out for a GPU
IN=$ROOT/work/armA_inputs
OUT=$ROOT/results/armA_msa
mkdir -p "$OUT" "$ROOT/logs"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | awk -F'[, ]+' -v need="$NEED_MIB" '{f=$4-$2; if(f>=need){print f, $1}}' \
    | sort -rn | head -1 | cut -d' ' -f2
}

GPU=""; deadline=$(( $(date +%s) + WAIT_MIN*60 ))
while [ -z "$GPU" ] && [ "$(date +%s)" -lt "$deadline" ]; do
  GPU=$(pick_gpu)
  [ -z "$GPU" ] && { echo "$(date +%T) no card with ${NEED_MIB} MiB free; waiting"; sleep 120; }
done

if [ -n "$GPU" ]; then
  echo "$(date +%T) running MSA stage on GPU $GPU"
  DEV="$GPU"; GPUFLAG=""
else
  echo "$(date +%T) no GPU reached ${NEED_MIB} MiB free in ${WAIT_MIN} min; CPU path"
  echo "   (reliable, ~10x slower per search: 573s vs 58s measured; content identical)"
  DEV=0; GPUFLAG="--nouse_mmseqs_gpu"
fi

before=$(ls "$OUT"/*/*_data.json 2>/dev/null | wc -l)
env CUDA_VISIBLE_DEVICES="$DEV" XLA_PYTHON_CLIENT_PREALLOCATE=false \
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
      --mmseqs_n_threads=32 \
      --batch_size=8192 \
      --norun_inference $GPUFLAG \
  >> "$ROOT/logs/armA_msa.log" 2>&1
rc=$?
after=$(ls "$OUT"/*/*_data.json 2>/dev/null | wc -l)

# Progress is the check, not the exit code.
echo "$(date +%T) rc=$rc, data.json files $before -> $after (target 2500)"
[ "$after" -le "$before" ] && { echo "NO PROGRESS -- stopping"; exit 1; }
echo "stage 1 done: $after of 2500"
