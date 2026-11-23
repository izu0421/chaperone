#!/bin/bash
# Arm A, stage 2: fold the 2,500 pilot pairs on the MSAs stage 1 built.
#
# Stage 1 (run_armA_msa.sh) wrote results/armA_msa/<pair>/<pair>_data.json with
# AlphaFast's own MMseqs2 alignments embedded. This stage folds those with
# --norun_data_pipeline, exactly as arm B did on the omicMSAs, so the ONLY thing
# differing between arms B and A is the alignment.
#
# Arm C (RF2-PPI on AlphaFast MSAs) is dropped on the user's call 2026-11-17.
#
# Worth recording before the numbers land: stage 1's MSAs are 6.8x DEEPER than
# arm B's omicMSAs (median 4,924 vs 720 sequences, deeper on 87% of chains;
# arm B's distribution is smooth with no cap, so this is not our converter
# truncating). design.md framed B-A as "does AlphaFast's FAST MSA preserve the
# discrimination that omicMSA DEPTH buys" -- but AlphaFast is the deeper arm, so
# B-A measures curated-shallow against deeper-raw-search. Read the result that
# way, not the way the design document words it.
#
# Same two rules as arm B, both earned:
#   XLA_PYTHON_CLIENT_PREALLOCATE=false, or JAX takes ~75% of a shared card.
#   Progress is the check, never the exit code -- a GPU failure can return 0
#   having produced nothing.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
SHARD=$1
GPU=$2
NSHARD=${3:-3}
MSA=$ROOT/results/armA_msa
STAGE=$ROOT/work/armA_fold_stage_$SHARD
OUT=$ROOT/results/armA_out
mkdir -p "$STAGE" "$OUT" "$ROOT/logs"

while true; do
  find "$STAGE" -maxdepth 1 -name '*.json' -delete 2>/dev/null
  n=$(/data/yzy21/micromamba/envs/structmod/bin/python -I - "$MSA" "$OUT" "$STAGE" "$SHARD" "$NSHARD" <<'PY'
import json, shutil, sys
from pathlib import Path
msa, out, stage = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shard, nshard = int(sys.argv[4]), int(sys.argv[5])
todo = []
for f in sorted(msa.glob("*/*_data.json")):
    name = f.parent.name
    done = list((out / name).glob("*_summary_confidences.json")) + \
           list((out / name.lower()).glob("*_summary_confidences.json"))
    if done:
        continue
    # size-sort: AF3 cost is ~bucket^2.79, so dealing round-robin over a
    # size-sorted list keeps the shards balanced. Alphabetical would hand one
    # GPU most of the large pairs and idle the rest for a day.
    #
    # Read the length from work/armA_inputs (2.6 MB in total), NOT from the
    # data.json here: those carry the embedded MSAs and come to 44 GB, so
    # parsing them for a sequence length cost minutes on every cycle of the
    # loop. Same sequences either way -- stage 1 only added alignments.
    src = msa.parent.parent / "work" / "armA_inputs" / f"{name}.json"
    d = json.loads(src.read_text()) if src.exists() else json.loads(f.read_text())
    todo.append((sum(len(s["protein"]["sequence"]) for s in d["sequences"]), f))
todo.sort()
mine = [f for i, (_, f) in enumerate(todo) if i % nshard == shard]
for f in mine:
    shutil.copy(f, stage / f.name)
print(len(mine))
PY
)
  echo "[armA-fold shard $SHARD gpu $GPU] $(date +%H:%M:%S) $n pairs remaining"
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
    >> "$ROOT/logs/armA_fold_shard${SHARD}.log" 2>&1
  rc=$?
  after=$(ls -d "$OUT"/*/ 2>/dev/null | wc -l)

  if [ "$after" -le "$before" ]; then
    echo "[armA-fold shard $SHARD] NO PROGRESS (rc=$rc, $before -> $after). Stopping." \
      | tee -a "$ROOT/logs/armA_fold_shard${SHARD}.log"
    exit 1
  fi
  echo "[armA-fold shard $SHARD] rc=$rc, outputs $before -> $after"
done
echo "[armA-fold shard $SHARD] complete $(date +%F\ %H:%M)"
