#!/bin/bash
# Arm D: RF2-PPI on the omicMSA paired alignments -- the published setting, and the anchor the
# other arms are read against. Paths inside the container must be relative to the bind mount.
set -eu
ROOT=/data/yzy21/yy/space_agent/structural_modelling
cd "$ROOT/work/armD_msas"
sed 's|^work/armD_msas/||' "$ROOT/bench/pilot_pairs.txt_input" > armD_input
singularity exec --nv \
  --bind "$ROOT/work/armD_msas":/work/users \
  --bind "$ROOT/ext/RoseTTAFold2-PPI":/home/RoseTTAFold2-PPI \
  "$ROOT/ext/SE3nv-20230612.sif" \
  /bin/bash -c "cd /work/users && python /home/RoseTTAFold2-PPI/src/predict_list_PPI.py \
     -list_fn armD_input -model_file /home/RoseTTAFold2-PPI/src/models/RF2-PPI.pt"
cp armD_input.log "$ROOT/results/armD_out/" 2>/dev/null || true
echo "arm D done $(date +%F\ %H:%M); scored $(grep -c . armD_input.log) lines"
