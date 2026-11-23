#!/bin/bash
# Arm D harness check, level 1: run RF2-PPI on the repo's OWN pre-paired example MSAs and
# compare with the expected output they ship. If this does not reproduce, nothing measured with
# this harness afterwards is interpretable.
#
# The README warns the model is non-deterministic (sd > 0.1 for ~5% of cases), so the run is
# repeated and the spread reported -- a single run matching or not matching proves little.
set -eu
ROOT=/data/yzy21/yy/space_agent/structural_modelling
SIF=$ROOT/ext/SE3nv-20230612.sif
REPO=$ROOT/ext/RoseTTAFold2-PPI
OUT=$ROOT/results/armD_example
EXPECTED=$ROOT/ext/RoseTTAFold2-PPI/examples/expected_output/protein_pairs_input.log
mkdir -p "$OUT"

# wait for the container, then assert it is complete rather than trusting the file's existence
while [ ! -f "$SIF" ] || [ "$(stat -c %s "$SIF")" -lt 14361309184 ]; do sleep 30; done
echo "container complete: $(stat -c %s "$SIF") bytes at $(date +%H:%M)"

cd "$REPO/examples"
for rep in 1 2 3; do
  cp protein_pairs_input "$OUT/rep${rep}_input"
  singularity exec --nv \
    --bind "$REPO/examples":/work/users \
    --bind "$REPO":/home/RoseTTAFold2-PPI \
    "$SIF" /bin/bash -c \
    "cd /work/users && python /home/RoseTTAFold2-PPI/src/predict_list_PPI.py \
       -list_fn protein_pairs_input \
       -model_file /home/RoseTTAFold2-PPI/src/models/RF2-PPI.pt" 2>&1 | tail -3
  cp protein_pairs_input.log "$OUT/rep${rep}.log"
  echo "=== replicate $rep ==="; cat "$OUT/rep${rep}.log"
done

echo "=== EXPECTED (shipped by the authors) ==="; cat "$EXPECTED"
