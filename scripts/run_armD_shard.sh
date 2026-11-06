#!/bin/bash
# Arm D: RF2-PPI on the omicMSA paired alignments -- the published setting, and the anchor the
# other arms are read against.
#
# The first attempt died at pair 8 of 2,500 with a CUDA OOM, because arm B had the cards. It had
# no resume, so a restart would have thrown away the seven scored pairs and re-run them. This
# version is resumable and sharded:
#
#   * results accumulate in ONE master TSV, appended to, never overwritten. A pair already in it
#     is not re-run. predict_list_PPI.py writes its own log per invocation, so each cycle's log is
#     appended to the master and the per-invocation file is discarded.
#   * the todo list is rebuilt from the master every cycle, so an OOM costs the pair in flight.
#   * progress is the completion check, NOT the exit code. RF2-PPI returns 0 on a GPU it could
#     not allocate on, which is how the first run looked like a clean finish in the shell while
#     the traceback sat in the log.
#   * three shards deal from the SAME todo list by index, so a shard that dies does not strand
#     its share -- the others pick the pairs up on their next cycle.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
SHARD=$1
GPU=$2
W=$ROOT/work/armD_msas
MASTER=$ROOT/results/armD_out/armD_scores.tsv
mkdir -p "$ROOT/results/armD_out"
[ -s "$MASTER" ] || printf 'Input_MSA\tInteraction_probability\tCompute_time\n' > "$MASTER"

# seed the master with the seven pairs the crashed run did score
if [ -s "$ROOT/results/armD_out/armD_partial_20261105.log" ]; then
  tail -n +2 "$ROOT/results/armD_out/armD_partial_20261105.log" \
    | sed 's|^protein_paired_msas//||' >> "$MASTER"
  sort -u -o "$MASTER" "$MASTER"
  mv "$ROOT/results/armD_out/armD_partial_20261105.log" \
     "$ROOT/results/armD_out/.seeded_partial" 2>/dev/null || true
fi

cd "$W"
while true; do
  LIST=armD_todo_$SHARD
  cut -f1 "$MASTER" | tail -n +1 | sed 's|^protein_paired_msas//||' | sort -u > /tmp/armD_done_$SHARD
  awk -v s="$SHARD" 'NR==FNR{d[$0]=1;next}
       {k=$1; sub(/^.*\//,"",k); if(!(k in d)){ if(n%3==s) print $0; n++ }}' \
      /tmp/armD_done_$SHARD "$ROOT/bench/pilot_pairs.txt_input" \
    | sed 's|^work/armD_msas/||' > "$LIST"
  n=$(wc -l < "$LIST")
  total_done=$(( $(wc -l < "$MASTER") - 1 ))
  echo "[armD shard $SHARD gpu $GPU] $(date +%H:%M:%S) master=$total_done/2500, mine=$n"
  [ "$n" -eq 0 ] && break

  before=$total_done
  env CUDA_VISIBLE_DEVICES="$GPU" singularity exec --nv \
    --bind "$W":/work/users \
    --bind "$ROOT/ext/RoseTTAFold2-PPI":/home/RoseTTAFold2-PPI \
    "$ROOT/ext/SE3nv-20230612.sif" \
    /bin/bash -c "cd /work/users && python /home/RoseTTAFold2-PPI/src/predict_list_PPI.py \
       -list_fn $LIST -model_file /home/RoseTTAFold2-PPI/src/models/RF2-PPI.pt" \
    >> "$ROOT/logs/armD_shard${SHARD}.log" 2>&1
  rc=$?

  if [ -s "${LIST}.log" ]; then
    tail -n +2 "${LIST}.log" | sed 's|^protein_paired_msas//||' >> "$MASTER"
    sort -u -o "$MASTER" "$MASTER"
    rm -f "${LIST}.log"
  fi
  after=$(( $(wc -l < "$MASTER") - 1 ))

  if [ "$after" -le "$before" ]; then
    echo "[armD shard $SHARD] NO PROGRESS (rc=$rc, $before -> $after). Stopping." \
      | tee -a "$ROOT/logs/armD_shard${SHARD}.log"
    exit 1
  fi
  echo "[armD shard $SHARD] rc=$rc, master $before -> $after"
done
echo "[armD shard $SHARD] complete $(date +%F\ %H:%M)"
