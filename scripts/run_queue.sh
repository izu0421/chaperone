#!/bin/bash
# The whole queue, in one process: orphan screen -> specificity matrix -> HuRI
# pilot, each scored as it finishes.
#
# The first wait is on the orphan shards' actual PIDs, because those shards
# were started before the marker-file convention existed. Everything after
# waits on marker files. Neither can be fooled by a command line, which is how
# the first attempt went wrong: `pgrep -f run_specificity_fold` matched the
# shell whose own command line contained that string, so two watchers fired
# before anything had started.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
cd "$ROOT" || exit 1
exec >>"$ROOT/logs/queue.log" 2>&1
STRUCTMOD=/data/yzy21/micromamba/envs/structmod/bin/python
JUPYTER=/home/yzy21/micromamba/envs/hsa/bin/jupyter

say() { echo "[queue] $(date -u +%F\ %T) $*"; }
wait_pids() { while kill -0 "$@" 2>/dev/null; do sleep 120; done; }
markers_gone() { ! compgen -G "$ROOT/run/$1_shard*.running" >/dev/null; }
wait_stage() {                      # $1 = tag
  for _ in $(seq 1 60); do compgen -G "$ROOT/run/$1_shard*.running" >/dev/null && break; sleep 30; done
  until markers_gone "$1"; do sleep 120; done
}

say "queue started; waiting for orphan shards $ORPHAN_PIDS"
for p in $ORPHAN_PIDS; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done
sleep 60
say "orphan screen finished: $(find results/orphan_out -maxdepth 2 -name '*_summary_confidences.json' | wc -l) of 5000 folds complete"

say "scoring the orphan screen"
"$STRUCTMOD" -I scripts/14_score_orphans.py --workers 3 --seed-from orphan_scores_snapshot.csv && say "orphan scoring done" || say "orphan scoring FAILED"

say "redrawing Figure 4"
(cd /data/yzy21/yy/space_agent/manuscript && "$JUPYTER" nbconvert --to notebook --execute --inplace fig4_deorphanisation.ipynb) \
  && say "Figure 4 redrawn" || say "Figure 4 FAILED"

say "launching the specificity matrix on GPUs 1 and 2"
nohup setsid bash scripts/run_specificity_fold.sh 0 1 2 >>logs/spec_shard0.out 2>&1 &
nohup setsid bash scripts/run_specificity_fold.sh 1 2 2 >>logs/spec_shard1.out 2>&1 &
wait_stage specificity
sleep 60
say "specificity matrix finished: $(ls results/specificity_out 2>/dev/null | wc -l) of 256"
"$STRUCTMOD" -I scripts/18_score_specificity.py --workers 3 && say "matrix scored" || say "matrix scoring FAILED"

say "launching the HuRI pilot on GPUs 1 and 2"
nohup setsid bash scripts/run_huri_pilot_fold.sh 0 1 2 >>logs/huri_shard0.out 2>&1 &
nohup setsid bash scripts/run_huri_pilot_fold.sh 1 2 2 >>logs/huri_shard1.out 2>&1 &
wait_stage huri
sleep 60
say "HuRI pilot finished: $(ls results/huri_pilot_out 2>/dev/null | wc -l) of 600"
say "queue complete"
