#!/bin/bash
# The whole queue in one process: orphan screen -> specificity matrix -> HuRI pilot,
# each scored as it finishes.
#
# Written as a v2 file rather than an edit of run_queue.sh because bash reads a
# script incrementally as it executes: rewriting the file a running shell is
# reading moves the byte offset under it and makes it execute nonsense.
#
# Three rules, each from a way this queue has gone wrong:
#
# 1. Never wait on a process by pattern. `pgrep -f run_specificity_fold` matches
#    the shell whose own command line contains that string, which already fired
#    two watchers before anything had started. The orphan shards predate the
#    marker convention so they are waited on by actual PID; every later stage
#    waits on marker files that a shard creates and removes via trap.
#
# 2. Never advance on a stage that did not actually run. The old version
#    advanced whenever the markers disappeared, which is also what happens when
#    a shard dies in its first second -- GPU busy, bad path, OOM. It would then
#    score nothing, log FAILED, and launch the next stage anyway, so one early
#    death cost the whole queue. Each stage now has to clear a completion bar,
#    and the queue stops if it does not.
#
# 3. Restart a shard that dies with work still to do, instead of silently
#    running the rest of the stage at half throughput.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
cd "$ROOT" || exit 1
exec >>"$ROOT/logs/queue.log" 2>&1
STRUCTMOD=/data/yzy21/micromamba/envs/structmod/bin/python
JUPYTER=/home/yzy21/micromamba/envs/hsa/bin/jupyter
WORKERS=3                 # this box has 4 cores; the GPU folds need the rest

say() { echo "[queue] $(date -u +%F\ %T) $*"; }
die() { say "STOPPING: $*"; exit 1; }

n_done() {                # $1 = out dir under results/
  find "$ROOT/results/$1" -maxdepth 2 -name '*_summary_confidences.json' 2>/dev/null | wc -l
}
markers() { compgen -G "$ROOT/run/$1_shard*.running" 2>/dev/null | wc -l; }

# Wait for a fold stage, restarting a dead shard while work remains, and refuse to
# return unless the stage really got somewhere.
# $1 tag  $2 runner script  $3 out dir  $4 expected total  $5 min fraction to accept
wait_stage() {
  local tag=$1 runner=$2 out=$3 total=$4 minfrac=$5
  local waited=0
  while [ "$(markers "$tag")" -eq 0 ] && [ $waited -lt 1800 ]; do
    sleep 30; waited=$((waited + 30))
  done
  [ "$(markers "$tag")" -eq 0 ] && die "$tag: no shard came up within 30 min"

  local last_done=-1 stuck=0
  while [ "$(markers "$tag")" -gt 0 ]; do
    sleep 120
    local now; now=$(n_done "$out")
    if [ "$now" -eq "$last_done" ]; then
      stuck=$((stuck + 1))
    else
      stuck=0; last_done=$now
    fi
    # a shard died but folds remain: bring it back rather than finish at half rate
    if [ "$(markers "$tag")" -lt 2 ] && [ "$now" -lt "$total" ]; then
      local i
      for i in 0 1; do
        if [ ! -f "$ROOT/run/${tag}_shard$i.running" ]; then
          say "$tag: shard $i is not running and $((total - now)) folds remain; restarting"
          nohup setsid bash "scripts/$runner" "$i" "$((i + 1))" 2 \
            >>"logs/${tag}_shard$i.out" 2>&1 &
          sleep 30
        fi
      done
    fi
    [ $stuck -ge 90 ] && die "$tag: no fold completed in 3 h with a shard alive"
  done

  sleep 60
  local got; got=$(n_done "$out")
  local need; need=$(awk -v t="$total" -v f="$minfrac" 'BEGIN{printf "%d", t*f}')
  say "$tag finished: $got of $total folds"
  [ "$got" -lt "$need" ] && die "$tag produced only $got of $total folds (needed $need). \
Not scoring it and not starting the next stage: a stage that did not run makes every \
number after it meaningless."
  return 0
}

say "=== queue v2 started (pid $$) ==="

# --- stage 1: the orphan screen, already running on PIDs given by the caller ---
if [ -n "${ORPHAN_PIDS:-}" ]; then
  say "waiting for orphan shards $ORPHAN_PIDS"
  for p in $ORPHAN_PIDS; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done
  sleep 60
fi
ORPHAN_DONE=$(n_done orphan_out)
say "orphan screen: $ORPHAN_DONE of 5000 folds"
[ "$ORPHAN_DONE" -lt 4500 ] && die "orphan screen only reached $ORPHAN_DONE of 5000"

say "scoring the orphan screen"
"$STRUCTMOD" -I scripts/14_score_orphans.py --workers $WORKERS \
  --seed-from orphan_scores_snapshot.csv \
  && say "orphan scoring done" || die "orphan scoring failed"

say "checking fold provenance"
"$STRUCTMOD" -I scripts/check_fold_provenance.py --stage orphan \
  && say "provenance clean" || say "PROVENANCE WARNING -- see above"

say "redrawing Figure 4"
(cd /data/yzy21/yy/space_agent/manuscript \
  && "$JUPYTER" nbconvert --to notebook --execute --inplace fig4_deorphanisation.ipynb) \
  && say "Figure 4 redrawn" || say "Figure 4 FAILED (the figure guard may have refused a stale file)"

# --- stage 2: the specificity matrix ---
say "launching the specificity matrix on GPUs 1 and 2"
nohup setsid bash scripts/run_specificity_fold.sh 0 1 2 >>logs/specificity_shard0.out 2>&1 &
nohup setsid bash scripts/run_specificity_fold.sh 1 2 2 >>logs/specificity_shard1.out 2>&1 &
wait_stage specificity run_specificity_fold.sh specificity_out 256 0.95 || exit 1
"$STRUCTMOD" -I scripts/18_score_specificity.py --workers $WORKERS \
  && say "matrix scored" || say "matrix scoring FAILED"

# --- stage 3: the HuRI pilot ---
say "launching the HuRI pilot on GPUs 1 and 2"
nohup setsid bash scripts/run_huri_pilot_fold.sh 0 1 2 >>logs/huri_shard0.out 2>&1 &
nohup setsid bash scripts/run_huri_pilot_fold.sh 1 2 2 >>logs/huri_shard1.out 2>&1 &
wait_stage huri run_huri_pilot_fold.sh huri_pilot_out 600 0.90 || exit 1

say "=== queue complete ==="
"$STRUCTMOD" -I scripts/00_status.py
