#!/bin/bash
# Wait for the orphan screen's fold shards to exit, then score and redraw.
#
# The shards stop themselves: run_orphan_fold.sh breaks when its queue is
# empty and exits 1 on no progress, so "no shard processes left" is the real
# completion signal and does not need a count to be guessed at. A shard that
# died early also ends the wait, which is why the fold count is checked and
# reported rather than assumed.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
LOG=$ROOT/logs/on_complete.log
exec >>"$LOG" 2>&1
echo "=== watcher started $(date -u +%F\ %T) ==="

while pgrep -f "run_orphan_fold" >/dev/null; do sleep 120; done
echo "shards exited $(date -u +%F\ %T)"
sleep 60                                  # let the last CIF finish writing

n_folds=$(find "$ROOT/results/orphan_out" -maxdepth 1 -mindepth 1 -type d | wc -l)
n_want=$(( $(wc -l < "$ROOT/work/orphan_manifest.csv") - 1 ))
echo "folds on disk: $n_folds of $n_want"

echo "--- scoring"
~/micromamba/envs/structmod/bin/python -I "$ROOT/scripts/14_score_orphans.py" --workers 24 || exit 1

echo "--- redrawing Figure 4"
cd /data/yzy21/yy/space_agent/manuscript || exit 1
~/micromamba/envs/hsa/bin/jupyter nbconvert --to notebook --execute --inplace fig4_deorphanisation.ipynb || exit 1

echo "=== watcher done $(date -u +%F\ %T) ==="
