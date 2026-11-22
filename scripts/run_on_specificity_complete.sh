#!/bin/bash
# Score the specificity matrix once its fold shards exit.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
exec >>"$ROOT/logs/spec_on_complete.log" 2>&1
echo "=== watcher started $(date -u +%F\ %T) ==="
# wait for the shards to appear first, so this does not fire before they start
while ! pgrep -f "run_specificity_fold" >/dev/null; do sleep 60; done
echo "shards running $(date -u +%F\ %T)"
while pgrep -f "run_specificity_fold" >/dev/null; do sleep 120; done
echo "shards exited $(date -u +%F\ %T)"
sleep 60
n=$(find "$ROOT/results/specificity_out" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
echo "folds on disk: $n of 256"
~/micromamba/envs/structmod/bin/python -I "$ROOT/scripts/18_score_specificity.py" --workers 24
echo "=== watcher done $(date -u +%F\ %T) ==="
