#!/bin/bash
# Start the HuRI pilot once the specificity matrix releases the GPUs.
#
# Third in the chain: orphan screen -> specificity matrix -> HuRI pilot. The
# matrix comes first because it is the cheap test of a claim this project has
# already made in writing; HuRI is the expensive test of a claim it has not.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
exec >>"$ROOT/logs/huri_launch.log" 2>&1
echo "=== waiting for the specificity matrix $(date -u +%F\ %T) ==="
# wait for the matrix to start before waiting for it to finish, or this fires
# in the gap between the orphan screen ending and the matrix beginning
while ! pgrep -f "run_specificity_fold" >/dev/null; do sleep 60; done
echo "matrix running $(date -u +%F\ %T)"
while pgrep -f "run_specificity_fold" >/dev/null; do sleep 120; done
echo "matrix finished $(date -u +%F\ %T); starting the HuRI pilot"
sleep 90
cd "$ROOT" || exit 1
nohup setsid bash scripts/run_huri_pilot_fold.sh 0 1 2 >>logs/huri_shard0.out 2>&1 &
nohup setsid bash scripts/run_huri_pilot_fold.sh 1 2 2 >>logs/huri_shard1.out 2>&1 &
echo "launched 2 shards on GPUs 1 and 2 $(date -u +%F\ %T)"
