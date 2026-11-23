#!/bin/bash
# Start the specificity matrix once the orphan screen releases the GPUs.
#
# GPU 0 is left alone: another user's job is resident on it. The orphan
# screen runs shards on GPUs 1 and 2, so this takes those two over as they
# free up. Scoring the orphan screen is CPU work and overlaps fine.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
exec >>"$ROOT/logs/spec_launch.log" 2>&1
echo "=== waiting for the orphan screen $(date -u +%F\ %T) ==="
while pgrep -f "run_orphan_fold" >/dev/null; do sleep 120; done
echo "orphan shards exited $(date -u +%F\ %T); starting the specificity matrix"
sleep 90
cd "$ROOT" || exit 1
nohup setsid bash scripts/run_specificity_fold.sh 0 1 2 >>logs/spec_shard0.out 2>&1 &
nohup setsid bash scripts/run_specificity_fold.sh 1 2 2 >>logs/spec_shard1.out 2>&1 &
echo "launched 2 shards on GPUs 1 and 2 $(date -u +%F\ %T)"
