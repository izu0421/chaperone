#!/bin/bash
# Wait for the FULL build, then fold across three GPUs.
#
# Waiting for the manifest to exist was not enough: a --limit 40 test run had
# already written one, so the shards started on 40 jobs and would have exited
# when those were done, leaving nothing folding when the real manifest landed.
# A stale artefact read as current, which is this project's most frequent bug.
# The gate is now the row count AND the file being stable for two checks.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
cd "$ROOT"
MAN=work/v2_manifest.csv
MIN=4000
prev=-1
while true; do
  n=0
  [ -f "$MAN" ] && n=$(( $(wc -l < "$MAN") - 1 ))
  if [ "$n" -ge "$MIN" ] && [ "$n" -eq "$prev" ]; then break; fi
  prev=$n
  sleep 30
done
echo "$(date +%H:%M) manifest stable at $n inputs"
for i in 0 1 2; do
  # only start a shard that is not already running
  if [ ! -f "run/v2_shard$i.running" ]; then
    nohup bash scripts/run_v2_fold.sh $i $i 3 > logs/v2_launch_$i.out 2>&1 &
    echo "  shard $i -> GPU $i"
    sleep 4
  else
    echo "  shard $i already running"
  fi
done
