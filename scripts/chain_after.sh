#!/bin/bash
# Wait for a stage's shards to finish, then run the next thing.
#
#   chain_after.sh <tag> <command...>
#
# "Finished" means: at least one shard marker existed at some point, and now
# none do. Markers are files written by the fold runners and removed on exit,
# so this cannot be confused by a process whose command line happens to
# contain a script's name -- which is what fired the first two watchers early,
# the same failure as pkill -f matching its own invocation.
set -u
ROOT=/data/yzy21/yy/space_agent/structural_modelling
TAG=$1; shift
alive() { compgen -G "$ROOT/run/${TAG}_shard*.running" >/dev/null; }

echo "[chain:$TAG] waiting for shards to appear $(date -u +%F\ %T)"
for _ in $(seq 1 720); do alive && break; sleep 60; done
if ! alive; then echo "[chain:$TAG] shards never appeared; giving up"; exit 1; fi
echo "[chain:$TAG] shards running $(date -u +%F\ %T)"

while alive; do sleep 120; done
echo "[chain:$TAG] shards finished $(date -u +%F\ %T); running: $*"
sleep 60
cd "$ROOT" || exit 1
exec "$@"
