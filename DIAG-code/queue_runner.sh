#!/usr/bin/env bash
# Queue runner: one per GPU container, serial task execution.
# Tasks: single-command .sh files in DIAG-code/queue/pending/.
# Done -> ../done/, failed -> ../failed/; logs -> outputs/queue/<task>.log.
# Idle: sleep 30s. Add tasks by dropping files (git pull) -- no ssh needed.
set -euo pipefail
cd /root/DIAG_fresh 2>/dev/null || cd /root/DIAG
PENDING=DIAG-code/queue/pending
DONE=DIAG-code/queue/done
FAILED=DIAG-code/queue/failed
mkdir -p "$PENDING" "$DONE" "$FAILED" outputs/queue
while true; do
  task=$(ls "$PENDING"/*.sh 2>/dev/null | head -1 || true)
  if [ -z "$task" ]; then sleep 30; continue; fi
  name=$(basename "$task" .sh)
  lock="$PENDING/.$name.lock"
  exec 9>"$lock" || continue
  flock -n 9 || { exec 9>&-; continue; }
  mv "$task" "$DONE/$name.sh.running"
  echo "[$(date -Is)] START $name" | tee "outputs/queue/$name.log"
  if bash "$DONE/$name.sh.running" >> "outputs/queue/$name.log" 2>&1; then
    echo "[$(date -Is)] DONE $name" | tee -a "outputs/queue/$name.log"
    mv "$DONE/$name.sh.running" "$DONE/$name.sh"
  else
    echo "[$(date -Is)] FAILED $name rc=$?" | tee -a "outputs/queue/$name.log"
    mv "$DONE/$name.sh.running" "$FAILED/$name.sh"
  fi
  exec 9>&-
  rm -f "$lock"
done
