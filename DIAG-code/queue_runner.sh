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
# pull latest tasks each loop (no ssh needed to feed the queue)
git pull -q 2>/dev/null || true
while true; do
  git pull -q 2>/dev/null || true
  # Runner identity: hostname is stable per container (tahara-<id>-main).
  # Tasks may be prefixed <id>-<name>.sh to pin a runner; unprefixed or
  # already-claimed tasks are shared. Claim via atomic mv to done/*.running.
  ME=$(hostname)
  # OWNER-PREFIX protocol (fix double-claim 2026-10-02): task files MUST be
  # named <owner>-<name>.sh where owner is a runner hostname prefix or
  # "shared". A runner ONLY takes tasks matching its hostname prefix or
  # "shared". Cross-runner dup happened because task-* matched everyone and
  # per-container git clones diverge (claim is local mv, invisible to peer
  # until next pull). shared-* still races; use only for idempotent tasks.
  task=""
  for cand in "$PENDING"/"$ME"-*.sh "$PENDING"/shared-*.sh; do
    [ -e "$cand" ] || continue
    task="$cand"; break
  done
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
    rc=$?
    echo "[$(date -Is)] FAILED $name rc=$rc" | tee -a "outputs/queue/$name.log"
    mv "$DONE/$name.sh.running" "$FAILED/$name.sh"
  fi
  exec 9>&-
  rm -f "$lock"
done
