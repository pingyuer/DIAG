#!/usr/bin/env bash
# 双向 tarball 同步：本地 <-> 双 GPU 容器，附 code_sha=sha256sum。
# 用法: DIAG-code/sync.sh push [port] | pull [port] | sha
#   push: 本地代码打包 -> 容器 /root/DIAG（默认双容器全推）
#   pull: 容器 /root/DIAG/outputs 打包 -> 本地 outputs/remote-<port>/
#   sha:  打印当前代码包的 code_sha
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=root@172.16.240.188
PORTS_ALL="32237 31035"
REMOTE_DIR=/root/DIAG
SHA_FILE=.code_sha
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=15"

make_tarball() { # $1 = 输出 tgz 路径；同时刷新 .code_sha
  tar -czf "$1" \
    --exclude=.git --exclude=.venv --exclude=__pycache__ \
    --exclude=mlruns --exclude=outputs --exclude=checkpoints \
    --exclude='*.pt' --exclude='*.pth' --exclude=.board \
    --exclude=.omp/run --exclude=.omp/cache \
    DIAG-code src pyproject.toml .python-version
  sha256sum "$1" | awk '{print $1}' > "$SHA_FILE"
  echo "code_sha=$(cat "$SHA_FILE") tarball=$1"
}

do_push() {
  local tgz=/tmp/diag-code.tgz
  make_tarball "$tgz"
  for p in "$@"; do
    echo "== push :$p"
    # shellcheck disable=SC2086
    scp -P "$p" $SSH_OPTS "$tgz" "$SHA_FILE" "$HOST:/tmp/"
    # shellcheck disable=SC2086
    ssh -p "$p" $SSH_OPTS "$HOST" \
      "mkdir -p $REMOTE_DIR && tar -xzf /tmp/diag-code.tgz -C $REMOTE_DIR && cp /tmp/.code_sha $REMOTE_DIR/$SHA_FILE && cat $REMOTE_DIR/$SHA_FILE"
  done
}

do_pull() {
  mkdir -p outputs
  for p in "$@"; do
    echo "== pull :$p"
    # shellcheck disable=SC2086
    ssh -p "$p" $SSH_OPTS "$HOST" \
      "tar -czf /tmp/diag-out.tgz -C $REMOTE_DIR outputs" \
      || { echo "pull :$p 失败：远端 $REMOTE_DIR/outputs 不存在"; return 1; }
    # shellcheck disable=SC2086
    scp -P "$p" $SSH_OPTS "$HOST:/tmp/diag-out.tgz" "/tmp/diag-out-$p.tgz"
    mkdir -p "outputs/remote-$p"
    tar -xzf "/tmp/diag-out-$p.tgz" -C "outputs/remote-$p" --strip-components=1
    echo "pulled -> outputs/remote-$p/"
  done
}

cmd="${1:-}"; shift || true
case "$cmd" in
  push) ports="${*:-$PORTS_ALL}"; do_push $ports ;;
  pull) ports="${*:-$PORTS_ALL}"; do_pull $ports ;;
  sha)  make_tarball /tmp/diag-code-sha.tgz ;;
  *) echo "usage: $0 push [port] | pull [port] | sha" >&2; exit 2 ;;
esac
