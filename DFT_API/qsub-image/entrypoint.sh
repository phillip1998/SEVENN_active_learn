#!/bin/bash
set -euo pipefail

SGE_ROOT="${SGE_ROOT:-/var/lib/gridengine}"
SGE_CELL="${SGE_CELL:-default}"
SGE_QMASTER_PORT="${SGE_QMASTER_PORT:-6444}"
SGE_EXECD_PORT="${SGE_EXECD_PORT:-6445}"
ACT_QMASTER="${SGE_ROOT}/${SGE_CELL}/common/act_qmaster"

if [[ ! -f "${SGE_ROOT}/${SGE_CELL}/common/bootstrap" ]]; then
  FOUND_BOOTSTRAP="$(
    find /var/lib/gridengine /opt/sge /usr/local/sge /usr/share/gridengine /usr/lib/gridengine \
      -path "*/common/bootstrap" -type f -print -quit 2>/dev/null || true
  )"
  if [[ -n "$FOUND_BOOTSTRAP" ]]; then
    COMMON_DIR="$(dirname "$FOUND_BOOTSTRAP")"
    SGE_CELL="$(basename "$(dirname "$COMMON_DIR")")"
    SGE_ROOT="$(dirname "$(dirname "$COMMON_DIR")")"
    ACT_QMASTER="${SGE_ROOT}/${SGE_CELL}/common/act_qmaster"
  fi
fi

export SGE_ROOT SGE_CELL SGE_QMASTER_PORT SGE_EXECD_PORT

if [[ "${QSUB_HELPER_DEBUG:-0}" == "1" ]]; then
  echo "[qsub-helper] SGE_ROOT=$SGE_ROOT" >&2
  echo "[qsub-helper] SGE_CELL=$SGE_CELL" >&2
  echo "[qsub-helper] SGE_QMASTER_PORT=$SGE_QMASTER_PORT" >&2
  echo "[qsub-helper] SGE_EXECD_PORT=$SGE_EXECD_PORT" >&2
  echo "[qsub-helper] ACT_QMASTER=$ACT_QMASTER" >&2
  echo "[qsub-helper] SGE_QMASTER=${SGE_QMASTER:-}" >&2
  if [[ -f "${SGE_ROOT}/${SGE_CELL}/common/bootstrap" ]]; then
    echo "[qsub-helper] bootstrap=${SGE_ROOT}/${SGE_CELL}/common/bootstrap" >&2
  else
    echo "[qsub-helper] bootstrap=missing" >&2
  fi
fi

CURRENT_QMASTER=""
if [[ -f "$ACT_QMASTER" ]]; then
  CURRENT_QMASTER="$(tr -d '[:space:]' < "$ACT_QMASTER" || true)"
fi

if [[ -z "$CURRENT_QMASTER" && -n "${SGE_QMASTER:-}" ]]; then
  mkdir -p "$(dirname "$ACT_QMASTER")"
  printf "%s\n" "$SGE_QMASTER" > "$ACT_QMASTER"
  CURRENT_QMASTER="$SGE_QMASTER"
fi

if [[ "${QSUB_HELPER_DEBUG:-0}" == "1" ]]; then
  echo "[qsub-helper] act_qmaster=$CURRENT_QMASTER" >&2
fi

if [[ -z "$CURRENT_QMASTER" ]]; then
  cat >&2 <<'EOF'
Unable to find Grid Engine qmaster config.

Expected:
  /var/lib/gridengine/default/common/act_qmaster

Fix by mounting the host Grid Engine config:
  -v /var/lib/gridengine:/var/lib/gridengine:ro

Or set SGE_QMASTER:
  -e SGE_QMASTER=your-qmaster-hostname
EOF
  exit 2
fi

if [[ ! -f "${SGE_ROOT}/${SGE_CELL}/common/bootstrap" ]]; then
  cat >&2 <<EOF
Unable to find Grid Engine bootstrap config.

Expected:
  ${SGE_ROOT}/${SGE_CELL}/common/bootstrap

Run build_from_host.sh on the cluster submit host where qsub works, or mount the real SGE_ROOT.
EOF
  exit 2
fi

if [[ -d /usr/local/sge-libs ]]; then
  export LD_LIBRARY_PATH="/usr/local/sge-libs:${LD_LIBRARY_PATH:-}"
fi

exec /bin/bash -lc "$*"
