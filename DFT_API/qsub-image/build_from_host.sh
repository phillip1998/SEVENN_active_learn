#!/bin/bash
set -euo pipefail

IMAGE_NAME="${1:-nfs-shared-storage-image}"
CONTEXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_ROOT="${CONTEXT_DIR}/host-root"

copy_config_path() {
  local src="$1"
  local dst="${HOST_ROOT}${src}"
  mkdir -p "$(dirname "$dst")"
  cp -L "$src" "$dst"
}

copy_library() {
  local src="$1"
  local base
  base="$(basename "$src")"

  case "$base" in
    libc.so.*|ld-linux*.so.*|libpthread.so.*|libdl.so.*|libm.so.*|librt.so.*|libresolv.so.*|libnsl.so.*|libutil.so.*)
      return 0
      ;;
  esac

  local dst="${HOST_ROOT}/usr/local/sge-libs/$(basename "$src")"
  mkdir -p "$(dirname "$dst")"
  cp -L "$src" "$dst"
}

copy_binary_with_libs() {
  local bin="$1"
  local target_name="$2"

  if [[ ! -x "$bin" ]]; then
    echo "Missing executable: $bin" >&2
    exit 1
  fi

  mkdir -p "${HOST_ROOT}/usr/local/bin"
  cp -L "$bin" "${HOST_ROOT}/usr/local/bin/${target_name}"

  ldd "$bin" \
    | awk '
      /=> \// { print $3 }
      /^[[:space:]]*\// { print $1 }
    ' \
    | while read -r lib; do
        [[ -n "$lib" && -f "$lib" ]] && copy_library "$lib"
      done
}

copy_dir_best_effort() {
  local src="$1"
  local dst="${HOST_ROOT}${src}"
  mkdir -p "$(dirname "$dst")"
  mkdir -p "$dst"
  tar -C "$src" --ignore-failed-read --warning=no-file-changed -cf - . \
    | tar -C "$dst" -xf -
}

rm -rf "$HOST_ROOT"
mkdir -p "$HOST_ROOT"

QSUB_BIN="$(command -v qsub || true)"
if [[ -z "$QSUB_BIN" ]]; then
  echo "qsub was not found in PATH. Run this script on the cluster submit host." >&2
  exit 1
fi

copy_binary_with_libs "$QSUB_BIN" qsub

for tool in qstat qdel qalter qconf; do
  tool_path="$(command -v "$tool" || true)"
  if [[ -n "$tool_path" ]]; then
    copy_binary_with_libs "$tool_path" "$tool"
  fi
done

if [[ -d /var/lib/gridengine ]]; then
  mkdir -p "${HOST_ROOT}/var/lib"
  copy_dir_best_effort /var/lib/gridengine
fi

if [[ -d /etc/gridengine ]]; then
  mkdir -p "${HOST_ROOT}/etc"
  copy_dir_best_effort /etc/gridengine
fi

if [[ -n "${SGE_ROOT:-}" && -d "${SGE_ROOT}" ]]; then
  if [[ -d "${SGE_ROOT}/${SGE_CELL:-default}/common" ]]; then
    copy_dir_best_effort "${SGE_ROOT}/${SGE_CELL:-default}/common"
  else
    copy_dir_best_effort "$SGE_ROOT"
  fi
fi

while read -r bootstrap; do
  [[ -z "$bootstrap" ]] && continue
  common_dir="$(dirname "$bootstrap")"
  cell_dir="$(dirname "$common_dir")"
  copy_dir_best_effort "$common_dir"
done < <(
  find /var/lib/gridengine /opt/sge /usr/local/sge /usr/share/gridengine /usr/lib/gridengine \
    -path "*/common/bootstrap" -type f -print 2>/dev/null || true
)

for dir in /usr/lib/gridengine /usr/share/gridengine; do
  if [[ -d "$dir" ]]; then
    copy_dir_best_effort "$dir"
  fi
done

if ! find "$HOST_ROOT" -path "*/common/bootstrap" -type f -print -quit | grep -q .; then
  cat >&2 <<'EOF'
Warning: no Grid Engine common/bootstrap file was copied.
If qsub works on this host, find the real file with:
  find / -path '*/common/bootstrap' 2>/dev/null
Then set SGE_ROOT before building, for example:
  SGE_ROOT=/opt/sge bash build_from_host.sh nfs-shared-storage-image
EOF
fi

docker build --no-cache -t "$IMAGE_NAME" "$CONTEXT_DIR"
