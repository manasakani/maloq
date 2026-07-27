#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
FROZEN_LAUNCHER=${SC26_FROZEN_LAUNCHER:-${SCRIPT_DIR}/03_nabladft_nte64_edge2_next_controls_2gpu_impl.sh}

if [[ "${FROZEN_LAUNCHER}" != /* || ! -x "${FROZEN_LAUNCHER}" ]]; then
  echo "Frozen launcher must be an executable absolute path: ${FROZEN_LAUNCHER}" >&2
  exit 1
fi

unset SC26_FROZEN_LAUNCHER
exec "${FROZEN_LAUNCHER}" "$@"
