#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/dataset/seongsu/shared-home/workspace/project"
TOOL_DIR="${PROJECT_ROOT}/_auto_script/dft_visualization"
DFT_DATASET_DIR="/dataset/seongsu/shared-home/projects/dft-dataset"
DFT_MONITOR_DIR="/dataset/seongsu/shared-home/projects/dft-monitor"
BASE_PY="/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python"
VIS_ENV="/dataset/seongsu/shared-home/conda/envs/proj-dft-visualization-sc26"
VIS_PY="${VIS_ENV}/bin/python"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/dft-visualization"
PREDICTIONS_DIR="${OUTPUT_DIR}/predictions"

export DFT_SHARED_ROOT="/dataset/seongsu/shared-home"
export DFT_MONITOR_WEB_DIR="${DFT_MONITOR_DIR}"
export DFT_PREDICTIONS_ROOT="${PREDICTIONS_DIR}"
export DFT_CACHE_DIR="${OUTPUT_DIR}/cache"

usage() {
  echo "Usage: $0 {prepare|validate|serve}"
  echo "  prepare   create the lightweight web environment"
  echo "  validate  run backend, Hamiltonian, density, and 3D-grid checks"
  echo "  serve     run the dashboard in the foreground"
}

require_sources() {
  test -f "${DFT_DATASET_DIR}/server.py"
  test -f "${DFT_MONITOR_DIR}/detail.html"
  test -f "${DFT_MONITOR_DIR}/data/index.json"
  test -f "${DFT_MONITOR_DIR}/data/samples/qh9_000002.json"
}

scope="${1:-}"
case "${scope}" in
  prepare)
    require_sources
    if [[ ! -x "${VIS_PY}" ]]; then
      "${BASE_PY}" -m venv --system-site-packages "${VIS_ENV}"
    fi
    "${VIS_PY}" -m pip install --disable-pip-version-check -r "${TOOL_DIR}/requirements.txt"
    ;;
  validate)
    require_sources
    if [[ ! -x "${VIS_PY}" ]]; then
      echo "Visualization environment is missing. Run: $0 prepare" >&2
      exit 1
    fi
    mkdir -p "${OUTPUT_DIR}" "${PREDICTIONS_DIR}"
    "${VIS_PY}" "${TOOL_DIR}/validate_visualizer.py"
    ;;
  serve)
    require_sources
    if [[ ! -x "${VIS_PY}" ]]; then
      echo "Visualization environment is missing. Run: $0 prepare" >&2
      exit 1
    fi
    mkdir -p "${OUTPUT_DIR}" "${PREDICTIONS_DIR}"
    host="${DFT_VIS_HOST:-127.0.0.1}"
    port="${DFT_VIS_PORT:-9100}"
    cd "${DFT_DATASET_DIR}"
    exec "${VIS_PY}" -m uvicorn server:app --host "${host}" --port "${port}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
