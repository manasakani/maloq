#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
SSH_WRAPPER=${PROJECT_ROOT}/_auto_script/omol_open_shell_transfer/quasar_cpu_ssh.sh
REMOTE_HOST=irteam@quasar-cpu-0
REMOTE_DATASET=/home1/irteam/data-vol1/data/omol25/open_shell_maloq_ase
REMOTE_MANIFEST=/home1/irteam/data-vol1/datasets/omol25/manifests/ml_mo_v1/strict_transition_metal.jsonl
LOCAL_DATASET=/dataset/seongsu/shared-home/datasets/omol25_open_shell_maloq_ase
EXPECTED_DB_BYTES=2005560193024
EXPECTED_TRAIN_SHARDS=945
EXPECTED_VAL_SHARDS=9
EXPECTED_TEST_SHARDS=11
EXPECTED_MANIFEST_SHA256=40cc77a84160862dfe8b73a6a867e24ec2bb881f1aacd3eb1fbf580860030f8f
RSYNC_MAX_ATTEMPTS=${RSYNC_MAX_ATTEMPTS:-100}

usage() {
  echo "Usage: $0 {status|sync|verify}" >&2
  exit 2
}

remote() {
  "${SSH_WRAPPER}" "${REMOTE_HOST}" "$@"
}

sync_path() {
  local source=$1
  local destination=$2
  local attempt=1
  while true; do
    if rsync \
      -rlt \
      --partial \
      --partial-dir=.rsync-partial \
      --protect-args \
      --human-readable \
      --info=progress2,stats2 \
      --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
      -e "${SSH_WRAPPER}" \
      "${REMOTE_HOST}:${source}" \
      "${destination}"; then
      return
    fi
    if (( attempt >= RSYNC_MAX_ATTEMPTS )); then
      echo "rsync failed after ${attempt} attempts: ${source}" >&2
      return 1
    fi
    echo "rsync attempt ${attempt} failed; retrying in 30 seconds: ${source}" >&2
    attempt=$((attempt + 1))
    sleep 30
  done
}

local_db_count() {
  local split=$1
  if [[ ! -d "${LOCAL_DATASET}/${split}" ]]; then
    echo 0
    return
  fi
  find "${LOCAL_DATASET}/${split}" -maxdepth 1 -type f -name '*.db' | wc -l
}

status() {
  echo "Remote dataset: ${REMOTE_HOST}:${REMOTE_DATASET}"
  remote \
    "python -c \"import json; p='${REMOTE_DATASET}/dataset_metadata.json'; d=json.load(open(p)); print(d['status'], d['result']['database_bytes'], d['result']['successful_rows'])\""
  echo "Local dataset: ${LOCAL_DATASET}"
  printf 'Local shards: train=%s val=%s test=%s\n' \
    "$(local_db_count train)" "$(local_db_count val)" "$(local_db_count test)"
  df -P -B1 /dataset | tail -1
}

sync_dataset() {
  if [[ ! -x "${SSH_WRAPPER}" ]]; then
    echo "SSH wrapper is not executable: ${SSH_WRAPPER}" >&2
    exit 1
  fi
  if [[ -e /dataset/seongsu/shared-home/datasets/omol25_metal_organic_density ]]; then
    echo "Legacy canonical dataset still exists; deprecate it before syncing." >&2
    exit 1
  fi
  remote \
    "test -f '${REMOTE_DATASET}/dataset_metadata.json' && test -d '${REMOTE_DATASET}/train' && test -d '${REMOTE_DATASET}/val' && test -d '${REMOTE_DATASET}/test'"

  mkdir -p "${LOCAL_DATASET}" "${LOCAL_DATASET}/manifests"
  mkdir -p \
    "${LOCAL_DATASET}/train" \
    "${LOCAL_DATASET}/val" \
    "${LOCAL_DATASET}/test"

  # Seed one complete shard per split first so loader validation can proceed
  # while the resumable full synchronization continues.
  sync_path \
    "${REMOTE_DATASET}/dataset_metadata.json" \
    "${LOCAL_DATASET}/"
  sync_path \
    "${REMOTE_DATASET}/train/omol_open_shell_train_00000.db" \
    "${LOCAL_DATASET}/train/"
  sync_path \
    "${REMOTE_DATASET}/val/omol_open_shell_val_00000.db" \
    "${LOCAL_DATASET}/val/"
  sync_path \
    "${REMOTE_DATASET}/test/omol_open_shell_test_00000.db" \
    "${LOCAL_DATASET}/test/"

  sync_path "${REMOTE_DATASET}/" "${LOCAL_DATASET}/"
  sync_path \
    "${REMOTE_MANIFEST}" \
    "${LOCAL_DATASET}/manifests/strict_transition_metal.jsonl"
  verify_dataset
}

verify_dataset() {
  /dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python - \
    "${LOCAL_DATASET}" \
    "${EXPECTED_DB_BYTES}" \
    "${EXPECTED_TRAIN_SHARDS}" \
    "${EXPECTED_VAL_SHARDS}" \
    "${EXPECTED_TEST_SHARDS}" \
    "${EXPECTED_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_db_bytes = int(sys.argv[2])
expected_shards = {
    "train": int(sys.argv[3]),
    "val": int(sys.argv[4]),
    "test": int(sys.argv[5]),
}
expected_manifest_sha256 = sys.argv[6]

metadata_path = root / "dataset_metadata.json"
if not metadata_path.is_file():
    raise FileNotFoundError(metadata_path)
metadata = json.loads(metadata_path.read_text())
if metadata.get("status") != "SUCCEEDED":
    raise RuntimeError(f"dataset status is {metadata.get('status')!r}")
if int(metadata["result"]["database_bytes"]) != expected_db_bytes:
    raise RuntimeError("dataset metadata byte count changed")

actual_db_bytes = 0
actual_shards = {}
for split, expected in expected_shards.items():
    databases = sorted((root / split).glob("*.db"))
    actual_shards[split] = len(databases)
    if len(databases) != expected:
        raise RuntimeError(
            f"{split} has {len(databases)} shards; expected {expected}"
        )
    actual_db_bytes += sum(path.stat().st_size for path in databases)
if actual_db_bytes != expected_db_bytes:
    raise RuntimeError(
        f"database bytes={actual_db_bytes}; expected={expected_db_bytes}"
    )

manifest_path = root / "manifests" / "strict_transition_metal.jsonl"
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if digest != expected_manifest_sha256:
    raise RuntimeError(f"manifest SHA256 changed: {digest}")

print(
    json.dumps(
        {
            "status": "verified",
            "schema": metadata["schema"],
            "rows": metadata["result"]["successful_rows"],
            "shards": actual_shards,
            "database_bytes": actual_db_bytes,
            "manifest_sha256": digest,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
}

[[ $# -eq 1 ]] || usage
case "$1" in
  status) status ;;
  sync) sync_dataset ;;
  verify) verify_dataset ;;
  *) usage ;;
esac
