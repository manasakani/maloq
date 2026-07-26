#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
SSH_WRAPPER=${PROJECT_ROOT}/_auto_script/omol_open_shell_transfer/quasar_cpu_ssh.sh
REMOTE_HOST=irteam@quasar-cpu-0
REMOTE_DATASET=/home1/irteam/data-vol1/datasets/omol25/lmdb/omol_dm_unsolvated_electrolytes_all_supported_elements_85_5_10_e3nn_sad_orca_raw_sign_charge_scaled_fast_v1
REMOTE_INDEX=/home1/irteam/data-vol1/datasets/omol25/lmdb/omol_dm_unsolvated_electrolytes_all_supported_elements_85_5_10_e3nn_sad_orca_raw_sign_charge_scaled_fast_completed_index_20260608
REMOTE_MANIFEST=/home1/irteam/data-vol1/datasets/omol25/manifests/unsolvated_electrolytes_all_supported_elements_85_5_10_v1
LOCAL_DATASET=/dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb
LOCAL_INDEX=${LOCAL_DATASET}/_index
LOCAL_MANIFEST=${LOCAL_DATASET}/manifests/unsolvated_electrolytes_all_supported_elements_85_5_10_v1
TRANSFER_DIR=${LOCAL_DATASET}/_transfer
FILE_LIST=${TRANSFER_DIR}/completed-files.txt
SEED_LIST=${TRANSFER_DIR}/seed-files.txt
PYTHON=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
VERIFY=${PROJECT_ROOT}/_auto_script/omol_electrolyte_transfer/verify_completed_electrolyte.py
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

sync_file_list() {
  local list=$1
  local attempt=1
  while true; do
    if rsync \
      -rlt \
      --files-from="${list}" \
      --partial \
      --partial-dir=.rsync-partial \
      --protect-args \
      --human-readable \
      --info=progress2,stats2 \
      --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
      -e "${SSH_WRAPPER}" \
      "${REMOTE_HOST}:${REMOTE_DATASET}/" \
      "${LOCAL_DATASET}/"; then
      return
    fi
    if (( attempt >= RSYNC_MAX_ATTEMPTS )); then
      echo "rsync failed after ${attempt} attempts: ${list}" >&2
      return 1
    fi
    echo "rsync attempt ${attempt} failed; retrying in 30 seconds: ${list}" >&2
    attempt=$((attempt + 1))
    sleep 30
  done
}

build_file_lists() {
  "${PYTHON}" - "${LOCAL_INDEX}" "${REMOTE_DATASET}" "${FILE_LIST}" "${SEED_LIST}" <<'PY'
import json
import sys
from pathlib import Path

index_root = Path(sys.argv[1])
remote_root = Path(sys.argv[2])
file_list = Path(sys.argv[3])
seed_list = Path(sys.argv[4])

all_files = {
    "summary.json",
    "train.shard_lengths.json",
    "val.shard_lengths.json",
}
seed_files = set()
for split in ("train", "val", "test"):
    shards = {}
    with (index_root / f"{split}.index.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            lmdb = Path(row["lmdb"])
            summary = Path(row["summary"])
            if not lmdb.is_relative_to(remote_root):
                raise RuntimeError(f"LMDB path escaped source root: {lmdb}")
            if not summary.is_relative_to(remote_root):
                raise RuntimeError(f"summary path escaped source root: {summary}")
            shards[lmdb] = summary
    if not shards:
        raise RuntimeError(f"{split} completed index is empty")
    for shard, summary in shards.items():
        relative_shard = shard.relative_to(remote_root)
        all_files.add(str(relative_shard / "data.mdb"))
        all_files.add(str(relative_shard / "lock.mdb"))
        all_files.add(str(summary.relative_to(remote_root)))
    first_shard = sorted(shards)[0]
    first_summary = shards[first_shard]
    relative_first = first_shard.relative_to(remote_root)
    seed_files.update(
        {
            str(relative_first / "data.mdb"),
            str(relative_first / "lock.mdb"),
            str(first_summary.relative_to(remote_root)),
        }
    )

file_list.write_text("\n".join(sorted(all_files)) + "\n")
seed_list.write_text("\n".join(sorted(seed_files)) + "\n")
print(f"completed file list: {len(all_files):,} files")
print(f"seed file list: {len(seed_files):,} files")
PY
}

local_shards() {
  local split=$1
  if [[ ! -d "${LOCAL_DATASET}/${split}" ]]; then
    echo 0
    return
  fi
  find "${LOCAL_DATASET}/${split}" -mindepth 2 -maxdepth 2 \
    -type f -name data.mdb | wc -l
}

status() {
  printf 'Remote source: %s:%s\n' "${REMOTE_HOST}" "${REMOTE_DATASET}"
  remote "python -c \"import json; d=json.load(open('${REMOTE_INDEX}/summary.json')); print(d['dataset'], d['splits'])\""
  printf 'Local dataset: %s\n' "${LOCAL_DATASET}"
  printf 'Local completed shards: train=%s val=%s test=%s\n' \
    "$(local_shards train)" "$(local_shards val)" "$(local_shards test)"
  if [[ -d "${LOCAL_DATASET}" ]]; then
    du -sh "${LOCAL_DATASET}"
  fi
  df -P -B1 /dataset | tail -1
}

sync_dataset() {
  if [[ ! -x "${SSH_WRAPPER}" ]]; then
    echo "SSH wrapper is not executable: ${SSH_WRAPPER}" >&2
    exit 1
  fi
  remote \
    "test -f '${REMOTE_INDEX}/summary.json' && test -f '${REMOTE_DATASET}/summary.json' && test -f '${REMOTE_MANIFEST}/summary.json'"

  mkdir -p \
    "${LOCAL_DATASET}" \
    "${LOCAL_INDEX}" \
    "${LOCAL_MANIFEST}" \
    "${TRANSFER_DIR}"

  sync_path "${REMOTE_INDEX}/" "${LOCAL_INDEX}/"
  sync_path "${REMOTE_MANIFEST}/" "${LOCAL_MANIFEST}/"
  build_file_lists

  # Make one complete shard per split available for loader validation first.
  sync_file_list "${SEED_LIST}"
  sync_file_list "${FILE_LIST}"
  "${PYTHON}" "${VERIFY}" --root "${LOCAL_DATASET}"
}

[[ $# -eq 1 ]] || usage
case "$1" in
  status) status ;;
  sync) sync_dataset ;;
  verify) "${PYTHON}" "${VERIFY}" --root "${LOCAL_DATASET}" ;;
  *) usage ;;
esac
