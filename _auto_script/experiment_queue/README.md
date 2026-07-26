# SC26 durable experiment queue

This queue schedules existing experiment launchers on the two SC26 GPU hosts.
It uses atomic NFS directories for job claims and GPU leases. It never deletes
job history, overwrites experiment outputs, kills a running job, or
automatically reclaims a stale lease.

## Manifest

Keep user-authored manifests with the dated experiment:

```yaml
jobs:
  - id: nabla-nte128-muon-shift-v2
    description: NTE-128/3 Muon shift-only baseline
    launcher: /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/09_nabladft_shift_only_baselines_2gpu.sh
    args: [full, nte128, "{gpus}"]
    gpu_count: 2
    allowed_hosts: [any]
    priority: 0
    input_files:
      - /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/nte128e3_muon_shift_only_nabladft.yaml
```

`{gpus}` is replaced with a comma-separated allocation. A full job should be
enqueued from a clean committed tree. `--allow-dirty` pins the complete current
Git fingerprint, but the job blocks if that fingerprint changes before launch.

## Commands

```bash
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
QUEUE=/dataset/seongsu/shared-home/workspace/project/_auto_script/experiment_queue/sc26_queue.py

${PY} ${QUEUE} enqueue /absolute/path/to/queue.yaml
${PY} ${QUEUE} list
${PY} ${QUEUE} run-once --dry-run --host-label server-1
${PY} ${QUEUE} show JOB_ID
${PY} ${QUEUE} retry JOB_ID
${PY} ${QUEUE} retry JOB_ID --refresh-source
${PY} ${QUEUE} cancel JOB_ID
${PY} ${QUEUE} doctor
```

Run two persistent worker slots on each host initially. Each slot executes one
job at a time; shared claims and GPU leases prevent overlap:

```bash
SC26_QUEUE_HOST_LABEL=server-1 SC26_QUEUE_WORKER_SLOT=0 /dataset/seongsu/shared-home/workspace/project/_auto_script/experiment_queue/run_queue_worker.sh start
SC26_QUEUE_HOST_LABEL=server-1 SC26_QUEUE_WORKER_SLOT=1 /dataset/seongsu/shared-home/workspace/project/_auto_script/experiment_queue/run_queue_worker.sh start

SC26_QUEUE_HOST_LABEL=server-2 SC26_QUEUE_WORKER_SLOT=0 /dataset/seongsu/shared-home/workspace/project/_auto_script/experiment_queue/run_queue_worker.sh start
SC26_QUEUE_HOST_LABEL=server-2 SC26_QUEUE_WORKER_SLOT=1 /dataset/seongsu/shared-home/workspace/project/_auto_script/experiment_queue/run_queue_worker.sh start
```

The worker launcher deliberately has no `stop` command. Inspect `doctor`, the
claim PID, GPU ownership, and process state before stopping a worker or
training process.
