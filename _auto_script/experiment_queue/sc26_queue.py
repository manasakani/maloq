#!/usr/bin/env python3
"""Durable, NFS-safe experiment queue for the two SC26 GPU servers."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


CANONICAL_PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
DEFAULT_QUEUE_ROOT = CANONICAL_PROJECT_ROOT / "outputs" / "experiment-queue"
JOB_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,95}$")
RUNNABLE_STATES = {"queued", "waiting_gpu", "waiting_storage"}
RETRYABLE_STATES = {"blocked", "cancelled", "failed", "interrupted"}
LANE_STATES = (
    "queued",
    "waiting_gpu",
    "waiting_storage",
    "validating",
    "running",
    "blocked",
    "interrupted",
    "cancelled",
    "complete",
    "failed",
)
MAX_GPU_MEMORY_USED_MIB = 1024
MAX_GPU_UTILIZATION_PERCENT = 5
DATASET_POLICY_LIMIT_BYTES = 40_000_000_000_000
DATASET_POLICY_ATTENTION_PERCENT = 80.0
DEFAULT_POLL_SECONDS = 30
DEFAULT_HEARTBEAT_SECONDS = 15
SOURCE_STATUS_LIMIT = 2_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def append_event(job_dir: Path, event: str, **details: Any) -> None:
    record = {"at": utc_now(), "event": event, **details}
    with (job_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_git(project_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(project_root: Path) -> dict[str, Any]:
    commit = run_git(project_root, "rev-parse", "HEAD").decode().strip()
    status = run_git(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if len(status) > SOURCE_STATUS_LIMIT:
        raise ValueError("Git status is unexpectedly large; refuse to enqueue.")
    tracked_diff = run_git(project_root, "diff", "--binary", "HEAD")
    untracked_raw = run_git(
        project_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked: list[dict[str, Any]] = []
    for raw_path in untracked_raw.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode()
        path = (project_root / relative).resolve()
        if project_root.resolve() not in path.parents:
            raise ValueError(f"Untracked path escapes project root: {relative}")
        if not path.is_file():
            continue
        untracked.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    fingerprint_payload = json.dumps(
        {
            "commit": commit,
            "status_sha256": sha256_bytes(status),
            "tracked_diff_sha256": sha256_bytes(tracked_diff),
            "untracked": untracked,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.decode(errors="replace"),
        "status_sha256": sha256_bytes(status),
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "untracked": untracked,
        "fingerprint": sha256_bytes(fingerprint_payload),
        "captured_at": utc_now(),
    }


def input_snapshots(paths: Sequence[str], project_root: Path) -> list[dict[str, Any]]:
    snapshots = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Input file is missing: {path}")
        snapshots.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return snapshots


def load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - project env has PyYAML
            raise RuntimeError("PyYAML is required for YAML queue manifests.") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Queue manifest must contain an object.")
    return value


@dataclass(frozen=True)
class QueuePaths:
    project_root: Path
    queue_root: Path

    @property
    def jobs(self) -> Path:
        return self.queue_root / "jobs"

    @property
    def claims(self) -> Path:
        return self.queue_root / "claims"

    @property
    def locks(self) -> Path:
        return self.queue_root / "locks"

    @property
    def workers(self) -> Path:
        return self.queue_root / "workers"

    def ensure(self) -> None:
        for path in (self.jobs, self.claims, self.locks, self.workers):
            path.mkdir(parents=True, exist_ok=True)


def normalize_launcher(raw_launcher: str, project_root: Path) -> Path:
    launcher = Path(raw_launcher)
    if not launcher.is_absolute():
        launcher = project_root / launcher
    launcher = launcher.resolve()
    experiment_root = (project_root / "_my_script" / "experiment").resolve()
    if launcher != experiment_root and experiment_root not in launcher.parents:
        raise ValueError(
            "Launcher must be below the canonical _my_script/experiment directory."
        )
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError(f"Launcher is missing or not executable: {launcher}")
    return launcher


def validate_job_spec(raw: dict[str, Any], project_root: Path) -> dict[str, Any]:
    job_id = str(raw.get("id", ""))
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError(
            f"Invalid job id {job_id!r}; use 3-96 lowercase letters, numbers, "
            "dots, underscores, or hyphens."
        )
    launcher = normalize_launcher(str(raw.get("launcher", "")), project_root)
    args = raw.get("args")
    if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
        raise ValueError(f"{job_id}: args must be a list of strings.")
    gpu_count = int(raw.get("gpu_count", 2))
    if gpu_count < 1 or gpu_count > 8:
        raise ValueError(f"{job_id}: gpu_count must be between 1 and 8.")
    if sum(value.count("{gpus}") for value in args) != 1:
        raise ValueError(f"{job_id}: args must contain exactly one '{{gpus}}'.")
    allowed_hosts = raw.get("allowed_hosts", ["any"])
    if (
        not isinstance(allowed_hosts, list)
        or not allowed_hosts
        or not all(isinstance(value, str) and value for value in allowed_hosts)
    ):
        raise ValueError(f"{job_id}: allowed_hosts must be a non-empty string list.")
    input_files = raw.get("input_files", [])
    if not isinstance(input_files, list) or not all(
        isinstance(value, str) for value in input_files
    ):
        raise ValueError(f"{job_id}: input_files must be a string list.")
    environment = raw.get("environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str)
        and key
        and isinstance(value, (str, int, float, bool))
        for key, value in environment.items()
    ):
        raise ValueError(f"{job_id}: environment must contain scalar values.")
    priority = int(raw.get("priority", 0))
    if priority < -1000 or priority > 1000:
        raise ValueError(f"{job_id}: priority must be between -1000 and 1000.")
    return {
        "id": job_id,
        "launcher": str(launcher),
        "args": args,
        "gpu_count": gpu_count,
        "allowed_hosts": sorted(set(allowed_hosts)),
        "priority": priority,
        "input_files": input_files,
        "environment": {key: str(value) for key, value in environment.items()},
        "description": str(raw.get("description", "")).strip(),
    }


def initial_state(job_id: str) -> dict[str, Any]:
    now = utc_now()
    return {
        "id": job_id,
        "status": "queued",
        "attempts": 0,
        "created_at": now,
        "updated_at": now,
        "reason": None,
        "exit_code": None,
        "worker": None,
        "host": None,
        "gpus": [],
        "pid": None,
    }


def update_state(job_dir: Path, **changes: Any) -> dict[str, Any]:
    state = read_json(job_dir / "state.json")
    state.update(changes)
    state["updated_at"] = utc_now()
    atomic_write_json(job_dir / "state.json", state)
    return state


def enqueue_manifest(
    paths: QueuePaths,
    manifest_path: Path,
    *,
    allow_dirty: bool,
) -> list[str]:
    paths.ensure()
    manifest = load_manifest(manifest_path)
    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Queue manifest must contain a non-empty jobs list.")
    jobs = [validate_job_spec(raw, paths.project_root) for raw in raw_jobs]
    job_ids = [job["id"] for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Queue manifest contains duplicate job ids.")
    existing = [job_id for job_id in job_ids if (paths.jobs / job_id).exists()]
    if existing:
        raise ValueError(f"Job ids already exist: {', '.join(existing)}")

    source = source_snapshot(paths.project_root)
    if source["dirty"] and not allow_dirty:
        raise ValueError(
            "The repository is dirty. Commit the experiment or pass --allow-dirty "
            "to record and strictly pin the current worktree fingerprint."
        )
    created: list[str] = []
    for job in jobs:
        job_id = job["id"]
        input_files = job.pop("input_files")
        request = {
            **job,
            "enqueued_at": utc_now(),
            "manifest": str(manifest_path.resolve()),
            "source": source,
            "inputs": input_snapshots(input_files, paths.project_root),
        }
        temporary = paths.jobs / f".{job_id}.{os.getpid()}.tmp"
        temporary.mkdir()
        atomic_write_json(temporary / "request.json", request)
        atomic_write_json(temporary / "state.json", initial_state(job_id))
        append_event(
            temporary,
            "enqueued",
            source_fingerprint=source["fingerprint"],
            dirty=source["dirty"],
        )
        os.rename(temporary, paths.jobs / job_id)
        created.append(job_id)
    return created


def list_jobs(paths: QueuePaths) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    paths.ensure()
    jobs = []
    for job_dir in sorted(paths.jobs.iterdir()):
        if not job_dir.is_dir() or job_dir.name.startswith("."):
            continue
        try:
            request = read_json(job_dir / "request.json")
            state = read_json(job_dir / "state.json")
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        jobs.append((request, state))
    return jobs


def eligible_for_host(
    request: dict[str, Any],
    hostname: str,
    host_label: str | None,
) -> bool:
    allowed = set(request["allowed_hosts"])
    return bool({"any", hostname, host_label} & allowed)


def job_sort_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, str]:
    request, state = item
    return (-int(request["priority"]), str(state["created_at"]))


def claim_job(paths: QueuePaths, job_id: str, worker: str, hostname: str) -> Path | None:
    claim_dir = paths.claims / job_id
    try:
        claim_dir.mkdir()
    except FileExistsError:
        return None
    atomic_write_json(
        claim_dir / "lease.json",
        {
            "job_id": job_id,
            "worker": worker,
            "hostname": hostname,
            "worker_pid": os.getpid(),
            "child_pid": None,
            "claimed_at": utc_now(),
            "heartbeat_at": utc_now(),
        },
    )
    return claim_dir


def update_lease(claim_dir: Path, **changes: Any) -> None:
    lease = read_json(claim_dir / "lease.json")
    lease.update(changes)
    lease["heartbeat_at"] = utc_now()
    atomic_write_json(claim_dir / "lease.json", lease)


def release_simple_lease(lease_dir: Path) -> None:
    lease_file = lease_dir / "lease.json"
    if lease_file.exists():
        lease_file.unlink()
    lease_dir.rmdir()


def gpu_inventory() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    inventory = []
    for line in result.stdout.splitlines():
        values = [int(value.strip()) for value in line.split(",")]
        if len(values) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line!r}")
        inventory.append(
            {
                "index": values[0],
                "memory_used_mib": values[1],
                "utilization_percent": values[2],
            }
        )
    return inventory


def dataset_storage_policy() -> dict[str, Any]:
    result = subprocess.run(
        ["df", "-B1", "--output=used", "/dataset"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 2:
        raise RuntimeError(f"Unexpected df output for /dataset: {result.stdout!r}")
    used_bytes = int(rows[1])
    used_percent = used_bytes / DATASET_POLICY_LIMIT_BYTES * 100.0
    return {
        "used_bytes": used_bytes,
        "policy_limit_bytes": DATASET_POLICY_LIMIT_BYTES,
        "policy_remaining_bytes": DATASET_POLICY_LIMIT_BYTES - used_bytes,
        "policy_used_percent": used_percent,
        "attention": used_percent >= DATASET_POLICY_ATTENTION_PERCENT,
        "exceeded": used_bytes >= DATASET_POLICY_LIMIT_BYTES,
    }


def acquire_gpu_leases(
    paths: QueuePaths,
    hostname: str,
    job_id: str,
    gpu_count: int,
    inventory_provider: Callable[[], list[dict[str, int]]] = gpu_inventory,
) -> tuple[list[int], list[Path]] | None:
    inventory = inventory_provider()
    idle = [
        gpu["index"]
        for gpu in inventory
        if gpu["memory_used_mib"] <= MAX_GPU_MEMORY_USED_MIB
        and gpu["utilization_percent"] <= MAX_GPU_UTILIZATION_PERCENT
    ]
    for candidate in itertools.combinations(sorted(idle), gpu_count):
        acquired: list[Path] = []
        for gpu_index in candidate:
            lock_dir = paths.locks / f"{hostname}-gpu-{gpu_index}"
            try:
                lock_dir.mkdir()
            except FileExistsError:
                break
            atomic_write_json(
                lock_dir / "lease.json",
                {
                    "job_id": job_id,
                    "hostname": hostname,
                    "gpu": gpu_index,
                    "acquired_at": utc_now(),
                },
            )
            acquired.append(lock_dir)
        if len(acquired) == gpu_count:
            return list(candidate), acquired
        for lock_dir in reversed(acquired):
            release_simple_lease(lock_dir)
    return None


def choose_master_port(
    paths: QueuePaths,
    hostname: str,
    job_id: str,
) -> tuple[int, Path]:
    start = 20000 + int(hashlib.sha256(job_id.encode()).hexdigest()[:6], 16) % 20000
    for offset in range(20000):
        port = 20000 + (start - 20000 + offset) % 20000
        lock_dir = paths.locks / f"{hostname}-port-{port}"
        try:
            lock_dir.mkdir()
        except FileExistsError:
            continue
        with socket.socket() as probe:
            try:
                # PyTorch TCPStore listens on every local IPv4 address. Probe
                # the same wildcard address so an established connection that
                # already uses this port on a non-loopback interface is not
                # mistaken for an available rendezvous port.
                probe.bind(("0.0.0.0", port))
            except OSError:
                lock_dir.rmdir()
                continue
        atomic_write_json(
            lock_dir / "lease.json",
            {
                "job_id": job_id,
                "hostname": hostname,
                "port": port,
                "acquired_at": utc_now(),
            },
        )
        return port, lock_dir
    raise RuntimeError("Could not allocate a local master port.")


def materialize_command(request: dict[str, Any], gpus: Sequence[int]) -> list[str]:
    gpu_csv = ",".join(str(gpu) for gpu in gpus)
    return [
        request["launcher"],
        *(argument.replace("{gpus}", gpu_csv) for argument in request["args"]),
    ]


def verify_inputs(request: dict[str, Any]) -> None:
    for record in request["inputs"]:
        path = Path(record["path"])
        if not path.is_file():
            raise ValueError(f"Input disappeared: {path}")
        if path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Input changed after enqueue: {path}")


def run_claimed_job(
    paths: QueuePaths,
    request: dict[str, Any],
    state: dict[str, Any],
    claim_dir: Path,
    *,
    hostname: str,
    host_label: str | None,
    worker: str,
    inventory_provider: Callable[[], list[dict[str, int]]] = gpu_inventory,
    storage_provider: Callable[[], dict[str, Any]] = dataset_storage_policy,
) -> str:
    job_id = request["id"]
    job_dir = paths.jobs / job_id
    storage = storage_provider()
    if storage["exceeded"]:
        if state["status"] != "waiting_storage":
            append_event(job_dir, "waiting_storage", storage=storage)
        update_state(
            job_dir,
            status="waiting_storage",
            reason="shared_dataset_policy_limit_exceeded",
            worker=None,
            host=None,
        )
        release_simple_lease(claim_dir)
        return "waiting_storage"
    if storage["attention"]:
        append_event(job_dir, "storage_attention", storage=storage)

    current_source = source_snapshot(paths.project_root)
    if current_source["fingerprint"] != request["source"]["fingerprint"]:
        update_state(
            job_dir,
            status="blocked",
            reason="source_changed_after_enqueue",
            worker=None,
            host=None,
        )
        append_event(
            job_dir,
            "blocked",
            reason="source_changed_after_enqueue",
            expected=request["source"]["fingerprint"],
            actual=current_source["fingerprint"],
        )
        release_simple_lease(claim_dir)
        return "blocked"
    try:
        verify_inputs(request)
    except ValueError as error:
        update_state(
            job_dir,
            status="blocked",
            reason=str(error),
            worker=None,
            host=None,
        )
        append_event(job_dir, "blocked", reason=str(error))
        release_simple_lease(claim_dir)
        return "blocked"

    allocation = acquire_gpu_leases(
        paths,
        hostname,
        job_id,
        int(request["gpu_count"]),
        inventory_provider,
    )
    if allocation is None:
        if state["status"] != "waiting_gpu":
            append_event(job_dir, "waiting_gpu", hostname=hostname)
        update_state(
            job_dir,
            status="waiting_gpu",
            reason="no_idle_gpu_allocation",
            worker=None,
            host=None,
        )
        release_simple_lease(claim_dir)
        return "waiting_gpu"

    gpus, gpu_locks = allocation
    port_lock: Path | None = None
    child: subprocess.Popen[bytes] | None = None
    normal_completion = False
    try:
        master_port, port_lock = choose_master_port(paths, hostname, job_id)
        attempt = int(state["attempts"]) + 1
        command = materialize_command(request, gpus)
        log_path = job_dir / f"attempt-{attempt:03d}.log"
        environment = os.environ.copy()
        environment.update(request["environment"])
        environment.update(
            {
                "EXPECTED_HOST": hostname,
                "MASTER_PORT": str(master_port),
                "SC26_QUEUE_JOB_ID": job_id,
                "SC26_QUEUE_ATTEMPT": str(attempt),
                "SC26_QUEUE_HOST_LABEL": host_label or hostname,
                "SC26_QUEUE_GPUS": ",".join(str(gpu) for gpu in gpus),
            }
        )
        append_event(
            job_dir,
            "starting",
            attempt=attempt,
            worker=worker,
            hostname=hostname,
            gpus=gpus,
            master_port=master_port,
            command=command,
        )
        update_state(
            job_dir,
            status="running",
            attempts=attempt,
            reason=None,
            exit_code=None,
            worker=worker,
            host=hostname,
            gpus=gpus,
            pid=None,
            log=str(log_path),
            started_at=utc_now(),
        )
        with log_path.open("ab", buffering=0) as log_handle:
            header = (
                f"[sc26-queue] at={utc_now()} worker={worker} host={hostname} "
                f"gpus={gpus} port={master_port}\n"
                f"[sc26-queue] command={json.dumps(command)}\n"
            ).encode()
            log_handle.write(header)
            child = subprocess.Popen(
                command,
                cwd=paths.project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            update_lease(claim_dir, child_pid=child.pid)
            update_state(job_dir, pid=child.pid)
            while True:
                try:
                    child.wait(timeout=DEFAULT_HEARTBEAT_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    update_lease(claim_dir, child_pid=child.pid)
            exit_code = int(child.returncode)
        normal_completion = True
        final_status = "complete" if exit_code == 0 else "failed"
        update_state(
            job_dir,
            status=final_status,
            reason=None if exit_code == 0 else "launcher_exit_nonzero",
            exit_code=exit_code,
            worker=None,
            host=hostname,
            gpus=gpus,
            pid=None,
            finished_at=utc_now(),
        )
        append_event(
            job_dir,
            final_status,
            attempt=attempt,
            exit_code=exit_code,
            hostname=hostname,
            gpus=gpus,
        )
        return final_status
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            update_state(
                job_dir,
                status="interrupted",
                reason="worker_interrupted_child_may_still_be_running",
                worker=worker,
                host=hostname,
                gpus=gpus,
                pid=child.pid,
            )
            append_event(
                job_dir,
                "interrupted",
                child_pid=child.pid,
                action="claim_and_gpu_locks_retained",
            )
            raise
        normal_completion = True
        raise
    except Exception as error:
        if child is not None and child.poll() is None:
            try:
                update_state(
                    job_dir,
                    status="interrupted",
                    reason=(
                        "queue_worker_error_child_may_still_be_running:"
                        f"{type(error).__name__}:{error}"
                    ),
                    worker=worker,
                    host=hostname,
                    gpus=gpus,
                    pid=child.pid,
                )
                append_event(
                    job_dir,
                    "interrupted",
                    child_pid=child.pid,
                    reason=f"{type(error).__name__}:{error}",
                    action="claim_and_gpu_locks_retained",
                )
            except Exception:
                pass
            return "interrupted"
        normal_completion = True
        update_state(
            job_dir,
            status="failed",
            reason=f"queue_worker_error:{type(error).__name__}:{error}",
            worker=None,
            host=hostname,
            gpus=gpus,
            pid=None,
        )
        append_event(
            job_dir,
            "failed",
            reason=f"queue_worker_error:{type(error).__name__}:{error}",
        )
        return "failed"
    finally:
        if normal_completion:
            if port_lock is not None and port_lock.exists():
                release_simple_lease(port_lock)
            for lock_dir in reversed(gpu_locks):
                if lock_dir.exists():
                    release_simple_lease(lock_dir)
            if claim_dir.exists():
                release_simple_lease(claim_dir)


def run_one(
    paths: QueuePaths,
    *,
    hostname: str,
    host_label: str | None,
    worker: str,
    dry_run: bool,
    inventory_provider: Callable[[], list[dict[str, int]]] = gpu_inventory,
    storage_provider: Callable[[], dict[str, Any]] = dataset_storage_policy,
) -> str | None:
    candidates = [
        item
        for item in list_jobs(paths)
        if item[1].get("status") in RUNNABLE_STATES
        and eligible_for_host(item[0], hostname, host_label)
    ]
    for request, state in sorted(candidates, key=job_sort_key):
        if dry_run:
            print(
                json.dumps(
                    {
                        "id": request["id"],
                        "status": state["status"],
                        "priority": request["priority"],
                        "gpu_count": request["gpu_count"],
                        "allowed_hosts": request["allowed_hosts"],
                        "command_template": [
                            request["launcher"],
                            *request["args"],
                        ],
                    },
                    indent=2,
                )
            )
            return "dry-run"
        claim_dir = claim_job(paths, request["id"], worker, hostname)
        if claim_dir is None:
            continue
        return run_claimed_job(
            paths,
            request,
            state,
            claim_dir,
            hostname=hostname,
            host_label=host_label,
            worker=worker,
            inventory_provider=inventory_provider,
            storage_provider=storage_provider,
        )
    return None


def retry_job(
    paths: QueuePaths,
    job_id: str,
    *,
    refresh_source: bool,
    allow_dirty: bool,
) -> None:
    job_dir = paths.jobs / job_id
    if not job_dir.is_dir():
        raise ValueError(f"Unknown job: {job_id}")
    if (paths.claims / job_id).exists():
        raise ValueError(f"Job {job_id} still has an active claim; inspect it first.")
    state = read_json(job_dir / "state.json")
    if state["status"] not in RETRYABLE_STATES:
        raise ValueError(
            f"Job {job_id} is {state['status']}; retry accepts "
            f"{sorted(RETRYABLE_STATES)}."
        )
    request = read_json(job_dir / "request.json")
    if refresh_source:
        source = source_snapshot(paths.project_root)
        if source["dirty"] and not allow_dirty:
            raise ValueError("Refuse to refresh to a dirty source without --allow-dirty.")
        request["source"] = source
        request["inputs"] = input_snapshots(
            [record["path"] for record in request["inputs"]],
            paths.project_root,
        )
        request["source_refreshed_at"] = utc_now()
        atomic_write_json(job_dir / "request.json", request)
    update_state(
        job_dir,
        status="queued",
        reason=None,
        exit_code=None,
        worker=None,
        host=None,
        gpus=[],
        pid=None,
    )
    append_event(
        job_dir,
        "retried",
        refresh_source=refresh_source,
        source_fingerprint=request["source"]["fingerprint"],
    )


def cancel_job(paths: QueuePaths, job_id: str) -> None:
    job_dir = paths.jobs / job_id
    if not job_dir.is_dir():
        raise ValueError(f"Unknown job: {job_id}")
    state = read_json(job_dir / "state.json")
    if state["status"] in {"running", "interrupted"} or (paths.claims / job_id).exists():
        raise ValueError(
            "Refuse to cancel a claimed/running job because queue cancellation "
            "must not kill training. Inspect the PID and GPU first."
        )
    if state["status"] == "complete":
        raise ValueError("A completed job cannot be cancelled.")
    update_state(job_dir, status="cancelled", reason="cancelled_before_start")
    append_event(job_dir, "cancelled", reason="cancelled_before_start")


def format_jobs(paths: QueuePaths) -> str:
    rows = []
    for request, state in sorted(list_jobs(paths), key=job_sort_key):
        rows.append(
            (
                request["id"],
                state["status"],
                str(request["priority"]),
                str(request["gpu_count"]),
                str(state["attempts"]),
                str(state.get("host") or "-"),
                ",".join(str(value) for value in state.get("gpus", [])) or "-",
                str(state.get("reason") or "-"),
            )
        )
    headers = ("ID", "STATUS", "PRI", "GPU", "TRY", "HOST", "DEVICES", "REASON")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def doctor(paths: QueuePaths) -> dict[str, Any]:
    paths.ensure()
    now = datetime.now(timezone.utc)
    claims = []
    for claim_dir in sorted(paths.claims.iterdir()):
        if not claim_dir.is_dir():
            continue
        try:
            lease = read_json(claim_dir / "lease.json")
            heartbeat = datetime.fromisoformat(lease["heartbeat_at"])
            age = (now - heartbeat).total_seconds()
            claims.append({**lease, "heartbeat_age_seconds": round(age, 1)})
        except Exception as error:
            claims.append({"job_id": claim_dir.name, "error": str(error)})
    gpu_locks = []
    for lock_dir in sorted(paths.locks.iterdir()):
        if lock_dir.is_dir() and "-gpu-" in lock_dir.name:
            try:
                gpu_locks.append(read_json(lock_dir / "lease.json"))
            except Exception as error:
                gpu_locks.append({"lock": lock_dir.name, "error": str(error)})
    state_counts = {state: 0 for state in LANE_STATES}
    invalid_states = []
    for request, state in list_jobs(paths):
        status = state.get("status")
        if status in state_counts:
            state_counts[status] += 1
        else:
            invalid_states.append({"id": request["id"], "status": status})
    return {
        "generated_at": utc_now(),
        "queue_root": str(paths.queue_root),
        "state_counts": {key: value for key, value in state_counts.items() if value},
        "claims": claims,
        "gpu_locks": gpu_locks,
        "invalid_states": invalid_states,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=CANONICAL_PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--queue-root",
        type=Path,
        default=DEFAULT_QUEUE_ROOT,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue = subparsers.add_parser("enqueue")
    enqueue.add_argument("manifest", type=Path)
    enqueue.add_argument("--allow-dirty", action="store_true")

    subparsers.add_parser("list")

    show = subparsers.add_parser("show")
    show.add_argument("job_id")

    run_once = subparsers.add_parser("run-once")
    run_once.add_argument("--host-label")
    run_once.add_argument("--dry-run", action="store_true")

    worker = subparsers.add_parser("worker")
    worker.add_argument("--host-label")
    worker.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    worker.add_argument("--max-jobs", type=int, default=0)

    retry = subparsers.add_parser("retry")
    retry.add_argument("job_id")
    retry.add_argument("--refresh-source", action="store_true")
    retry.add_argument("--allow-dirty", action="store_true")

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("job_id")

    subparsers.add_parser("doctor")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    queue_root = args.queue_root
    if not queue_root.is_absolute():
        queue_root = project_root / queue_root
    paths = QueuePaths(project_root=project_root, queue_root=queue_root.resolve())
    paths.ensure()
    try:
        if args.command == "enqueue":
            created = enqueue_manifest(
                paths,
                args.manifest,
                allow_dirty=args.allow_dirty,
            )
            print("\n".join(created))
        elif args.command == "list":
            print(format_jobs(paths))
        elif args.command == "show":
            job_dir = paths.jobs / args.job_id
            payload = {
                "request": read_json(job_dir / "request.json"),
                "state": read_json(job_dir / "state.json"),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.command == "run-once":
            hostname = socket.gethostname()
            worker = f"{args.host_label or hostname}:{os.getpid()}"
            result = run_one(
                paths,
                hostname=hostname,
                host_label=args.host_label,
                worker=worker,
                dry_run=args.dry_run,
            )
            if result is None:
                print("No eligible job.")
        elif args.command == "worker":
            if args.poll_seconds < 5:
                raise ValueError("--poll-seconds must be at least 5.")
            hostname = socket.gethostname()
            worker = f"{args.host_label or hostname}:{os.getpid()}"
            completed = 0
            print(
                f"SC26 queue worker={worker} host={hostname} "
                f"queue={paths.queue_root}",
                flush=True,
            )
            while True:
                result = run_one(
                    paths,
                    hostname=hostname,
                    host_label=args.host_label,
                    worker=worker,
                    dry_run=False,
                )
                if result in {"complete", "failed"}:
                    completed += 1
                if args.max_jobs and completed >= args.max_jobs:
                    break
                if result is None or result in {"waiting_gpu", "waiting_storage"}:
                    time.sleep(args.poll_seconds)
        elif args.command == "retry":
            retry_job(
                paths,
                args.job_id,
                refresh_source=args.refresh_source,
                allow_dirty=args.allow_dirty,
            )
        elif args.command == "cancel":
            cancel_job(paths, args.job_id)
        elif args.command == "doctor":
            print(json.dumps(doctor(paths), indent=2, sort_keys=True))
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
