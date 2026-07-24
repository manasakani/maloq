#!/usr/bin/env python3
"""Collect one GPU server's read-only NVIDIA status as JSON."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


GPU_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "pstate",
    "compute_mode",
)
PROCESS_FIELDS = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
STORAGE_PATHS = ("/", "/dataset/seongsu/shared-home")
SHARED_DATASET_LIMIT_BYTES = 40_000_000_000_000


def _number(value: str, *, integer: bool = False) -> int | float | None:
    value = value.strip()
    if value in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        return int(float(value)) if integer else float(value)
    except ValueError:
        return None


def parse_gpu_csv(text: str) -> list[dict[str, object]]:
    """Parse nvidia-smi GPU CSV output."""
    gpus: list[dict[str, object]] = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != len(GPU_FIELDS):
            raise ValueError(
                f"Expected {len(GPU_FIELDS)} GPU fields, received {len(row)}: {row}"
            )
        values = dict(zip(GPU_FIELDS, (value.strip() for value in row), strict=True))
        memory_used = _number(values["memory.used"], integer=True)
        memory_total = _number(values["memory.total"], integer=True)
        gpus.append(
            {
                "index": _number(values["index"], integer=True),
                "name": values["name"],
                "uuid": values["uuid"],
                "pci_bus_id": values["pci.bus_id"],
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
                "utilization_percent": _number(
                    values["utilization.gpu"], integer=True
                ),
                "temperature_c": _number(values["temperature.gpu"], integer=True),
                "power_draw_w": _number(values["power.draw"]),
                "power_limit_w": _number(values["power.limit"]),
                "pstate": values["pstate"],
                "compute_mode": values["compute_mode"],
                "processes": [],
            }
        )
    return gpus


def parse_process_csv(text: str) -> list[dict[str, object]]:
    """Parse nvidia-smi compute-process CSV output."""
    processes: list[dict[str, object]] = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != len(PROCESS_FIELDS):
            continue
        values = dict(
            zip(PROCESS_FIELDS, (value.strip() for value in row), strict=True)
        )
        pid = _number(values["pid"], integer=True)
        if pid is None:
            continue
        processes.append(
            {
                "gpu_uuid": values["gpu_uuid"],
                "pid": pid,
                "process_name": Path(values["process_name"]).name,
                "memory_used_mib": _number(
                    values["used_gpu_memory"], integer=True
                ),
            }
        )
    return processes


def parse_storage_df(text: str) -> list[dict[str, object]]:
    """Parse POSIX-style ``df -PT -B1`` output and remove duplicate mounts."""
    storage: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for line in text.splitlines():
        if not line.strip() or line.startswith("Filesystem"):
            continue
        parts = line.split(maxsplit=6)
        if len(parts) != 7:
            continue
        filesystem, filesystem_type, total, used, available, capacity, mountpoint = (
            parts
        )
        key = (filesystem, mountpoint)
        if key in seen:
            continue
        seen.add(key)
        total_bytes = _number(total, integer=True)
        used_bytes = _number(used, integer=True)
        available_bytes = _number(available, integer=True)
        used_percent = _number(capacity.rstrip("%"), integer=True)
        if total_bytes is None or used_bytes is None:
            continue
        if mountpoint == "/":
            label = "System disk"
            kind = "local"
        elif mountpoint == "/dataset" or mountpoint.startswith("/dataset/"):
            label = "Shared dataset"
            kind = "shared"
        else:
            label = mountpoint
            kind = "other"
        volume: dict[str, object] = {
            "label": label,
            "kind": kind,
            "filesystem": filesystem,
            "filesystem_type": filesystem_type,
            "mountpoint": mountpoint,
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "available_bytes": available_bytes,
            "used_percent": used_percent,
        }
        if kind == "shared":
            policy_remaining = SHARED_DATASET_LIMIT_BYTES - used_bytes
            policy_percent = used_bytes / SHARED_DATASET_LIMIT_BYTES * 100
            volume.update(
                {
                    "policy_limit_bytes": SHARED_DATASET_LIMIT_BYTES,
                    "policy_remaining_bytes": max(0, policy_remaining),
                    "policy_used_percent": policy_percent,
                    "policy_exceeded": policy_remaining < 0,
                }
            )
        storage.append(volume)
    return storage


def collect_storage() -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "/usr/bin/df",
                "-P",
                "-T",
                "-B1",
                "--",
                *STORAGE_PATHS,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_storage_df(result.stdout)


def process_metadata(pids: list[int]) -> dict[int, dict[str, str]]:
    if not pids:
        return {}
    command = [
        "/usr/bin/ps",
        "-p",
        ",".join(str(pid) for pid in sorted(set(pids))),
        "-o",
        "pid=,user=,etime=,comm=",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    metadata: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        pid_text, user, elapsed, command_name = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        metadata[pid] = {
            "user": user,
            "elapsed": elapsed,
            "command": command_name,
        }
    return metadata


def collect(server_id: str, label: str) -> dict[str, object]:
    gpu_query = ",".join(GPU_FIELDS)
    gpu_result = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            f"--query-gpu={gpu_query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    gpus = parse_gpu_csv(gpu_result.stdout)

    process_query = ",".join(PROCESS_FIELDS)
    process_result = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            f"--query-compute-apps={process_query}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    processes = parse_process_csv(process_result.stdout)
    metadata = process_metadata(
        [int(process["pid"]) for process in processes if process["pid"] is not None]
    )
    gpu_by_uuid = {str(gpu["uuid"]): gpu for gpu in gpus}
    for process in processes:
        pid = int(process["pid"])
        process.update(metadata.get(pid, {}))
        gpu = gpu_by_uuid.get(str(process["gpu_uuid"]))
        if gpu is not None:
            gpu["processes"].append(
                {key: value for key, value in process.items() if key != "gpu_uuid"}
            )

    uptime_seconds = None
    try:
        uptime_seconds = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        pass

    return {
        "id": server_id,
        "label": label,
        "hostname": socket.gethostname(),
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": uptime_seconds,
        "storage": collect_storage(),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, dest="server_id")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.server_id, args.label),
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
