from __future__ import annotations

import importlib.util
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SITE_ROOT = (
    Path("/dataset/seongsu/shared-home/workspace/project")
    / "_auto_script/gpu_monitor_site"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_module("gpu_monitor_collector", SITE_ROOT / "collect_gpu_status.py")
server = _load_module("gpu_monitor_server", SITE_ROOT / "server.py")
history_store_module = _load_module(
    "gpu_monitor_history_store",
    SITE_ROOT / "history_store.py",
)


def test_parse_gpu_csv_preserves_h100_metrics():
    text = (
        "0, NVIDIA H100 80GB HBM3, GPU-abc, 00000000:19:00.0, "
        "2048, 81559, 73, 61, 412.5, 700.0, P0, Default\n"
    )
    gpu = collector.parse_gpu_csv(text)[0]
    assert gpu["index"] == 0
    assert gpu["memory_used_mib"] == 2048
    assert gpu["memory_total_mib"] == 81559
    assert gpu["utilization_percent"] == 73
    assert gpu["temperature_c"] == 61
    assert gpu["power_draw_w"] == 412.5
    assert gpu["processes"] == []


def test_parse_process_csv_uses_process_basename():
    text = "GPU-abc, 1234, /usr/bin/python3, 1932\n"
    process = collector.parse_process_csv(text)[0]
    assert process == {
        "gpu_uuid": "GPU-abc",
        "pid": 1234,
        "process_name": "python3",
        "memory_used_mib": 1932,
    }


def test_parse_storage_df_deduplicates_and_labels_mounts():
    text = (
        "Filesystem Type 1-blocks Used Available Capacity Mounted on\n"
        "/dev/nvme0n1p4 ext4 1000000 250000 750000 25% /\n"
        "host:/volume nfs 9000000 3000000 6000000 34% /dataset\n"
        "host:/volume nfs 9000000 3000000 6000000 34% /dataset\n"
    )
    storage = collector.parse_storage_df(text)
    assert [volume["label"] for volume in storage] == [
        "System disk",
        "Shared dataset",
    ]
    assert storage[0]["used_percent"] == 25
    assert storage[1]["kind"] == "shared"
    assert storage[1]["available_bytes"] == 6000000
    assert storage[1]["policy_limit_bytes"] == 40_000_000_000_000
    assert storage[1]["policy_remaining_bytes"] == 39_999_997_000_000
    assert abs(storage[1]["policy_used_percent"] - 0.0000075) < 1e-12
    assert storage[1]["policy_exceeded"] is False


def test_fleet_summary_separates_idle_busy_and_warning():
    servers = [
        {
            "online": True,
            "gpus": [
                {
                    "memory_used_mib": 0,
                    "memory_total_mib": 81559,
                    "utilization_percent": 0,
                    "temperature_c": 34,
                    "power_draw_w": 70,
                    "processes": [],
                },
                {
                    "memory_used_mib": 2000,
                    "memory_total_mib": 81559,
                    "utilization_percent": 65,
                    "temperature_c": 56,
                    "power_draw_w": 400,
                    "processes": [{"pid": 1, "user": "gpuuser"}],
                },
                {
                    "memory_used_mib": 0,
                    "memory_total_mib": 81559,
                    "utilization_percent": 0,
                    "temperature_c": 83,
                    "power_draw_w": 100,
                    "processes": [],
                },
            ],
        }
    ]
    summary = server.fleet_summary(servers)
    assert summary["servers_online"] == 1
    assert summary["gpus_total"] == 3
    assert summary["gpus_idle"] == 1
    assert summary["gpus_busy"] == 1
    assert summary["gpus_warning"] == 1
    assert summary["power_draw_w"] == 570
    assert summary["gpus_reporting"] == 3
    assert abs(summary["utilization_average_percent"] - 65 / 3) < 1e-12
    assert summary["utilization_peak_percent"] == 65
    assert summary["utilization_peak_gpu"] == "server / GPU ?"
    assert abs(summary["temperature_average_c"] - 173 / 3) < 1e-12
    assert summary["temperature_peak_c"] == 83
    assert summary["processes_active"] == 1
    assert summary["active_users"] == ["gpuuser"]


def test_history_store_records_gpu_process_and_storage(tmp_path):
    store = history_store_module.HistoryStore(
        tmp_path / "history.sqlite3",
        sample_interval_seconds=60,
    )
    fleet = {
        "servers_online": 1,
        "servers_total": 1,
        "gpus_total": 1,
        "gpus_idle": 0,
        "gpus_busy": 1,
        "gpus_warning": 0,
        "gpus_offline": 0,
        "memory_used_mib": 2048,
        "memory_total_mib": 81559,
        "power_draw_w": 350,
    }
    servers = [
        {
            "id": "server-1",
            "hostname": "gpu-1",
            "online": True,
            "cached": False,
            "latency_ms": 10,
            "uptime_seconds": 1000,
            "error": None,
            "storage": [
                {
                    "mountpoint": "/dataset",
                    "kind": "shared",
                    "filesystem_type": "nfs",
                    "total_bytes": 100_000_000,
                    "used_bytes": 20_000_000,
                    "available_bytes": 80_000_000,
                    "used_percent": 20,
                    "policy_limit_bytes": 40_000_000_000_000,
                    "policy_remaining_bytes": 39_999_980_000_000,
                    "policy_used_percent": 0.00005,
                    "policy_exceeded": False,
                }
            ],
            "gpus": [
                {
                    "index": 0,
                    "uuid": "GPU-abc",
                    "state": "busy",
                    "utilization_percent": 80,
                    "memory_used_mib": 2048,
                    "memory_total_mib": 81559,
                    "temperature_c": 55,
                    "power_draw_w": 350,
                    "power_limit_w": 700,
                    "pstate": "P0",
                    "processes": [
                        {
                            "pid": 123,
                            "user": "gpuuser",
                            "command": "python",
                            "process_name": "python3",
                            "elapsed": "01:02",
                            "memory_used_mib": 2048,
                        }
                    ],
                }
            ],
        }
    ]
    sampled_at = datetime.now(timezone.utc).isoformat()
    summary = store.record(sampled_at, servers, fleet)
    assert summary["sample_count"] == 1
    assert summary["tracking_since"] == sampled_at

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM gpu_samples").fetchone()[0] == 1
        process = connection.execute(
            "SELECT user, pid, command FROM process_samples"
        ).fetchone()
        assert process == ("gpuuser", 123, "python")
        policy = connection.execute(
            "SELECT policy_limit_bytes FROM storage_samples"
        ).fetchone()
        assert policy == (40_000_000_000_000,)

    history = store.recent_gpu_history(points_per_gpu=10)
    assert history[("server-1", 0)][0]["utilization_percent"] == 80
    report = store.gpu_history("server-1", 0, hours=1)
    assert report["raw_point_count"] == 1
    assert report["points"][0]["memory_used_mib"] == 2048
    assert report["processes"][0]["user"] == "gpuuser"

    fleet_report = store.aggregate_history(hours=1)
    assert fleet_report["scope"] == "fleet"
    assert fleet_report["raw_point_count"] == 1
    assert fleet_report["points"][0]["reporting_gpus"] == 1
    assert fleet_report["points"][0]["utilization_average_percent"] == 80
    assert abs(
        fleet_report["points"][0]["memory_utilization_percent"]
        - 2048 / 81559 * 100
    ) < 1e-12
    assert fleet_report["processes"][0]["server_id"] == "server-1"
    assert fleet_report["processes"][0]["gpu_index"] == 0

    server_report = store.aggregate_history(hours=1, server_id="server-1")
    assert server_report["scope"] == "server"
    assert server_report["server_id"] == "server-1"
    assert server_report["summary"]["latest"]["process_count"] == 1

    storage_report = store.storage_history("server-1", "/dataset", hours=1)
    assert storage_report["scope"] == "storage"
    assert storage_report["raw_point_count"] == 1
    assert storage_report["kind"] == "shared"
    assert storage_report["points"][0]["used_bytes"] == 20_000_000
    assert (
        storage_report["points"][0]["effective_remaining_bytes"]
        == 39_999_980_000_000
    )
    assert storage_report["summary"]["change_bytes"] == 0
