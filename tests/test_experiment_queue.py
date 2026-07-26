from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
MODULE_PATH = (
    PROJECT_ROOT / "_auto_script" / "experiment_queue" / "sc26_queue.py"
)


def load_queue_module():
    spec = importlib.util.spec_from_file_location("sc26_queue", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def initialize_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    experiment = project / "_my_script" / "experiment" / "2026-07-26"
    experiment.mkdir(parents=True)
    launcher = experiment / "launcher.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" > \"${QUEUE_TEST_RESULT}\"\n"
        "printf '%s\\n' \"${MASTER_PORT}\" >> \"${QUEUE_TEST_RESULT}\"\n"
    )
    launcher.chmod(0o755)
    config = experiment / "config.yaml"
    config.write_text("name: queue-test\n")
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(
        ["git", "-C", str(project), "config", "user.email", "queue@test.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "config", "user.name", "Queue Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(project), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(project), "commit", "-qm", "queue fixture"],
        check=True,
    )
    return project, launcher


def write_manifest(
    path: Path,
    launcher: Path,
    config: Path,
    result: Path,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "jobs": [
                    {
                        "id": "queue-test-job",
                        "launcher": str(launcher),
                        "args": ["full", "{gpus}"],
                        "gpu_count": 2,
                        "allowed_hosts": ["test-host"],
                        "input_files": [str(config)],
                        "environment": {"QUEUE_TEST_RESULT": str(result)},
                    }
                ]
            },
            sort_keys=False,
        )
    )


def fake_inventory():
    return [
        {"index": 0, "memory_used_mib": 0, "utilization_percent": 0},
        {"index": 1, "memory_used_mib": 0, "utilization_percent": 0},
        {"index": 2, "memory_used_mib": 70000, "utilization_percent": 100},
    ]


def healthy_storage():
    return {
        "used_bytes": 1,
        "policy_limit_bytes": 40_000_000_000_000,
        "policy_remaining_bytes": 39_999_999_999_999,
        "policy_used_percent": 0.0,
        "attention": False,
        "exceeded": False,
    }


def test_enqueue_dry_run_and_complete(tmp_path):
    module = load_queue_module()
    project, launcher = initialize_project(tmp_path)
    queue_root = tmp_path / "queue"
    paths = module.QueuePaths(project, queue_root)
    manifest = tmp_path / "queue.yaml"
    result = tmp_path / "result.txt"
    config = launcher.parent / "config.yaml"
    write_manifest(manifest, launcher, config, result)

    assert module.enqueue_manifest(paths, manifest, allow_dirty=False) == [
        "queue-test-job"
    ]
    request, state = module.list_jobs(paths)[0]
    assert state["status"] == "queued"
    assert request["source"]["dirty"] is False
    assert request["inputs"][0]["sha256"] == module.sha256_file(config)

    assert (
        module.run_one(
            paths,
            hostname="test-host",
            host_label=None,
            worker="test-worker",
            dry_run=True,
            inventory_provider=fake_inventory,
            storage_provider=healthy_storage,
        )
        == "dry-run"
    )
    assert module.list_jobs(paths)[0][1]["status"] == "queued"

    assert (
        module.run_one(
            paths,
            hostname="test-host",
            host_label=None,
            worker="test-worker",
            dry_run=False,
            inventory_provider=fake_inventory,
            storage_provider=healthy_storage,
        )
        == "complete"
    )
    lines = result.read_text().splitlines()
    assert lines[0] == "full 0,1"
    assert 20000 <= int(lines[1]) < 40000
    state = module.list_jobs(paths)[0][1]
    assert state["status"] == "complete"
    assert state["attempts"] == 1
    assert not list(paths.claims.iterdir())
    assert not list(paths.locks.iterdir())


def test_source_change_blocks_without_launch(tmp_path):
    module = load_queue_module()
    project, launcher = initialize_project(tmp_path)
    paths = module.QueuePaths(project, tmp_path / "queue")
    manifest = tmp_path / "queue.yaml"
    result = tmp_path / "result.txt"
    config = launcher.parent / "config.yaml"
    write_manifest(manifest, launcher, config, result)
    module.enqueue_manifest(paths, manifest, allow_dirty=False)

    config.write_text("name: changed\n")
    assert (
        module.run_one(
            paths,
            hostname="test-host",
            host_label=None,
            worker="test-worker",
            dry_run=False,
            inventory_provider=fake_inventory,
            storage_provider=healthy_storage,
        )
        == "blocked"
    )
    assert not result.exists()
    state = module.list_jobs(paths)[0][1]
    assert state["status"] == "blocked"
    assert state["reason"] == "source_changed_after_enqueue"


def test_unavailable_resources_remain_waiting(tmp_path):
    module = load_queue_module()
    project, launcher = initialize_project(tmp_path)
    paths = module.QueuePaths(project, tmp_path / "queue")
    manifest = tmp_path / "queue.yaml"
    result = tmp_path / "result.txt"
    write_manifest(manifest, launcher, launcher.parent / "config.yaml", result)
    module.enqueue_manifest(paths, manifest, allow_dirty=False)

    exceeded_storage = {
        **healthy_storage(),
        "policy_remaining_bytes": -1,
        "policy_used_percent": 100.1,
        "attention": True,
        "exceeded": True,
    }
    assert (
        module.run_one(
            paths,
            hostname="test-host",
            host_label=None,
            worker="test-worker",
            dry_run=False,
            inventory_provider=fake_inventory,
            storage_provider=lambda: exceeded_storage,
        )
        == "waiting_storage"
    )
    assert module.list_jobs(paths)[0][1]["status"] == "waiting_storage"
    assert not result.exists()

    def busy_inventory():
        return [
            {"index": 0, "memory_used_mib": 70000, "utilization_percent": 100},
            {"index": 1, "memory_used_mib": 70000, "utilization_percent": 100},
        ]
    assert (
        module.run_one(
            paths,
            hostname="test-host",
            host_label=None,
            worker="test-worker",
            dry_run=False,
            inventory_provider=busy_inventory,
            storage_provider=healthy_storage,
        )
        == "waiting_gpu"
    )
    assert module.list_jobs(paths)[0][1]["status"] == "waiting_gpu"
    assert not result.exists()


def test_dirty_enqueue_requires_explicit_flag(tmp_path):
    module = load_queue_module()
    project, launcher = initialize_project(tmp_path)
    paths = module.QueuePaths(project, tmp_path / "queue")
    manifest = tmp_path / "queue.yaml"
    result = tmp_path / "result.txt"
    config = launcher.parent / "config.yaml"
    write_manifest(manifest, launcher, config, result)
    config.write_text("name: dirty\n")

    with pytest.raises(ValueError, match="repository is dirty"):
        module.enqueue_manifest(paths, manifest, allow_dirty=False)

    assert module.enqueue_manifest(paths, manifest, allow_dirty=True) == [
        "queue-test-job"
    ]
    request = json.loads(
        (paths.jobs / "queue-test-job" / "request.json").read_text()
    )
    assert request["source"]["dirty"] is True


def test_cancel_refuses_claimed_job(tmp_path):
    module = load_queue_module()
    project, launcher = initialize_project(tmp_path)
    paths = module.QueuePaths(project, tmp_path / "queue")
    manifest = tmp_path / "queue.yaml"
    result = tmp_path / "result.txt"
    write_manifest(manifest, launcher, launcher.parent / "config.yaml", result)
    module.enqueue_manifest(paths, manifest, allow_dirty=False)
    claim = module.claim_job(
        paths,
        "queue-test-job",
        "test-worker",
        "test-host",
    )
    assert claim is not None

    with pytest.raises(ValueError, match="must not kill training"):
        module.cancel_job(paths, "queue-test-job")
