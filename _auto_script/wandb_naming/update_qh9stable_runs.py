#!/usr/bin/env python3
"""Apply the audited SC26 naming scheme to inactive QH9Stable W&B runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
WANDB_PROJECT = "kaist-korea/maloq-qh9"
EXPECTED_EPOCHS = 80


def spec(
    run_id: str,
    local_output: str,
    name: str,
    group: str,
    job_type: str,
    *tags: str,
    local_status: str,
    note: str = "",
) -> dict[str, object]:
    return {
        "id": run_id,
        "local_output": local_output,
        "name": name,
        "group": group,
        "job_type": job_type,
        "tags": (
            "dataset:qh9stable",
            "sc26-seongsu",
            *tags,
        ),
        "local_status": local_status,
        "note": note,
    }


def complete(
    run_id: str,
    local_output: str,
    name: str,
    target: str,
    model: str,
    batch_size: int,
    gradient_accumulation: int,
) -> dict[str, object]:
    target_slug = "hdelta" if target == "hamiltonian" else "ddelta"
    return spec(
        run_id,
        local_output,
        name,
        f"qh9stable-{target_slug}-native",
        "full",
        "current",
        "status:complete",
        "scope:full",
        "objective:delta",
        f"target:{target}",
        f"model:{model}",
        "head:native",
        "optimizer:muon",
        f"batch-size:{batch_size}",
        f"grad-accum:{gradient_accumulation}",
        "effective-batch:32",
        "epochs:80",
        "seed:44",
        "version:v1",
        local_status="complete",
    )


def failed(
    run_id: str,
    local_output: str,
    name: str,
    target: str,
    model: str,
    batch_size: int,
    gradient_accumulation: int,
    cause: str,
    attempt: str,
) -> dict[str, object]:
    return spec(
        run_id,
        local_output,
        name,
        "qh9stable-failed",
        "failed-full",
        "auxiliary",
        "status:failed",
        "scope:full",
        "objective:delta",
        f"target:{target}",
        f"model:{model}",
        "head:native",
        "optimizer:muon",
        f"batch-size:{batch_size}",
        f"grad-accum:{gradient_accumulation}",
        f"effective-batch:{batch_size * gradient_accumulation}",
        f"cause:{cause}",
        f"attempt:{attempt}",
        "seed:44",
        local_status="failed",
        note=(
            "This run did not complete the configured 80 epochs; "
            f"audited local cause: {cause}."
        ),
    )


RUNS = (
    # Completed comparison runs.
    complete(
        "1es67blw",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-191207/density-maloq",
        "QH9Stable | DΔ | MALOQ | Native | V1",
        "density",
        "maloq",
        32,
        1,
    ),
    complete(
        "g0hgqsma",
        "outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-full-seed44-20260723-133856/density-maloq-nte",
        "QH9Stable | DΔ | NTE-64/2 | Native | V1",
        "density",
        "nte64e2",
        16,
        2,
    ),
    complete(
        "ylkky5bs",
        "outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-full-seed44-20260723-133856/hamiltonian-maloq",
        "QH9Stable | HΔ | MALOQ | Native | V1",
        "hamiltonian",
        "maloq",
        16,
        2,
    ),
    complete(
        "zaogg6si",
        "outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-full-seed44-20260723-133856/hamiltonian-maloq-nte",
        "QH9Stable | HΔ | NTE-64/2 | Native | V1",
        "hamiltonian",
        "nte64e2",
        16,
        2,
    ),
    # First six-lane attempt: externally terminated with exit code 143.
    failed(
        "yqmopd1v",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-165351/density-maloq-nte",
        "FAILED | QH9Stable DΔ | NTE-64/2 | SIGTERM-BS32-A1",
        "density",
        "nte64e2",
        32,
        1,
        "sigterm",
        "bs32-a1",
    ),
    failed(
        "ygnzdnle",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-165351/density-qhflow3",
        "FAILED | QH9Stable DΔ | QHFlow3 | SIGTERM-BS32-A1",
        "density",
        "qhf3",
        32,
        1,
        "sigterm",
        "bs32-a1",
    ),
    failed(
        "w14b0a3u",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-165351/density-maloq",
        "FAILED | QH9Stable DΔ | MALOQ | SIGTERM-BS32-A1",
        "density",
        "maloq",
        32,
        1,
        "sigterm",
        "bs32-a1",
    ),
    failed(
        "gdgzoiow",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-165351/hamiltonian-maloq-nte",
        "FAILED | QH9Stable HΔ | NTE-64/2 | SIGTERM-BS32-A1",
        "hamiltonian",
        "nte64e2",
        32,
        1,
        "sigterm",
        "bs32-a1",
    ),
    failed(
        "v5xua8gy",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-165351/hamiltonian-qhflow3",
        "FAILED | QH9Stable HΔ | QHFlow3 | SIGTERM-BS32-A1",
        "hamiltonian",
        "qhf3",
        32,
        1,
        "sigterm",
        "bs32-a1",
    ),
    failed(
        "o9zqj2dw",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-165351/hamiltonian-maloq",
        "FAILED | QH9Stable HΔ | MALOQ | SIGTERM-BS32-A1",
        "hamiltonian",
        "maloq",
        32,
        1,
        "sigterm",
        "bs32-a1",
    ),
    # Second batch-32 attempt: the four runs below failed with CUDA OOM.
    failed(
        "5cit7x4c",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-191207/density-maloq-nte",
        "FAILED | QH9Stable DΔ | NTE-64/2 | OOM-BS32-A2",
        "density",
        "nte64e2",
        32,
        1,
        "cuda-oom",
        "bs32-a2",
    ),
    failed(
        "961yxrlf",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-191207/hamiltonian-maloq-nte",
        "FAILED | QH9Stable HΔ | NTE-64/2 | OOM-BS32-A2",
        "hamiltonian",
        "nte64e2",
        32,
        1,
        "cuda-oom",
        "bs32-a2",
    ),
    failed(
        "f435yd3q",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-191207/hamiltonian-qhflow3",
        "FAILED | QH9Stable HΔ | QHFlow3 | OOM-BS32-A2",
        "hamiltonian",
        "qhf3",
        32,
        1,
        "cuda-oom",
        "bs32-a2",
    ),
    failed(
        "iuvge9gw",
        "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-20260722-191207/hamiltonian-maloq",
        "FAILED | QH9Stable HΔ | MALOQ | OOM-BS32-A2",
        "hamiltonian",
        "maloq",
        32,
        1,
        "cuda-oom",
        "bs32-a2",
    ),
    # OOM-recovery attempts that were externally terminated.
    failed(
        "12d9i5ky",
        "outputs/qh9stable-oom-recovery-four-lane-mb4-ga8-eb32-full-seed44-20260723-040732/density-maloq-nte",
        "FAILED | QH9Stable DΔ | NTE-64/2 | SIGTERM-MB4-GA8",
        "density",
        "nte64e2",
        4,
        8,
        "sigterm",
        "mb4-ga8",
    ),
    failed(
        "jrqx359z",
        "outputs/qh9stable-oom-recovery-four-lane-mb4-ga8-eb32-full-seed44-20260723-040732/hamiltonian-maloq",
        "FAILED | QH9Stable HΔ | MALOQ | SIGTERM-MB4-GA8",
        "hamiltonian",
        "maloq",
        4,
        8,
        "sigterm",
        "mb4-ga8",
    ),
    failed(
        "sk4kvugi",
        "outputs/qh9stable-oom-recovery-four-lane-mb4-ga8-eb32-full-seed44-20260723-040732/hamiltonian-maloq-nte",
        "FAILED | QH9Stable HΔ | NTE-64/2 | SIGTERM-MB4-GA8",
        "hamiltonian",
        "nte64e2",
        4,
        8,
        "sigterm",
        "mb4-ga8",
    ),
    failed(
        "xmxn4u2w",
        "outputs/qh9stable-oom-recovery-four-lane-mb4-ga8-eb32-full-seed44-20260723-040732/hamiltonian-qhflow3",
        "FAILED | QH9Stable HΔ | QHFlow3 | SIGTERM-MB4-GA8",
        "hamiltonian",
        "qhf3",
        4,
        8,
        "sigterm",
        "mb4-ga8",
    ),
    failed(
        "r7b75vtf",
        "outputs/qh9stable-oom-recovery-four-lane-mb4-ga2-eb8-full-seed44-20260723-041305/density-maloq-nte",
        "FAILED | QH9Stable DΔ | NTE-64/2 | SIGTERM-MB4-GA2",
        "density",
        "nte64e2",
        4,
        2,
        "sigterm",
        "mb4-ga2",
    ),
    failed(
        "hlncoar6",
        "outputs/qh9stable-oom-recovery-four-lane-mb4-ga2-eb8-full-seed44-20260723-041305/hamiltonian-maloq",
        "FAILED | QH9Stable HΔ | MALOQ | SIGTERM-MB4-GA2",
        "hamiltonian",
        "maloq",
        4,
        2,
        "sigterm",
        "mb4-ga2",
    ),
    failed(
        "s7u5badp",
        "outputs/qh9stable-oom-recovery-four-lane-mb4-ga2-eb8-full-seed44-20260723-041305/hamiltonian-maloq-nte",
        "FAILED | QH9Stable HΔ | NTE-64/2 | SIGTERM-MB4-GA2",
        "hamiltonian",
        "nte64e2",
        4,
        2,
        "sigterm",
        "mb4-ga2",
    ),
    failed(
        "j0auwheh",
        "outputs/qh9stable-oom-recovery-four-lane-mb4-ga2-eb8-full-seed44-20260723-041305/hamiltonian-qhflow3",
        "FAILED | QH9Stable HΔ | QHFlow3 | SIGTERM-MB4-GA2",
        "hamiltonian",
        "qhf3",
        4,
        2,
        "sigterm",
        "mb4-ga2",
    ),
)


# These are intentionally not members of RUNS. W&B still marks them running, so
# this audited cleanup must not alter their names, groups, tags, or notes.
EXCLUDED_RUNNING = (
    {
        "id": "ef0jlqcw",
        "local_output": (
            "outputs/qh9stable-delta-both-six-lane-bs32-full-seed44-"
            "20260722-191207/density-qhflow3"
        ),
        "local_progress": "50/80 epochs",
    },
    {
        "id": "o64mfwkd",
        "local_output": (
            "outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-full-"
            "seed44-20260723-133856/hamiltonian-qhflow3"
        ),
        "local_progress": "33 complete epochs; epoch 34 started",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update W&B. Without this flag, print an audited dry run.",
    )
    return parser.parse_args()


def validate_local_run(item: dict[str, object]) -> Path:
    output = PROJECT_ROOT / str(item["local_output"])
    matches = list((output / "wandb").glob(f"run-*-{item['id']}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"{item['id']} expected one local W&B directory below {output}; "
            f"found {matches}"
        )
    if item["local_status"] == "complete":
        required = (
            "backbone.pt",
            "backbone_state_dic.pt",
            "head.pt",
            "head_state_dic.pt",
            "comparison.json",
        )
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise RuntimeError(
                f"{item['id']} is labeled complete but lacks {missing}"
            )
        loss_path = output / "backbone_training_loss.txt"
        epochs = sum(1 for _ in loss_path.open(encoding="utf-8"))
        if epochs != EXPECTED_EPOCHS:
            raise RuntimeError(
                f"{item['id']} is labeled complete but has {epochs} loss rows"
            )
    return matches[0]


def main() -> None:
    args = parse_args()
    ids = [str(item["id"]) for item in RUNS]
    excluded_ids = [str(item["id"]) for item in EXCLUDED_RUNNING]
    all_ids = (*ids, *excluded_ids)
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Audited QH9Stable W&B IDs are not unique")

    api = wandb.Api(timeout=90)
    remote_runs = {run.id: run for run in api.runs(WANDB_PROJECT)}
    unknown_ids = sorted(set(remote_runs) - set(all_ids))
    missing_ids = sorted(set(all_ids) - set(remote_runs))
    if unknown_ids or missing_ids:
        raise RuntimeError(
            f"W&B inventory changed: unknown={unknown_ids}, missing={missing_ids}"
        )

    for item in EXCLUDED_RUNNING:
        run = remote_runs[str(item["id"])]
        if run.state != "running":
            raise RuntimeError(
                f"Excluded run {run.id} is no longer running: {run.state}"
            )
        print(
            json.dumps(
                {
                    "id": run.id,
                    "state": run.state,
                    "name": run.name,
                    "local_output": item["local_output"],
                    "local_progress": item["local_progress"],
                    "action": "excluded_running",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    changed = 0
    for item in RUNS:
        validate_local_run(item)
        inventory_run = remote_runs[str(item["id"])]
        if inventory_run.state == "running":
            raise RuntimeError(
                f"Refusing to alter active run {inventory_run.id}; "
                "add it to EXCLUDED_RUNNING"
            )
        # Objects returned by api.runs() may omit json_config, which update()
        # requires. Fetch a fully hydrated run after the inventory/state guard.
        run = api.run(f"{WANDB_PROJECT}/{item['id']}")
        if run.state != inventory_run.state:
            raise RuntimeError(
                f"Run {run.id} changed state during the audit: "
                f"{inventory_run.state} -> {run.state}"
            )

        desired_tags = tuple(dict.fromkeys(tuple(item["tags"])))
        provenance = (
            "[SC26 naming] Local artifacts: "
            f"{PROJECT_ROOT / str(item['local_output'])}."
        )
        if item["note"]:
            provenance += f" {item['note']}"
        existing_notes = (run.notes or "").strip()
        desired_notes = existing_notes
        if provenance not in existing_notes:
            desired_notes = "\n\n".join(
                value for value in (existing_notes, provenance) if value
            )

        before = {
            "name": run.name,
            "group": run.group,
            "job_type": run.job_type,
            "tags": list(run.tags or ()),
        }
        after = {
            "name": item["name"],
            "group": item["group"],
            "job_type": item["job_type"],
            "tags": list(desired_tags),
        }
        metadata_matches = (
            before["name"] == after["name"]
            and before["group"] == after["group"]
            and before["job_type"] == after["job_type"]
            and set(before["tags"]) == set(after["tags"])
        )
        if not metadata_matches or desired_notes != existing_notes:
            changed += 1

        if args.apply:
            run.name = str(item["name"])
            run.group = str(item["group"])
            run.job_type = str(item["job_type"])
            run.tags = desired_tags
            run.notes = desired_notes
            run.update()

        print(
            json.dumps(
                {
                    "id": item["id"],
                    "state": run.state,
                    "before": before,
                    "after": after,
                    "applied": args.apply,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    print(
        json.dumps(
            {
                "project": WANDB_PROJECT,
                "inactive_runs": len(RUNS),
                "excluded_running": len(EXCLUDED_RUNNING),
                "changed": changed,
                "applied": args.apply,
                "local_outputs_renamed": False,
                "runs_deleted": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
