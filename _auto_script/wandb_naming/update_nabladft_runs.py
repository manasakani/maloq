#!/usr/bin/env python3
"""Apply the audited SC26 naming scheme to local NablaDFT W&B runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
WANDB_PROJECT = "kaist-korea/maloq-nablaDFT"


def spec(
    run_id: str,
    local_output: str,
    name: str,
    group: str,
    job_type: str,
    *tags: str,
    note: str = "",
) -> dict[str, object]:
    group = {
        "ndft-archive-smoke": "nabla-smoke",
        "ndft-archive-failed": "nabla-failed",
    }.get(group, group)
    tags = tuple("auxiliary" if tag == "archive" else tag for tag in tags)
    if "| RAW |" in name:
        normalization_tags = ("normalization:none", "target:raw")
    elif "| SHIFT |" in name:
        normalization_tags = (
            "normalization:l0-shift-only",
            "target:mean-centered",
        )
    elif "| SHIFT+STD |" in name:
        normalization_tags = (
            "normalization:l0-shift-std",
            "target:standardized",
        )
    else:
        normalization_tags = ()
    return {
        "id": run_id,
        "local_output": local_output,
        "name": name,
        "group": group,
        "job_type": job_type,
        "tags": (
            "dataset:nabladft",
            "sc26-seongsu",
            *normalization_tags,
            *tags,
        ),
        "note": note,
    }


RUNS = (
    # Current comparison set.
    spec(
        "9ldrunh9",
        "outputs/nabladft-maloq-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260722-190344/maloq",
        "NablaDFT | MALOQ | MatMuon+SemHead | RAW | V1",
        "nabla-maloq-ss",
        "full",
        "current",
        "status:complete",
        "scope:full",
        "model:maloq",
        "head:muon",
        "scale-shift:off",
        "seed:44",
        "version:v1",
    ),
    spec(
        "loaiifgp",
        "outputs/nabladft-maloq-nte-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260722-190356/maloq-nte",
        "NablaDFT | NTE-64/2 | MatMuon+SemHead | RAW | V1",
        "nabla-nte64e2-head-ss",
        "full",
        "current",
        "status:complete",
        "scope:full",
        "model:nte64e2",
        "head:muon",
        "scale-shift:off",
        "seed:44",
        "version:v1",
    ),
    spec(
        "cvnthb0u",
        "outputs/nabladft-maloq-nte-do128-le3-head-comparison-parallel-2x2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-154105/native-head",
        "NablaDFT | NTE-128/3 | Native | RAW | V1",
        "nabla-nte128e3-head-ss",
        "full",
        "current",
        "status:complete",
        "scope:full",
        "model:nte128e3",
        "head:native",
        "scale-shift:off",
        "seed:44",
        "version:v1",
    ),
    spec(
        "119izc66",
        "outputs/nabladft-maloq-nte-do128-le3-head-comparison-parallel-2x2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-154105/muon-head",
        "NablaDFT | NTE-128/3 | MatMuon+SemHead | RAW | V1",
        "nabla-nte128e3-head-ss",
        "full",
        "current",
        "status:complete",
        "scope:full",
        "model:nte128e3",
        "head:muon",
        "scale-shift:off",
        "seed:44",
        "version:v1",
    ),
    spec(
        "27dk4l35",
        "outputs/nabladft-nte-do128-le3-native-head-scale-shift-2gpu-eb20-mb5-ga2-full-e20-seed44-20260724-042801/run",
        "NablaDFT | NTE-128/3 | Native | SHIFT+STD | V1",
        "nabla-nte128e3-head-ss",
        "full",
        "current",
        "status:artifact-complete",
        "wrapper:failed",
        "scope:full",
        "model:nte128e3",
        "head:native",
        "scale-shift:on",
        "seed:44",
        "version:v1",
        note="Training reached epoch 20 and saved all artifacts; the local wrapper exited 1.",
    ),
    spec(
        "qpa1dbz8",
        "outputs/nabladft-nte-do128-le3-muon-head-scale-shift-2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-164854/run",
        "NablaDFT | NTE-128/3 | MatMuon+SemHead | SHIFT+STD | V1",
        "nabla-nte128e3-head-ss",
        "full",
        "current",
        "status:artifact-complete",
        "wrapper:failed",
        "scope:full",
        "model:nte128e3",
        "head:muon",
        "scale-shift:on",
        "seed:44",
        "version:v1",
        note="Training reached epoch 20 and saved all artifacts; the local wrapper exited 127.",
    ),
    spec(
        "zqs1eohc",
        "outputs/nabladft-qhflow3-local-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-170017/qhflow3",
        "NablaDFT | QHFlow3 | MatMuon+SemHead | RAW | V2",
        "nabla-qhflow3-ss",
        "full",
        "current",
        "status:complete",
        "scope:full",
        "model:qhf3",
        "head:muon",
        "scale-shift:off",
        "seed:44",
        "version:v2",
    ),
    spec(
        "2ygp53bs",
        "outputs/nabladft-qhflow3-local-muon-head-scale-shift-2gpu-eb20-mb5-ga2-full-e20-seed44-20260724-043801/run",
        "NablaDFT | QHFlow3 | MatMuon+SemHead | SHIFT+STD | V1",
        "nabla-qhflow3-ss",
        "full",
        "current",
        "status:complete",
        "scope:full",
        "model:qhf3",
        "head:muon",
        "scale-shift:on",
        "seed:44",
        "version:v1",
    ),
    spec(
        "jal9l7uk",
        "outputs/nabladft-maloq-muon-head-ss/run",
        "NablaDFT | MALOQ | MatMuon+SemHead | SHIFT+STD | V1",
        "nabla-maloq-ss",
        "full",
        "current",
        "status:complete",
        "scope:full",
        "model:maloq",
        "head:muon",
        "scale-shift:on",
        "seed:44",
        "version:v1",
    ),
    # Current NTE scaling-law set.
    spec(
        "bbuqap9p",
        "outputs/ndft-nte-muon-scaling-4point-eb20-full-e20-seed44-20260724-165100/runs/p16m-w88-d3",
        "NablaDFT | NTE Scaling | 16M | W88 D3 | V1",
        "nabla-nte-muon-scaling",
        "full",
        "current",
        "scope:full",
        "scaling-law",
        "model:nte88e3",
        "head:muon",
        "scale-shift:off",
        "params:16m",
        "params:16125037",
        "width:88",
        "depth:3",
        "seed:44",
        "version:v1",
    ),
    spec(
        "w9m2o09g",
        "outputs/ndft-nte-muon-scaling-4point-eb20-full-e20-seed44-20260724-165100/runs/p33m-w128-d3",
        "NablaDFT | NTE Scaling | 33M | W128 D3 | V1",
        "nabla-nte-muon-scaling",
        "full",
        "current",
        "scope:full",
        "scaling-law",
        "model:nte128e3",
        "head:muon",
        "scale-shift:off",
        "params:33m",
        "params:33750157",
        "width:128",
        "depth:3",
        "seed:44",
        "version:v1",
    ),
    spec(
        "wosfc6ww",
        "outputs/ndft-nte-muon-scaling-4point-eb20-full-e20-seed44-20260724-165100/runs/p125m-w192-d5",
        "NablaDFT | NTE Scaling | 125M | W192 D5 | V1",
        "nabla-nte-muon-scaling",
        "full",
        "current",
        "scope:full",
        "scaling-law",
        "model:nte192e5",
        "head:muon",
        "scale-shift:off",
        "params:125m",
        "params:125004341",
        "width:192",
        "depth:5",
        "seed:44",
        "version:v1",
    ),
    spec(
        "z9gt0ohd",
        "outputs/ndft-nte-muon-scaling-4point-eb20-full-e20-seed44-20260724-165100/runs/p500m-w384-d5",
        "NablaDFT | NTE Scaling | 500M | W384 D5 | V1",
        "nabla-nte-muon-scaling",
        "full",
        "current",
        "scope:full",
        "scaling-law",
        "model:nte384e5",
        "head:muon",
        "scale-shift:off",
        "params:500m",
        "params:496331189",
        "width:384",
        "depth:5",
        "seed:44",
        "version:v1",
    ),
    # Superseded full runs.
    spec(
        "732zfuml",
        "outputs/nabladft-three-model-parallel-3x2gpu-eb20-mb5-ga2-full-e20-seed44-20260722-061320/maloq",
        "NablaDFT | MALOQ | Native | RAW | V1",
        "nabla-initial-3model",
        "full",
        "superseded",
        "status:complete",
        "scope:full",
        "model:maloq",
        "head:native",
        "scale-shift:off",
        "version:v1",
    ),
    spec(
        "cz7gx8ro",
        "outputs/nabladft-three-model-parallel-3x2gpu-eb20-mb5-ga2-full-e20-seed44-20260722-061333/maloq-nte",
        "NablaDFT | NTE-64/2 | Native | RAW | V1",
        "nabla-initial-3model",
        "full",
        "superseded",
        "status:complete",
        "scope:full",
        "model:nte64e2",
        "head:native",
        "scale-shift:off",
        "version:v1",
    ),
    spec(
        "tqmy5qme",
        "outputs/nabladft-three-model-parallel-3x2gpu-eb20-mb5-ga2-full-e20-seed44-20260722-061342/qhflow3",
        "NablaDFT | QHFlow3 | Native | RAW | V1",
        "nabla-initial-3model",
        "full",
        "superseded",
        "status:complete",
        "scope:full",
        "model:qhf3",
        "head:native",
        "objective:pre-fix",
        "scale-shift:off",
        "version:v1",
    ),
    spec(
        "wa3w0p5j",
        "outputs/nabladft-qhflow3-clean-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260722-181348/qhflow3",
        "NablaDFT | QHFlow3 | MatMuon+SemHead | RAW | V1",
        "nabla-qhflow3-ss",
        "full",
        "superseded",
        "status:complete",
        "scope:full",
        "model:qhf3",
        "head:muon",
        "objective:pre-fix",
        "scale-shift:off",
        "version:v1",
    ),
    # Smokes and failed/partial attempts.
    spec(
        "pl1ebefz",
        "outputs/nabladft-data-parallel-2gpu-eb20-baseline-smoke-20260722-050039/baseline",
        "SMOKE | MALOQ | DP | A1",
        "ndft-archive-smoke",
        "smoke",
        "archive",
        "scope:smoke",
        "model:maloq",
        "parallel:data",
    ),
    spec(
        "dnbiv03m",
        "outputs/nabladft-data-parallel-2gpu-eb20-baseline-smoke-20260722-050131/baseline",
        "SMOKE | MALOQ | DP | A2",
        "ndft-archive-smoke",
        "smoke",
        "archive",
        "scope:smoke",
        "model:maloq",
        "parallel:data",
    ),
    spec(
        "uizt6e30",
        "outputs/nabladft-distributed-graph-2gpu-eb20-baseline-smoke-20260722-050233/baseline",
        "SMOKE | MALOQ | DG | A1",
        "ndft-archive-smoke",
        "smoke",
        "archive",
        "scope:smoke",
        "model:maloq",
        "parallel:graph",
    ),
    spec(
        "mzdnxle0",
        "outputs/nabladft-distributed-graph-2gpu-eb20-baseline-smoke-20260722-050324/baseline",
        "SMOKE | MALOQ | DG | A2",
        "ndft-archive-smoke",
        "smoke",
        "archive",
        "scope:smoke",
        "model:maloq",
        "parallel:graph",
    ),
    spec(
        "yy5s73nv",
        "outputs/nabladft-data-parallel-2gpu-eb20-maloq-nte-smoke-20260722-050357/maloq-nte",
        "SMOKE | NTE-64/2 | DP",
        "ndft-archive-smoke",
        "smoke",
        "archive",
        "scope:smoke",
        "model:nte64e2",
        "parallel:data",
    ),
    spec(
        "ij15ubpj",
        "outputs/nabladft-distributed-graph-2gpu-eb20-maloq-nte-smoke-20260722-050434/maloq-nte",
        "SMOKE | NTE-64/2 | DG",
        "ndft-archive-smoke",
        "smoke",
        "archive",
        "scope:smoke",
        "model:nte64e2",
        "parallel:graph",
    ),
    spec(
        "as7amzph",
        "outputs/nabladft-qhflow3-2gpu-eb20-smoke-seed44-20260722-052320/qhflow3",
        "SMOKE | QHFlow3 | 1ep",
        "ndft-archive-smoke",
        "smoke",
        "archive",
        "scope:smoke",
        "model:qhf3",
    ),
    spec(
        "374a3i1y",
        "outputs/nabladft-wandb-step10-2gpu-eb20-smoke-20260722-052405/baseline",
        "SMOKE | W&B cadence | 1ep",
        "ndft-archive-smoke",
        "smoke",
        "archive",
        "scope:smoke",
        "check:wandb-cadence",
    ),
    spec(
        "f1eny2c8",
        "outputs/nabladft-maloq-muon-e20-full-seed44-20260722-050347/baseline",
        "FAILED | MALOQ | partial",
        "ndft-archive-failed",
        "failed-full",
        "archive",
        "status:failed",
        "scope:full",
        "model:maloq",
    ),
    spec(
        "asks97h2",
        "outputs/nabladft-maloq-nte-muon-e20-full-seed44-20260722-050406/maloq-nte",
        "FAILED | NTE-64/2 | startup",
        "ndft-archive-failed",
        "failed-full",
        "archive",
        "status:failed",
        "scope:full",
        "model:nte64e2",
    ),
    spec(
        "7j69zymx",
        "outputs/nabladft-maloq-vs-nte-muon-e20-full-seed44-20260722-050418/baseline",
        "FAILED | MALOQ vs NTE | partial",
        "ndft-archive-failed",
        "failed-full",
        "archive",
        "status:failed",
        "scope:full",
    ),
    spec(
        "rvl7v4vu",
        "outputs/nabladft-maloq-nte-qhflow3-muon-e20-full-seed44-20260722-053552/baseline",
        "FAILED | 3-model | startup",
        "ndft-archive-failed",
        "failed-full",
        "archive",
        "status:failed",
        "scope:full",
    ),
    spec(
        "u02xml0e",
        "outputs/nabladft-qhflow3-local-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-140051/qhflow3",
        "FAILED | QHFlow3 | loader A1",
        "ndft-archive-failed",
        "failed-full",
        "archive",
        "status:failed",
        "scope:full",
        "model:qhf3",
    ),
    spec(
        "52qlw16i",
        "outputs/nabladft-qhflow3-local-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-151840/qhflow3",
        "FAILED | QHFlow3 | loader A2",
        "ndft-archive-failed",
        "failed-full",
        "archive",
        "status:failed",
        "scope:full",
        "model:qhf3",
    ),
    spec(
        "cfnl9tmh",
        "outputs/nabladft-qhflow3-local-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-152825/qhflow3",
        "FAILED | QHFlow3 | partial",
        "ndft-archive-failed",
        "failed-full",
        "archive",
        "status:failed",
        "scope:full",
        "model:qhf3",
    ),
    spec(
        "m0hmvvyn",
        "outputs/nabladft-qhflow3-local-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-163933/qhflow3",
        "FAILED | QHFlow3 | 1/20ep",
        "ndft-archive-failed",
        "failed-full",
        "archive",
        "status:failed",
        "scope:full",
        "model:qhf3",
    ),
)

AUXILIARY_RUNS = tuple(
    item
    for item in RUNS
    if str(item["name"]).startswith(("FAILED | ", "SMOKE | "))
)
RUNS = tuple(item for item in RUNS if item not in AUXILIARY_RUNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--apply",
        action="store_true",
        help="Update W&B. Without this flag, print an audited dry run.",
    )
    action.add_argument(
        "--delete-auxiliary",
        action="store_true",
        help=(
            "Permanently delete the audited FAILED and SMOKE runs from W&B while "
            "retaining their local outputs."
        ),
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
    return matches[0]


def validate_auxiliary_remote_run(
    item: dict[str, object],
    run: wandb.apis.public.Run,
) -> None:
    name = str(item["name"])
    tags = set(run.tags or ())
    expected = {
        "name": name,
        "group": item["group"],
        "job_type": item["job_type"],
    }
    actual = {
        "name": run.name,
        "group": run.group,
        "job_type": run.job_type,
    }
    if actual != expected:
        raise RuntimeError(
            f"Refusing to delete {item['id']}: metadata changed from the audit; "
            f"expected {expected}, found {actual}"
        )
    if name.startswith("FAILED | ") and "status:failed" not in tags:
        raise RuntimeError(
            f"Refusing to delete {item['id']}: missing status:failed tag"
        )
    if name.startswith("SMOKE | ") and "scope:smoke" not in tags:
        raise RuntimeError(
            f"Refusing to delete {item['id']}: missing scope:smoke tag"
        )


def delete_auxiliary_runs(api: wandb.Api) -> None:
    remote_runs = {run.id: run for run in api.runs(WANDB_PROJECT)}
    deleted = 0
    already_deleted = 0
    for item in AUXILIARY_RUNS:
        validate_local_run(item)
        run = remote_runs.get(str(item["id"]))
        if run is None:
            already_deleted += 1
            print(
                json.dumps(
                    {"id": item["id"], "action": "already_deleted"},
                    sort_keys=True,
                )
            )
            continue
        validate_auxiliary_remote_run(item, run)
        print(
            json.dumps(
                {
                    "id": item["id"],
                    "name": run.name,
                    "group": run.group,
                    "job_type": run.job_type,
                    "action": "delete",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        run.delete(delete_artifacts=False)
        deleted += 1
    print(
        json.dumps(
            {
                "project": WANDB_PROJECT,
                "targets": len(AUXILIARY_RUNS),
                "deleted": deleted,
                "already_deleted": already_deleted,
                "local_outputs_retained": True,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    ids = [str(item["id"]) for item in (*RUNS, *AUXILIARY_RUNS)]
    if len(ids) != len(set(ids)):
        raise RuntimeError("RUNS contains duplicate W&B IDs")
    api = wandb.Api(timeout=60)
    if args.delete_auxiliary:
        delete_auxiliary_runs(api)
        return
    for item in RUNS:
        validate_local_run(item)

    changed = 0
    for item in RUNS:
        run = api.run(f"{WANDB_PROJECT}/{item['id']}")
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
        if args.apply:
            run.name = str(item["name"])
            run.group = str(item["group"])
            run.job_type = str(item["job_type"])
            run.tags = desired_tags
            run.notes = desired_notes
            run.update()
        metadata_matches = (
            before["name"] == after["name"]
            and before["group"] == after["group"]
            and before["job_type"] == after["job_type"]
            and set(before["tags"]) == set(after["tags"])
        )
        if not metadata_matches or desired_notes != existing_notes:
            changed += 1
        print(
            json.dumps(
                {
                    "id": item["id"],
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
                "runs": len(RUNS),
                "changed": changed,
                "applied": args.apply,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
