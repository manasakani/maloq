#!/usr/bin/env python3
"""Rename NablaDFT W&B normalization lanes from audited run config.

The script is read-only unless ``--apply`` is supplied.  It only selects runs
whose display name already contains a normalization lane token, so unrelated
NablaDFT experiments are left untouched.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
REPORT_ROOT = PROJECT_ROOT / "outputs" / "wandb-naming"
WANDB_PROJECT = "kaist-korea/maloq-nablaDFT"
LANE_TOKENS = {"SS0", "SS1", "RAW", "SHIFT", "SHIFT+STD"}
AUDIT_NOTE_PREFIX = "[SC26 normalization]"


def normalization_lane(config: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Return the display label and canonical tags for one audited config."""
    if not bool(config.get("scale_and_shift", False)):
        return "RAW", ("normalization:none", "target:raw")

    mode = config.get("scale_shift_mode", "standardize")
    if mode == "shift_only":
        return "SHIFT", (
            "normalization:l0-shift-only",
            "target:mean-centered",
        )
    if mode == "standardize":
        return "SHIFT+STD", (
            "normalization:l0-shift-std",
            "target:standardized",
        )
    raise ValueError(f"Unsupported scale_shift_mode: {mode!r}")


def rename_lane(name: str, label: str) -> str | None:
    """Replace an existing lane segment, or return None for unrelated runs."""
    segments = [segment.strip() for segment in name.split("|")]
    lane_indices = [
        index for index, segment in enumerate(segments) if segment in LANE_TOKENS
    ]
    if not lane_indices:
        return None
    if len(lane_indices) != 1:
        raise ValueError(f"Ambiguous normalization segments in {name!r}")
    segments[lane_indices[0]] = label
    return " | ".join(segments)


def canonical_tags(
    existing: tuple[str, ...] | list[str],
    lane_tags: tuple[str, ...],
) -> list[str]:
    retained = [
        tag
        for tag in existing
        if not tag.startswith("normalization:") and not tag.startswith("target:")
    ]
    return sorted(set((*retained, *lane_tags)))


def canonical_notes(notes: str | None, label: str) -> str:
    note = (notes or "").strip()
    note = note.replace("replacement SS0 run", "replacement SHIFT run")
    note = note.replace("SS1 final validation", "SHIFT+STD final validation")
    audit_line = (
        f"{AUDIT_NOTE_PREFIX} Audited label: {label} "
        "(RAW=no transform; SHIFT=mean subtraction; "
        "SHIFT+STD=mean subtraction and standard-deviation scaling)."
    )
    lines = [
        line for line in note.splitlines() if not line.startswith(AUDIT_NOTE_PREFIX)
    ]
    body = "\n".join(lines).strip()
    return f"{body}\n\n{audit_line}".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, print and save a dry-run report.",
    )
    parser.add_argument("--project", default=WANDB_PROJECT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = wandb.Api(timeout=60)
    discovered = api.runs(args.project, order="+created_at")
    selected_ids = [
        run.id
        for run in discovered
        if rename_lane(run.name or "", "RAW") is not None
    ]

    changes: list[dict[str, Any]] = []
    for run_id in selected_ids:
        # Fetch a fresh mutable object rather than mutating the collection view.
        run = api.run(f"{args.project}/{run_id}")
        config = dict(run.config)
        label, lane_tags = normalization_lane(config)
        desired_name = rename_lane(run.name or "", label)
        if desired_name is None:  # pragma: no cover - guarded by selection
            continue
        desired_tags = canonical_tags(run.tags, lane_tags)
        desired_notes = canonical_notes(run.notes, label)
        changed = (
            run.name != desired_name
            or sorted(run.tags) != desired_tags
            or (run.notes or "").strip() != desired_notes
        )
        record = {
            "id": run.id,
            "state": run.state,
            "label": label,
            "before_name": run.name,
            "after_name": desired_name,
            "before_tags": sorted(run.tags),
            "after_tags": desired_tags,
            "changed": changed,
        }
        changes.append(record)
        print(
            f"{run.id} [{run.state}] {run.name!r} -> {desired_name!r}"
            f"{' [change]' if changed else ' [ok]'}"
        )
        if args.apply and changed:
            run.name = desired_name
            run.tags = desired_tags
            run.notes = desired_notes
            run.update()

    counts = {
        label: sum(change["label"] == label for change in changes)
        for label in ("RAW", "SHIFT", "SHIFT+STD")
    }
    report = {
        "project": args.project,
        "mode": "apply" if args.apply else "dry-run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected": len(changes),
        "changed": sum(change["changed"] for change in changes),
        "counts": counts,
        "runs": changes,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_ROOT / (
        "nabladft-normalization-labels-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{report['mode']}.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report={report_path}")
    print(f"counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
