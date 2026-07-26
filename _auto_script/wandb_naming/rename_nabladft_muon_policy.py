#!/usr/bin/env python3
"""Give NablaDFT Muon runs compact, explicit W&B policy names.

The script is read-only unless ``--apply`` is supplied.  It only selects the
two audited Muon labels used by the SC26 NablaDFT experiments and verifies the
recorded optimizer/head configuration before changing metadata.
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
AUDIT_NOTE_PREFIX = "[SC26 Muon naming]"

POLICIES = {
    "Muon": {
        "label": "MatMuon+SemHead",
        "head_type": "maloq_muon",
        "routing_tag": "head-routing:semantic-matrix",
    },
    "MatMuon+AdamW+SemHead": {
        "label": "MatMuon+SemHead",
        "head_type": "maloq_muon",
        "routing_tag": "head-routing:semantic-matrix",
    },
    "MatrixMuon+AuxAdamW+SGHead": {
        "label": "MatMuon+SGHead",
        "head_type": "maloq_semantic_global_muon",
        "routing_tag": "head-routing:semantic-global",
    },
    "MatMuon+AdamW+SGHead": {
        "label": "MatMuon+SGHead",
        "head_type": "maloq_semantic_global_muon",
        "routing_tag": "head-routing:semantic-global",
    },
}


def renamed_policy(name: str) -> tuple[str, dict[str, str]] | None:
    """Replace one exact policy segment and return its audited policy."""
    segments = [segment.strip() for segment in name.split("|")]
    matches = [
        (index, POLICIES[segment])
        for index, segment in enumerate(segments)
        if segment in POLICIES
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Ambiguous Muon policy segments in {name!r}")
    index, policy = matches[0]
    segments[index] = policy["label"]
    return " | ".join(segments), policy


def validate_config(run_id: str, config: dict[str, Any], policy: dict[str, str]) -> None:
    """Refuse a label that does not describe the recorded run."""
    optimizer_type = config.get("optimizer_type")
    head_type = config.get("head_type")
    if optimizer_type != "muon" or head_type != policy["head_type"]:
        raise ValueError(
            f"{run_id}: expected optimizer_type='muon' and "
            f"head_type={policy['head_type']!r}, got "
            f"{optimizer_type!r} and {head_type!r}."
        )


def canonical_tags(
    existing: tuple[str, ...] | list[str],
    policy: dict[str, str],
) -> list[str]:
    """Attach exact optimizer/routing tags while retaining unrelated tags."""
    managed_prefixes = (
        "optimizer:",
        "muon-routing:",
        "aux-optimizer:",
        "head-routing:",
    )
    retained = [
        tag for tag in existing if not tag.startswith(managed_prefixes)
    ]
    return sorted(
        set(
            (
                *retained,
                "optimizer:muon",
                "muon-routing:ndim-ge-2",
                "aux-optimizer:adamw",
                policy["routing_tag"],
            )
        )
    )


def canonical_notes(notes: str | None, policy: dict[str, str]) -> str:
    """Add a short durable explanation without discarding existing notes."""
    lines = [
        line
        for line in (notes or "").strip().splitlines()
        if not line.startswith(AUDIT_NOTE_PREFIX)
    ]
    body = "\n".join(lines).strip()
    audit_line = (
        f"{AUDIT_NOTE_PREFIX} {policy['label']}: trainable ndim>=2 parameters "
        "use Muon; the suffix identifies the semantic matrix-head routing. "
        "Auxiliary optimizer details are recorded in W&B tags."
    )
    return f"{body}\n\n{audit_line}".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, only audit and save a report.",
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
        if renamed_policy(run.name or "") is not None
    ]

    changes: list[dict[str, Any]] = []
    for run_id in selected_ids:
        run = api.run(f"{args.project}/{run_id}")
        result = renamed_policy(run.name or "")
        if result is None:  # pragma: no cover - guarded by selection
            continue
        desired_name, policy = result
        validate_config(run.id, dict(run.config), policy)
        desired_tags = canonical_tags(run.tags, policy)
        desired_notes = canonical_notes(run.notes, policy)
        changed = (
            run.name != desired_name
            or sorted(run.tags) != desired_tags
            or (run.notes or "").strip() != desired_notes
        )
        record = {
            "id": run.id,
            "state": run.state,
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

    report = {
        "project": args.project,
        "mode": "apply" if args.apply else "dry-run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected": len(changes),
        "changed": sum(change["changed"] for change in changes),
        "runs": changes,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_ROOT / (
        "nabladft-muon-policy-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{report['mode']}.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report={report_path}")
    print(f"selected={report['selected']} changed={report['changed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
