#!/usr/bin/env python3
"""Compare trained NablaDFT QHFlow3 and NTE features layer by layer.

The analysis uses the same native MALOQ validation rows for both checkpoints.
It records component-normalized degree statistics, channel distributions, and
the learned channel contractions in the backbone output projection and Muon
matrix head.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

DEFAULT_DB = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/"
    "hamiltonian_databases/train_2k.db"
)
DEFAULT_QHF_CONFIG = (
    PROJECT_ROOT
    / "_my_script/experiment/2026-07-26/"
    "qhflow3_ov0_ntegrid_projection_muon_nabladft.yaml"
)
DEFAULT_NTE_CONFIG = (
    PROJECT_ROOT
    / "_my_script/experiment/2026-07-25/"
    "nte64e2_qhflow3_conditioning_nabladft.yaml"
)
DEFAULT_QHF_RUN = (
    PROJECT_ROOT
    / "outputs/"
    "nabla-qhf3-projmuon-ov0-ntegrid-v3/run"
)
DEFAULT_NTE_RUN = (
    PROJECT_ROOT
    / "outputs/"
    "nabla-nte64e2-muon-ss0-qcond-v1/run"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/nabladft-qhf-vs-nte-layer-analysis"
MODEL_LABELS = {
    "qhflow3": (
        "NablaDFT | QHFlow3 | MatrixMuon+ProjMuon+AuxAdamW | RAW | "
        "OV0 | NTEGrid10x11 | V3"
    ),
    "nte": "NablaDFT | NTE-64/2 | Muon | RAW | QHFcond | V1",
}
FEATURE_STAGE_ORDER = {
    "nte": {
        "node": [
            "input_after_edge_degree",
            "node_1_edgewise_raw",
            "node_1_edge_update",
            "node_1_atomwise_raw",
            "node_1_atom_update",
            "node_block_1",
            "node_2_edgewise_raw",
            "node_2_edge_update",
            "node_2_atomwise_raw",
            "node_2_atom_update",
            "node_block_2",
            "node_3_edgewise_raw",
            "node_3_edge_update",
            "node_3_atomwise_raw",
            "node_3_atom_update",
            "node_block_3",
            "pre_output_projection",
            "output_64",
            "head_gated_64",
            "head_semantic_output",
        ],
        "edge": [
            "input_before_edge_stack",
            "edge_1_edgewise_raw",
            "edge_1_edge_update",
            "edge_1_atomwise_raw",
            "edge_1_atom_update",
            "edge_block_1",
            "edge_2_edgewise_raw",
            "edge_2_edge_update",
            "edge_2_atomwise_raw",
            "edge_2_atom_update",
            "edge_block_2",
            "pre_output_projection",
            "output_64",
            "head_gated_64",
            "head_semantic_output",
        ],
    },
    "qhflow3": {
        "node": [
            "input_after_edge_degree",
            "node_1_edgewise_update",
            "node_1_atomwise_update",
            "node_block_1",
            "node_2_edgewise_update",
            "node_2_atomwise_update",
            "node_block_2",
            "node_3_edgewise_update",
            "node_3_atomwise_update",
            "node_block_3",
            "node_normalized",
            "pre_output_projection",
            "output_64",
            "head_gated_64",
            "head_semantic_output",
        ],
        "edge": [
            "pair_1_edgewise_raw",
            "pair_1_atomwise_update",
            "pair_block_1_raw",
            "pair_2_edgewise_raw",
            "pair_2_atomwise_update",
            "pair_block_2_raw",
            "pair_sum_pre_norm",
            "pair_normalized",
            "pre_output_projection",
            "output_64",
            "head_gated_64",
            "head_semantic_output",
        ],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbpath", type=Path, default=DEFAULT_DB)
    parser.add_argument("--qhflow3-config", type=Path, default=DEFAULT_QHF_CONFIG)
    parser.add_argument("--nte-config", type=Path, default=DEFAULT_NTE_CONFIG)
    parser.add_argument("--qhflow3-run", type=Path, default=DEFAULT_QHF_RUN)
    parser.add_argument("--nte-run", type=Path, default=DEFAULT_NTE_RUN)
    parser.add_argument(
        "--qhflow3-label",
        default=MODEL_LABELS["qhflow3"],
        help="Reader-facing label stored in CSV and report outputs.",
    )
    parser.add_argument(
        "--nte-label",
        default=MODEL_LABELS["nte"],
        help="Reader-facing label stored in CSV and report outputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=8,
        help="Number of held-out NablaDFT rows, starting at row 12081.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Validation molecules per forward pass.",
    )
    parser.add_argument(
        "--models",
        choices=("both", "qhflow3", "nte"),
        default="both",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29651,
        help="Single-process torch.distributed rendezvous port.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and checkpoints without loading data or CUDA models.",
    )
    return parser


@dataclass
class DegreeAccumulator:
    item_count: int = 0
    component_count: int = 0
    calls: int = 0
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    channel_sum_sq: Any = None

    def add(self, block: Any) -> None:
        """Add an [items, 2*l+1, channels] tensor."""
        import torch

        if block.ndim != 3:
            raise ValueError(f"Expected a rank-3 spherical block, got {block.shape}.")
        values = block.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError("Non-finite feature encountered during analysis.")
        items, components, channels = (int(value) for value in values.shape)
        channel_sum_sq = values.square().sum(dim=(0, 1)).cpu().to(torch.float64)
        if self.channel_sum_sq is None:
            self.channel_sum_sq = channel_sum_sq
        elif tuple(self.channel_sum_sq.shape) != tuple(channel_sum_sq.shape):
            raise ValueError(
                "A stage changed channel count across batches: "
                f"{self.channel_sum_sq.shape} versus {channel_sum_sq.shape}."
            )
        else:
            self.channel_sum_sq += channel_sum_sq
        self.item_count += items
        self.component_count += items * components
        self.calls += 1
        self.sum_sq += float(values.square().sum().item())
        self.sum_abs += float(values.abs().sum().item())


@dataclass
class FeatureCollector:
    accumulators: dict[tuple[str, str, str, int], DegreeAccumulator] = field(
        default_factory=dict
    )

    def add_degree(
        self,
        model: str,
        kind: str,
        stage: str,
        degree: int,
        block: Any,
    ) -> None:
        key = (model, kind, stage, int(degree))
        if key not in self.accumulators:
            self.accumulators[key] = DegreeAccumulator()
        self.accumulators[key].add(block)

    def add_native(
        self,
        model: str,
        kind: str,
        stage: str,
        tensor: Any,
        lmax: int,
    ) -> None:
        expected = (lmax + 1) ** 2
        if tensor.ndim != 3 or int(tensor.shape[1]) != expected:
            raise ValueError(
                f"{model}/{kind}/{stage}: expected [items,{expected},channels], "
                f"got {tuple(tensor.shape)}."
            )
        for degree in range(lmax + 1):
            self.add_degree(
                model,
                kind,
                stage,
                degree,
                tensor[:, degree**2 : (degree + 1) ** 2, :],
            )

    def add_degree_major_flat(
        self,
        model: str,
        kind: str,
        stage: str,
        tensor: Any,
        lmax: int,
        channels: int,
    ) -> None:
        expected = channels * (lmax + 1) ** 2
        if tensor.ndim != 2 or int(tensor.shape[1]) != expected:
            raise ValueError(
                f"{model}/{kind}/{stage}: expected [items,{expected}], "
                f"got {tuple(tensor.shape)}."
            )
        for degree in range(lmax + 1):
            start = degree**2 * channels
            block = tensor[
                :, start : start + channels * (2 * degree + 1)
            ].reshape(tensor.shape[0], channels, 2 * degree + 1)
            self.add_degree(
                model,
                kind,
                stage,
                degree,
                block.transpose(1, 2),
            )

    def add_semantic_output(
        self,
        model: str,
        kind: str,
        stage: str,
        tensor: Any,
        module: Any,
    ) -> None:
        offset = 0
        for degree in module._degrees:
            rows = getattr(module, f"_rows_l{degree}")
            channels = int(rows.numel())
            width = channels * (2 * degree + 1)
            block = tensor[:, offset : offset + width].reshape(
                tensor.shape[0], channels, 2 * degree + 1
            )
            self.add_degree(
                model,
                kind,
                stage,
                degree,
                block.transpose(1, 2),
            )
            offset += width
        if offset != int(tensor.shape[1]):
            raise ValueError(
                f"Semantic output parser consumed {offset} of {tensor.shape[1]}."
            )

    def finalize(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        import numpy as np

        degree_rows: list[dict[str, Any]] = []
        channel_rows: list[dict[str, Any]] = []
        stage_power = defaultdict(float)
        for (model, kind, stage, _), acc in self.accumulators.items():
            stage_power[(model, kind, stage)] += acc.sum_sq

        for key in sorted(self.accumulators):
            model, kind, stage, degree = key
            acc = self.accumulators[key]
            channel_power = acc.channel_sum_sq.numpy() / acc.component_count
            channel_rms = np.sqrt(channel_power)
            power_total = stage_power[(model, kind, stage)]
            power_sum = float(channel_power.sum())
            effective_channels = (
                float(power_sum**2 / np.square(channel_power).sum())
                if np.square(channel_power).sum() > 0.0
                else 0.0
            )
            degree_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "feature_kind": kind,
                    "stage": stage,
                    "degree": degree,
                    "channels": int(channel_rms.size),
                    "items": acc.item_count,
                    "calls": acc.calls,
                    "rms": math.sqrt(
                        acc.sum_sq
                        / (acc.component_count * int(channel_rms.size))
                    ),
                    "mean_abs": (
                        acc.sum_abs
                        / (acc.component_count * int(channel_rms.size))
                    ),
                    "power_fraction": (
                        acc.sum_sq / power_total if power_total > 0.0 else 0.0
                    ),
                    "channel_rms_mean": float(channel_rms.mean()),
                    "channel_rms_std": float(channel_rms.std()),
                    "channel_rms_median": float(np.median(channel_rms)),
                    "channel_rms_q10": float(np.quantile(channel_rms, 0.10)),
                    "channel_rms_q90": float(np.quantile(channel_rms, 0.90)),
                    "channel_rms_min": float(channel_rms.min()),
                    "channel_rms_max": float(channel_rms.max()),
                    "effective_channels": effective_channels,
                    "effective_channel_fraction": (
                        effective_channels / int(channel_rms.size)
                    ),
                }
            )
            for channel, rms in enumerate(channel_rms.tolist()):
                channel_rows.append(
                    {
                        "model": model,
                        "feature_kind": kind,
                        "stage": stage,
                        "degree": degree,
                        "channel": channel,
                        "channel_rms": rms,
                    }
                )
        return degree_rows, channel_rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_checkpoint_paths(run_dir: Path) -> tuple[Path, Path]:
    return (
        run_dir / "backbone_state_dic.pt",
        run_dir / "head_state_dic.pt",
    )


def _load_legacy_state_payload(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load legacy checkpoint {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Legacy checkpoint {path} must contain a dictionary payload."
        )
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(
            f"Legacy checkpoint {path} has an empty or invalid state dict."
        )
    invalid_keys = [key for key in state if not isinstance(key, str)]
    invalid_values = [
        key for key, value in state.items() if not isinstance(value, torch.Tensor)
    ]
    if invalid_keys or invalid_values:
        raise RuntimeError(
            f"Legacy checkpoint {path} has invalid state-dict entries: "
            f"non_string_keys={len(invalid_keys)}, "
            f"non_tensor_values={len(invalid_values)}."
        )
    return payload, state


def validate_checkpoint_bundle(run_dir: Path) -> dict[str, Any]:
    """Validate a fixed checkpoint generation or the legacy state-dict pair.

    Fixed-workflow runs keep the current and previous epoch in one atomic
    payload.  ``load_training_checkpoint`` validates the current generation
    and automatically falls back to ``training_state.prev.pt`` when needed.
    Older runs remain analyzable through their separate backbone/head files.
    """
    from maloq.train_utils.training_workflow_fixed import (
        checkpoint_candidates,
        load_training_checkpoint,
    )

    run_dir = run_dir.expanduser().resolve()
    fixed_candidates = checkpoint_candidates(run_dir)
    fixed_present = any(path.is_file() for path in fixed_candidates)
    fixed_error: str | None = None
    if fixed_present:
        try:
            payload, selected = load_training_checkpoint(run_dir)
        except RuntimeError as exc:
            fixed_error = str(exc)
        else:
            return {
                "format": "fixed",
                "path": str(selected),
                "epoch": payload["completed_epoch"],
            }

    backbone_path, head_path = _legacy_checkpoint_paths(run_dir)
    if backbone_path.is_file() and head_path.is_file():
        try:
            _, backbone_state = _load_legacy_state_payload(backbone_path)
            _, head_state = _load_legacy_state_payload(head_path)
        except RuntimeError as exc:
            legacy_error = str(exc)
        else:
            result = {
                "format": "legacy",
                "backbone_path": str(backbone_path),
                "head_path": str(head_path),
                "backbone_keys": len(backbone_state),
                "head_keys": len(head_state),
            }
            if fixed_error is not None:
                result["fixed_checkpoint_error"] = fixed_error
            return result
    else:
        legacy_error = None

    missing = [
        str(path)
        for path in (backbone_path, head_path)
        if not path.is_file()
    ]
    details = (
        f"\nFixed checkpoint validation failed:\n{fixed_error}"
        if fixed_error is not None
        else ""
    )
    if legacy_error is not None:
        details += f"\nLegacy checkpoint validation failed:\n{legacy_error}"
        raise RuntimeError(
            f"No usable model checkpoint bundle in {run_dir}." + details
        )
    raise FileNotFoundError(
        f"No usable model checkpoint bundle in {run_dir}. "
        "Expected a valid training_state.pt/training_state.prev.pt generation "
        "or both legacy state dictionaries. Missing legacy files:\n"
        + "\n".join(missing)
        + details
    )


def validate_inputs(
    args: argparse.Namespace,
    models: list[str],
) -> dict[str, dict[str, Any]]:
    paths = [args.dbpath, args.output_dir.parent]
    if "qhflow3" in models:
        paths.extend([args.qhflow3_config, args.qhflow3_run])
    if "nte" in models:
        paths.extend([args.nte_config, args.nte_run])
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing analysis inputs:\n" + "\n".join(missing))
    if not 1 <= args.num_molecules <= 64:
        raise ValueError("--num-molecules must be between 1 and 64.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 1 <= args.master_port <= 65535:
        raise ValueError("--master-port must be between 1 and 65535.")
    run_dirs = {"qhflow3": args.qhflow3_run, "nte": args.nte_run}
    return {
        model: validate_checkpoint_bundle(run_dirs[model])
        for model in models
    }


def source_provenance() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain").splitlines()
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "worktree_dirty": bool(status),
        "worktree_change_count": len(status),
    }


def load_config(path: Path) -> dict[str, Any]:
    from maloq.core.config import MaloqConfig

    config = MaloqConfig.from_file(path).to_workflow_config()
    config.update(
        dataset_name="nablaDFT",
        loss_target="fock_matrix",
        delta_learning=False,
        use_wandb=False,
        train_or_eval="eval",
        distribute_graphs=False,
        rcut_orbitals=8.0,
        rcut_gaussian=16.0,
    )
    return config


def make_loader(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Any, ...]:
    from maloq.dataset_utils import get_loader
    from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

    database = HamiltonianDatabase(str(args.dbpath))
    val_start = 12081
    val_end = val_start + args.num_molecules
    return get_loader.get_loader(
        database=database,
        start_idx=val_start,
        end_idx=val_end,
        dataset_name="nablaDFT",
        rcut=8.0,
        batch_size=args.batch_size,
        dtype=config["dtype"],
        half_edges=False,
        make_fock_targets=True,
        scale_shift_data=None,
        is_open_shell=False,
        loss_target_string="fock_matrix",
        distribute_graphs=False,
        tiling_dims=None,
        partition_type=None,
        train_or_eval="eval",
        delta_learning=False,
        load_delta_auxiliary_matrix=False,
    )


def build_backbone(
    model: str,
    config: dict[str, Any],
    required_irreps: Any,
    device: Any,
) -> Any:
    from maloq.helm.esen_osh import eSEN_Backbone
    from maloq.helm.qhflow3_clean import QHFlow3MaloqBackbone

    if model == "qhflow3":
        backbone = QHFlow3MaloqBackbone(
            sh_lmax=required_irreps.lmax,
            hidden_size=config["l_embedding_dim"],
            bottle_hidden_size=config["output_l_embedding_dim"],
            num_gnn_layers=config["num_mp_layers"],
            num_ham_gnn_layers=(
                config["num_mp_layers"]
                if config["num_edge_layers"] is None
                else config["num_edge_layers"]
            ),
            max_radius=config["qhflow3_max_radius"],
            radius_embed_dim=config["qhflow3_radius_embed_dim"],
            escn_edge_channels=config["hidden_dim"],
            escn_num_distance_basis=config["num_distance_basis"],
            esen_max_radius=config["rcut_gaussian"],
            grid_resolution=config["qhflow3_grid_resolution"],
            grid_ffn_chunk_size=config["qhflow3_grid_ffn_chunk_size"],
            basis="def2-svp-nabla",
            delta_learning=False,
            delta_target="fock_matrix",
            default_hamiltonian_input="zero",
            use_block_S=config["qhflow3_use_overlap"],
            use_block_H=False,
            muonize_output_projection=config[
                "qhflow3_muonize_output_projection"
            ],
        )
    else:
        backbone = eSEN_Backbone(
            required_irreps,
            sphere_channels=config["l_embedding_dim"],
            hidden_channels=config["hidden_dim"],
            lmax=required_irreps.lmax,
            mmax=required_irreps.lmax,
            cutoff=config["rcut_gaussian"],
            grid_resolution=config["esen_grid_resolution"],
            edge_channels=config["l_embedding_dim"],
            num_layers=config["num_mp_layers"],
            act_type="gate",
            mlp_type=config["mlp_type"],
            gate_act_type=config["gate_act_type"],
            num_distance_basis=config["num_distance_basis"],
            gaussian_width=config["gaussian_width"],
            # TrainingWorkflow derives this from the matrix target before model
            # construction; feature analysis always uses the same edge target.
            include_edges=config.get("include_edges", True),
            open_shell=config["open_shell"],
            wigner_backend=config["wigner_backend"],
            distributed_graph_training=False,
            message_type=config.get("message_type", "source-target"),
            message_passing_schedule=config["message_passing_schedule"],
            initial_edge_state_mode=config.get(
                "initial_edge_state_mode",
                "edge_degree",
            ),
            num_edge_layers=config["num_edge_layers"],
            output_sphere_channels=config["output_l_embedding_dim"],
            nte_output_projection_mode=config[
                "nte_output_projection_mode"
            ],
            output_norm_sharing=config.get(
                "output_norm_sharing",
                "shared",
            ),
            use_edge_envelope=config["use_edge_envelope"],
            use_edge_scalar_modulation=config["use_edge_scalar_modulation"],
            residual_update_scale_mode=config["residual_update_scale_mode"],
            residual_update_scale_init=config["residual_update_scale_init"],
            residual_update_scale_log_range=config[
                "residual_update_scale_log_range"
            ],
            unscaled_node_layers=config["unscaled_node_layers"],
            repeat_system_embedding_each_node_block=config[
                "repeat_system_embedding_each_node_block"
            ],
            node_stack_mode=config["node_stack_mode"],
            edge_stack_mode=config["edge_stack_mode"],
            qhflow3_layer_gaussian_width=config[
                "qhflow3_layer_gaussian_width"
            ],
            qhflow3_layer_grid_ffn_chunk_size=config[
                "qhflow3_layer_grid_ffn_chunk_size"
            ],
            qhflow3_exact_pair_rng_aligned=config[
                "qhflow3_exact_pair_rng_aligned"
            ],
            edge_atom_norm_type=config["edge_atom_norm_type"],
            edge_post_residual_norm_type=config[
                "edge_post_residual_norm_type"
            ],
            direct_edgewise_layers=config["direct_edgewise_layers"],
            edge_atomwise_output_mode=config["edge_atomwise_output_mode"],
            edge_norm1_position=config["edge_norm1_position"],
            input_conditioning=config["nte_input_conditioning"],
            conditioning_basis="def2-svp-nabla",
            conditioning_delta_learning=False,
            conditioning_delta_target="fock_matrix",
        )
    return backbone.to(device)


def build_head(
    config: dict[str, Any],
    required_irreps: Any,
    orbital_basis: Any,
    ls_list: Any,
    device: Any,
) -> Any:
    from e3nn.o3 import Irreps
    from maloq.helm.muon_fock_head import MuonFockIrrepsHead

    channels = int(config["output_l_embedding_dim"])
    irreps_in = Irreps(
        [(channels, (degree, 1)) for degree in range(required_irreps.lmax + 1)]
    )
    return MuonFockIrrepsHead(
        irreps_in=irreps_in,
        irreps_out=required_irreps,
        lmax=required_irreps.lmax,
        sphere_channels=channels,
        reduce_edge=config["reduce_edge"],
        open_shell=config["open_shell"],
        ls_list=ls_list,
        reduce_node=config["reduce_node"],
        reduce_node_intra=config["reduce_node_intra"],
        orbital_basis=orbital_basis,
    ).to(device)


def _strict_load_state(
    module: Any,
    state: dict[str, Any],
    source: Path,
) -> None:
    incompat = module.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch for {source}: {incompat.missing_keys}, "
            f"{incompat.unexpected_keys}"
        )


def load_checkpoint(module: Any, path: Path) -> dict[str, Any]:
    """Load one legacy state-dict file."""
    payload, state = _load_legacy_state_payload(path)
    _strict_load_state(module, state, path)
    return {
        "path": str(path.resolve()),
        "format": "legacy",
        "sha256": _sha256_file(path),
        "epoch": payload.get("epoch"),
        "keys": len(state),
    }


def load_run_checkpoints(
    backbone: Any,
    head: Any,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore a fixed workflow bundle, with legacy pair fallback."""
    from maloq.train_utils.training_workflow_fixed import (
        checkpoint_candidates,
        load_training_checkpoint,
    )

    run_dir = run_dir.expanduser().resolve()
    fixed_candidates = checkpoint_candidates(run_dir)
    fixed_present = any(path.is_file() for path in fixed_candidates)
    fixed_error: str | None = None
    if fixed_present:
        try:
            payload, selected = load_training_checkpoint(run_dir)
        except RuntimeError as exc:
            fixed_error = str(exc)
        else:
            backbone_state = payload["backbone_state_dict"]
            head_state = payload["head_state_dict"]
            _strict_load_state(backbone, backbone_state, selected)
            _strict_load_state(head, head_state, selected)
            common = {
                "path": str(selected),
                "format": "fixed",
                "sha256": _sha256_file(selected),
                "epoch": payload["completed_epoch"],
                "schema_version": payload["schema_version"],
            }
            return (
                {
                    **common,
                    "component": "backbone",
                    "keys": len(backbone_state),
                },
                {
                    **common,
                    "component": "head",
                    "keys": len(head_state),
                },
            )

    backbone_path, head_path = _legacy_checkpoint_paths(run_dir)
    if backbone_path.is_file() and head_path.is_file():
        backbone_meta = load_checkpoint(backbone, backbone_path)
        head_meta = load_checkpoint(head, head_path)
        if fixed_error is not None:
            backbone_meta["fixed_checkpoint_error"] = fixed_error
            head_meta["fixed_checkpoint_error"] = fixed_error
        return backbone_meta, head_meta

    details = (
        f"\nFixed checkpoint validation failed:\n{fixed_error}"
        if fixed_error is not None
        else ""
    )
    raise FileNotFoundError(
        f"No usable checkpoint bundle in {run_dir}; expected a valid fixed "
        "training state or both legacy state dictionaries."
        + details
    )


def _native_output(value: Any) -> Any:
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"Ambiguous hook output tuple of length {len(value)}.")
        value = value[0]
    return value


def register_feature_hooks(
    model: str,
    backbone: Any,
    head: Any,
    collector: FeatureCollector,
    lmax: int,
) -> list[Any]:
    handles = []

    def native_output_hook(kind: str, stage: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            collector.add_native(
                model, kind, stage, _native_output(output), lmax=lmax
            )

        return hook

    def native_pre_hook(kind: str, stage: str, input_index: int = 0):
        def hook(_module: Any, inputs: Any) -> None:
            collector.add_native(
                model, kind, stage, inputs[input_index], lmax=lmax
            )

        return hook

    def flat_pre_hook(kind: str, stage: str, channels: int):
        def hook(_module: Any, inputs: Any) -> None:
            collector.add_degree_major_flat(
                model,
                kind,
                stage,
                inputs[0],
                lmax=lmax,
                channels=channels,
            )

        return hook

    def flat_output_hook(kind: str, stage: str, channels: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            collector.add_degree_major_flat(
                model,
                kind,
                stage,
                _native_output(output),
                lmax=lmax,
                channels=channels,
            )

        return hook

    if model == "nte":
        handles.append(
            backbone.node_blocks[0].register_forward_pre_hook(
                native_pre_hook("node", "input_after_edge_degree", 0)
            )
        )
        for index, block in enumerate(backbone.node_blocks, start=1):
            handles.append(
                block.edge_wise.register_forward_hook(
                    native_output_hook("node", f"node_{index}_edgewise_raw")
                )
            )
            handles.append(
                block.edge_update_scale.register_forward_hook(
                    native_output_hook("node", f"node_{index}_edge_update")
                )
            )
            handles.append(
                block.atom_wise.register_forward_hook(
                    native_output_hook("node", f"node_{index}_atomwise_raw")
                )
            )
            handles.append(
                block.atom_update_scale.register_forward_hook(
                    native_output_hook("node", f"node_{index}_atom_update")
                )
            )
            handles.append(
                block.register_forward_hook(
                    native_output_hook("node", f"node_block_{index}")
                )
            )
        handles.append(
            backbone.edge_blocks[0].register_forward_pre_hook(
                native_pre_hook("edge", "input_before_edge_stack", 1)
            )
        )
        for index, block in enumerate(backbone.edge_blocks, start=1):
            handles.append(
                block.edge_wise.register_forward_hook(
                    native_output_hook("edge", f"edge_{index}_edgewise_raw")
                )
            )
            handles.append(
                block.edge_update_scale.register_forward_hook(
                    native_output_hook("edge", f"edge_{index}_edge_update")
                )
            )
            handles.append(
                block.atom_wise.register_forward_hook(
                    native_output_hook("edge", f"edge_{index}_atomwise_raw")
                )
            )
            handles.append(
                block.atom_update_scale.register_forward_hook(
                    native_output_hook("edge", f"edge_{index}_atom_update")
                )
            )
            handles.append(
                block.register_forward_hook(
                    native_output_hook("edge", f"edge_block_{index}")
                )
            )
        for kind in ("node", "edge"):
            projection = getattr(backbone, f"{kind}_output_projection")
            handles.append(
                projection.register_forward_pre_hook(
                    native_pre_hook(kind, "pre_output_projection")
                )
            )
            handles.append(
                projection.register_forward_hook(
                    native_output_hook(kind, "output_64")
                )
            )
    else:
        qhf = backbone.node_attr_backbone
        handles.append(
            qhf.blocks[0].register_forward_pre_hook(
                native_pre_hook("node", "input_after_edge_degree", 0)
            )
        )
        for index, block in enumerate(qhf.blocks, start=1):
            handles.append(
                block.edge_wise.register_forward_hook(
                    native_output_hook("node", f"node_{index}_edgewise_update")
                )
            )
            handles.append(
                block.atom_wise.register_forward_hook(
                    native_output_hook("node", f"node_{index}_atomwise_update")
                )
            )
            handles.append(
                block.register_forward_hook(
                    native_output_hook("node", f"node_block_{index}")
                )
            )
        handles.append(
            qhf.norm.register_forward_hook(
                native_output_hook("node", "node_normalized")
            )
        )
        for index, block in enumerate(qhf.xy_blocks, start=1):
            def pair_edgewise_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                captured_index: int = index,
            ) -> None:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise ValueError(
                        "QHFlow3 pair Edgewise hook expected (node, pair) output."
                    )
                collector.add_native(
                    model,
                    "edge",
                    f"pair_{captured_index}_edgewise_raw",
                    output[1],
                    lmax=lmax,
                )

            handles.append(block.edge_wise.register_forward_hook(pair_edgewise_hook))
            handles.append(
                block.atom_wise.register_forward_hook(
                    native_output_hook("edge", f"pair_{index}_atomwise_update")
                )
            )
            handles.append(
                block.register_forward_hook(
                    native_output_hook("edge", f"pair_block_{index}_raw")
                )
            )
        handles.append(
            qhf.xy_norm.register_forward_pre_hook(
                native_pre_hook("edge", "pair_sum_pre_norm")
            )
        )
        handles.append(
            qhf.xy_norm.register_forward_hook(
                native_output_hook("edge", "pair_normalized")
            )
        )
        for kind, projection in (
            ("node", backbone.output_ii),
            ("edge", backbone.output_ij),
        ):
            handles.append(
                projection.register_forward_pre_hook(
                    flat_pre_hook(kind, "pre_output_projection", 128)
                )
            )
            handles.append(
                projection.register_forward_hook(
                    flat_output_hook(kind, "output_64", 64)
                )
            )

    for kind, semantic in (
        ("node", head.node_semantic_layers[0]),
        ("edge", head.edge_semantic_layers[0]),
    ):
        handles.append(
            semantic.register_forward_pre_hook(
                flat_pre_hook(kind, "head_gated_64", 64)
            )
        )

        def semantic_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            captured_kind: str = kind,
            captured_semantic: Any = semantic,
        ) -> None:
            collector.add_semantic_output(
                model,
                captured_kind,
                "head_semantic_output",
                _native_output(output),
                captured_semantic,
            )

        handles.append(semantic.register_forward_hook(semantic_hook))
    return handles


def matrix_spectrum(
    model: str,
    family: str,
    kind: str,
    degree: int,
    matrix: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import torch

    weight = matrix.detach().cpu().to(torch.float64)
    singular = torch.linalg.svdvals(weight).numpy()
    squared = np.square(singular)
    spectral = float(singular[0]) if singular.size else 0.0
    frobenius = float(np.sqrt(squared.sum()))
    stable_rank = (
        float(squared.sum() / (spectral**2)) if spectral > 0.0 else 0.0
    )
    singular_sum = float(singular.sum())
    if singular_sum > 0.0:
        probability = singular / singular_sum
        effective_rank = float(np.exp(-(probability * np.log(probability)).sum()))
    else:
        effective_rank = 0.0
    positive = singular[singular > max(spectral * 1.0e-8, 1.0e-12)]
    condition = (
        float(spectral / positive[-1])
        if positive.size and positive.size == singular.size
        else math.inf
    )
    summary = {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "family": family,
        "feature_kind": kind,
        "degree": degree,
        "output_channels": int(weight.shape[0]),
        "input_channels": int(weight.shape[1]),
        "output_input_ratio": float(weight.shape[0] / weight.shape[1]),
        "frobenius_norm": frobenius,
        "spectral_norm": spectral,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "condition_number": condition,
        "min_singular_value": float(singular[-1]) if singular.size else 0.0,
        "max_singular_value": spectral,
    }
    rows = [
        {
            "model": model,
            "family": family,
            "feature_kind": kind,
            "degree": degree,
            "singular_index": index,
            "singular_value": float(value),
        }
        for index, value in enumerate(singular.tolist())
    ]
    return summary, rows


def effective_projection_matrices(
    projection: Any,
    lmax: int,
) -> dict[int, Any]:
    """Return exact output-by-input matrices used for every degree.

    Native ``SO3_Linear`` weights already are the effective matrices.  e3nn
    projections additionally multiply every instruction by ``path_weight``;
    this applies both to ordinary ``e3nn.o3.Linear`` and the Muon-visible
    wrappers used by QHFlow3 and NTE's QHFProj ablation.
    """
    import torch

    weight = getattr(projection, "weight", None)
    if (
        weight is not None
        and weight.ndim == 3
        and int(weight.shape[0]) == lmax + 1
        and not hasattr(projection, "linear")
    ):
        return {degree: weight[degree] for degree in range(lmax + 1)}

    # QHFlow3IrrepLinear owns a MuonVisibleIrrepLinear in ``linear``; a native
    # QHFlow3 projection is itself the Muon-visible wrapper.
    wrapper = projection
    child = getattr(projection, "linear", None)
    if (
        child is not None
        and hasattr(child, "weight")
        and hasattr(child, "linear")
        and hasattr(child.linear, "instructions")
    ):
        wrapper = child

    external = getattr(wrapper, "linear", None)
    wrapper_weight = getattr(wrapper, "weight", None)
    if (
        external is not None
        and hasattr(external, "instructions")
        and wrapper_weight is not None
        and wrapper_weight.ndim == 3
    ):
        matrices: dict[int, Any] = {}
        path_index = 0
        for instruction in external.instructions:
            if instruction.i_in < 0:
                continue
            _, ir_in = external.irreps_in[instruction.i_in]
            _, ir_out = external.irreps_out[instruction.i_out]
            matrix = wrapper_weight[path_index]
            path_index += 1
            if ir_in.l != ir_out.l:
                continue
            if ir_in.l in matrices:
                raise RuntimeError(
                    "Projection has multiple weighted paths for degree "
                    f"{ir_in.l}; a single channel-contraction matrix is "
                    "not well-defined."
                )
            matrices[ir_in.l] = matrix * instruction.path_weight
        if path_index != int(wrapper_weight.shape[0]):
            raise RuntimeError(
                "Muon-visible projection path count does not match its weight "
                f"tensor: {path_index} != {wrapper_weight.shape[0]}."
            )
    elif hasattr(projection, "instructions"):
        matrices = {}
        for instruction_index, instruction in enumerate(
            projection.instructions
        ):
            if instruction.i_in < 0:
                continue
            _, ir_in = projection.irreps_in[instruction.i_in]
            _, ir_out = projection.irreps_out[instruction.i_out]
            if ir_in.l != ir_out.l:
                continue
            if ir_in.l in matrices:
                raise RuntimeError(
                    "Projection has multiple weighted paths for degree "
                    f"{ir_in.l}; a single channel-contraction matrix is "
                    "not well-defined."
                )
            view = projection.weight_view_for_instruction(instruction_index)
            matrices[ir_in.l] = view.T * instruction.path_weight
    elif (
        weight is not None
        and weight.ndim == 3
        and int(weight.shape[0]) == lmax + 1
    ):
        matrices = {
            degree: weight[degree] for degree in range(lmax + 1)
        }
    else:
        raise TypeError(
            "Unsupported output projection type for contraction analysis: "
            f"{type(projection).__module__}.{type(projection).__qualname__}."
        )

    expected_degrees = set(range(lmax + 1))
    if set(matrices) != expected_degrees:
        raise RuntimeError(
            "Could not map every projection degree: "
            f"expected {sorted(expected_degrees)}, got {sorted(matrices)}."
        )
    if not all(
        isinstance(matrix, torch.Tensor) and matrix.ndim == 2
        for matrix in matrices.values()
    ):
        raise RuntimeError("Every effective projection matrix must be rank 2.")
    return matrices


def collect_contractions(
    model: str,
    backbone: Any,
    head: Any,
    lmax: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []

    projections = (
        (
            ("node", backbone.node_output_projection),
            ("edge", backbone.edge_output_projection),
        )
        if model == "nte"
        else (
            ("node", backbone.output_ii),
            ("edge", backbone.output_ij),
        )
    )
    for kind, projection in projections:
        degree_matrices = effective_projection_matrices(projection, lmax)
        for degree, matrix in sorted(degree_matrices.items()):
            summary, rows = matrix_spectrum(
                model,
                "backbone_128_to_64",
                kind,
                degree,
                matrix,
            )
            summaries.append(summary)
            singular_rows.extend(rows)

    if model == "nte":
        for kind, blocks in (
            ("node", backbone.node_blocks),
            ("edge", backbone.edge_blocks),
        ):
            for block_index, block in enumerate(blocks, start=1):
                for branch in ("edge_update_scale", "atom_update_scale"):
                    scales = getattr(block, branch).degree_scales()
                    if scales is None:
                        continue
                    for degree, scale in enumerate(scales.detach().cpu().tolist()):
                        residual_rows.append(
                            {
                                "model": model,
                                "feature_kind": kind,
                                "block": block_index,
                                "branch": branch,
                                "degree": degree,
                                "scale": float(scale),
                            }
                        )
    for kind, semantic in (
        ("node", head.node_semantic_layers[0]),
        ("edge", head.edge_semantic_layers[0]),
    ):
        for layer_index, degree in enumerate(semantic._degrees):
            rows_index = getattr(semantic, f"_rows_l{degree}")
            external = semantic.external_layers[layer_index]
            weighted_instructions = [
                instruction
                for instruction in external.instructions
                if instruction.i_in >= 0
            ]
            if len(weighted_instructions) != 1:
                raise RuntimeError(
                    f"Expected one semantic e3nn path for degree {degree}."
                )
            matrix = (
                semantic.weight.index_select(0, rows_index)
                * weighted_instructions[0].path_weight
            )
            summary, rows = matrix_spectrum(
                model,
                "muon_head_semantic",
                kind,
                degree,
                matrix,
            )
            summaries.append(summary)
            singular_rows.extend(rows)
    return summaries, singular_rows, residual_rows


def run_model(
    model: str,
    config: dict[str, Any],
    run_dir: Path,
    loader: Any,
    required_irreps: Any,
    orbital_basis: Any,
    ls_list: Any,
    collector: FeatureCollector,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    started = time.perf_counter()
    backbone = build_backbone(model, config, required_irreps, device)
    head = build_head(config, required_irreps, orbital_basis, ls_list, device)
    backbone_meta, head_meta = load_run_checkpoints(
        backbone,
        head,
        run_dir,
    )
    backbone.eval()
    head.eval()
    handles = register_feature_hooks(
        model, backbone, head, collector, required_irreps.lmax
    )
    batches = 0
    nodes = 0
    edges = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device)
                nodes += int(batch.pos.shape[0])
                edges += int(batch.edge_index.reshape(2, -1).shape[1])
                embeddings = backbone(batch)
                head(embeddings, batch)
                batches += 1
                del embeddings, batch
    finally:
        for handle in handles:
            handle.remove()

    contraction, singular, residual = collect_contractions(
        model, backbone, head, required_irreps.lmax
    )
    provenance = {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "config": str(
            (DEFAULT_QHF_CONFIG if model == "qhflow3" else DEFAULT_NTE_CONFIG)
            .resolve()
        ),
        "run_dir": str(run_dir.resolve()),
        "backbone_checkpoint": backbone_meta,
        "head_checkpoint": head_meta,
        "batches": batches,
        "nodes": nodes,
        "directed_edges": edges,
        "elapsed_seconds": time.perf_counter() - started,
        "parameters": {
            "backbone": sum(parameter.numel() for parameter in backbone.parameters()),
            "head": sum(parameter.numel() for parameter in head.parameters()),
        },
    }
    del backbone, head
    torch.cuda.empty_cache()
    return provenance, contraction, singular, residual


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_features(
    output_dir: Path,
    degree_rows: list[dict[str, Any]],
    models: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        (
            row["model"],
            row["feature_kind"],
            row["stage"],
            int(row["degree"]),
        ): row
        for row in degree_rows
    }
    figure, axes = plt.subplots(
        len(models),
        2,
        figsize=(15, 5.2 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, model in enumerate(models):
        for col_index, kind in enumerate(("node", "edge")):
            axis = axes[row_index][col_index]
            stages = [
                stage
                for stage in FEATURE_STAGE_ORDER[model][kind]
                if (model, kind, stage, 0) in lookup
            ]
            for degree in range(5):
                values = [
                    lookup[(model, kind, stage, degree)]["rms"]
                    for stage in stages
                    if (model, kind, stage, degree) in lookup
                ]
                matching_stages = [
                    stage
                    for stage in stages
                    if (model, kind, stage, degree) in lookup
                ]
                axis.plot(
                    matching_stages,
                    values,
                    marker="o",
                    linewidth=1.8,
                    label=f"l={degree}",
                )
            axis.set_yscale("log")
            axis.set_ylabel("component-normalized RMS")
            axis.set_title(f"{model.upper()} {kind}")
            axis.tick_params(axis="x", rotation=35)
            axis.grid(True, alpha=0.25)
            axis.legend(ncol=5, fontsize=8)
    figure.savefig(output_dir / "layer_degree_rms.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(models),
        2,
        figsize=(15, 5.2 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, model in enumerate(models):
        for col_index, kind in enumerate(("node", "edge")):
            axis = axes[row_index][col_index]
            stages = [
                stage
                for stage in FEATURE_STAGE_ORDER[model][kind]
                if (model, kind, stage, 0) in lookup
            ]
            for degree in range(5):
                values = [
                    lookup[(model, kind, stage, degree)][
                        "effective_channel_fraction"
                    ]
                    for stage in stages
                    if (model, kind, stage, degree) in lookup
                ]
                matching_stages = [
                    stage
                    for stage in stages
                    if (model, kind, stage, degree) in lookup
                ]
                axis.plot(
                    matching_stages,
                    values,
                    marker="o",
                    linewidth=1.8,
                    label=f"l={degree}",
                )
            axis.set_ylim(0.0, 1.05)
            axis.set_ylabel("effective channels / channels")
            axis.set_title(f"{model.upper()} {kind}")
            axis.tick_params(axis="x", rotation=35)
            axis.grid(True, alpha=0.25)
            axis.legend(ncol=5, fontsize=8)
    figure.savefig(output_dir / "layer_channel_participation.png", dpi=180)
    plt.close(figure)


def plot_contractions(
    output_dir: Path,
    contraction_rows: list[dict[str, Any]],
    models: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(
        len(models),
        2,
        figsize=(12, 4.7 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    width = 0.36
    degrees = np.arange(5)
    for row_index, model in enumerate(models):
        for col_index, kind in enumerate(("node", "edge")):
            axis = axes[row_index][col_index]
            for family_index, family in enumerate(
                ("backbone_128_to_64", "muon_head_semantic")
            ):
                subset = {
                    int(row["degree"]): row
                    for row in contraction_rows
                    if row["model"] == model
                    and row["feature_kind"] == kind
                    and row["family"] == family
                }
                values = [subset[degree]["stable_rank"] for degree in degrees]
                axis.bar(
                    degrees + (family_index - 0.5) * width,
                    values,
                    width=width,
                    label=family,
                )
            axis.set_xticks(degrees)
            axis.set_xlabel("degree l")
            axis.set_ylabel("stable rank")
            axis.set_title(f"{model.upper()} {kind} contraction")
            axis.grid(True, axis="y", alpha=0.25)
            axis.legend(fontsize=8)
    figure.savefig(output_dir / "contraction_stable_rank.png", dpi=180)
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    formatted = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    formatted.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )
    return "\n".join(formatted)


def write_report(
    output_dir: Path,
    degree_rows: list[dict[str, Any]],
    contraction_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    models: list[str],
    num_molecules: int,
) -> None:
    lookup = {
        (
            row["model"],
            row["feature_kind"],
            row["stage"],
            int(row["degree"]),
        ): row
        for row in degree_rows
    }
    lines = [
        "# NablaDFT QHFlow3 vs NTE layer feature analysis",
        "",
        f"Same held-out NablaDFT rows: `12081:{12081 + num_molecules}`.",
        "All RMS values are normalized per item, magnetic component, and channel.",
        "Power fractions retain the total `(2l+1)` degree contribution.",
        "",
        "## Interpretation boundary",
        "",
        "NTE EdgeBlock 2 consumes EdgeBlock 1's state. QHFlow3 PairBlock 1 and "
        "PairBlock 2 both consume the final node state independently; their raw "
        "outputs are added and only then normalized. The pair curves are therefore "
        "architectural branches, not recurrent edge states.",
        "",
        "## Inputs",
        "",
    ]
    input_rows = [
        [
            item["model"],
            item["batches"],
            item["nodes"],
            item["directed_edges"],
            f"{item['elapsed_seconds']:.2f}",
            item["parameters"]["backbone"],
            item["parameters"]["head"],
        ]
        for item in provenance
    ]
    lines.extend(
        [
            markdown_table(
                [
                    "model",
                    "batches",
                    "nodes",
                    "directed edges",
                    "seconds",
                    "backbone params",
                    "head params",
                ],
                input_rows,
            ),
            "",
        ]
    )
    if set(models) == {"qhflow3", "nte"}:
        qhf_l3_ratio = (
            lookup[("qhflow3", "node", "node_block_3", 3)]["rms"]
            / lookup[("nte", "node", "node_block_3", 3)]["rms"]
        )
        qhf_l4_ratio = (
            lookup[("qhflow3", "node", "node_block_3", 4)]["rms"]
            / lookup[("nte", "node", "node_block_3", 4)]["rms"]
        )
        qhf_l4_power = 100.0 * lookup[
            ("qhflow3", "node", "node_block_3", 4)
        ]["power_fraction"]
        nte_l4_power = 100.0 * lookup[
            ("nte", "node", "node_block_3", 4)
        ]["power_fraction"]
        node_output_ratios = [
            lookup[("qhflow3", "node", "output_64", degree)]["rms"]
            / lookup[("nte", "node", "output_64", degree)]["rms"]
            for degree in range(5)
        ]
        edge_output_ratios = [
            lookup[("qhflow3", "edge", "output_64", degree)]["rms"]
            / lookup[("nte", "edge", "output_64", degree)]["rms"]
            for degree in range(5)
        ]
        pair_branch_ratios = [
            lookup[("qhflow3", "edge", "pair_block_2_raw", degree)]["rms"]
            / lookup[("qhflow3", "edge", "pair_block_1_raw", degree)]["rms"]
            for degree in range(5)
        ]
        qhf_node2_l4_relative_update = (
            lookup[("qhflow3", "node", "node_2_edgewise_update", 4)]["rms"]
            / lookup[("qhflow3", "node", "node_block_1", 4)]["rms"]
        )
        nte_node2_l4_relative_update = (
            lookup[("nte", "node", "node_2_edge_update", 4)]["rms"]
            / lookup[("nte", "node", "node_block_1", 4)]["rms"]
        )
        qhf_node3_l4_relative_update = (
            lookup[("qhflow3", "node", "node_3_edgewise_update", 4)]["rms"]
            / lookup[("qhflow3", "node", "node_block_2", 4)]["rms"]
        )
        nte_node3_l4_relative_update = (
            lookup[("nte", "node", "node_3_edge_update", 4)]["rms"]
            / lookup[("nte", "node", "node_block_2", 4)]["rms"]
        )
        nte_node2_l4_scale = (
            lookup[("nte", "node", "node_2_edge_update", 4)]["rms"]
            / lookup[("nte", "node", "node_2_edgewise_raw", 4)]["rms"]
        )
        nte_edge2_scalar_update = lookup[
            ("nte", "edge", "edge_2_atom_update", 0)
        ]["rms"]
        nte_edge2_l4_update = lookup[
            ("nte", "edge", "edge_2_atom_update", 4)
        ]["rms"]
        nte_edge_l0_growth = (
            lookup[("nte", "edge", "edge_block_2", 0)]["rms"]
            / lookup[("nte", "edge", "edge_block_1", 0)]["rms"]
        )
        nte_edge_l4_growth = (
            lookup[("nte", "edge", "edge_block_2", 4)]["rms"]
            / lookup[("nte", "edge", "edge_block_1", 4)]["rms"]
        )
        nte_node_l1_fraction = lookup[
            ("nte", "node", "output_64", 1)
        ]["effective_channel_fraction"]
        qhf_node_l1_fraction = lookup[
            ("qhflow3", "node", "output_64", 1)
        ]["effective_channel_fraction"]
        nte_edge_l4_fraction = lookup[
            ("nte", "edge", "output_64", 4)
        ]["effective_channel_fraction"]
        qhf_edge_l4_fraction = lookup[
            ("qhflow3", "edge", "output_64", 4)
        ]["effective_channel_fraction"]
        contraction_lookup = {
            (
                row["model"],
                row["family"],
                row["feature_kind"],
                int(row["degree"]),
            ): row
            for row in contraction_rows
        }
        rank_gaps: dict[str, list[float]] = defaultdict(list)
        for family in ("backbone_128_to_64", "muon_head_semantic"):
            for kind in ("node", "edge"):
                for degree in range(5):
                    rank_gaps[family].append(
                        abs(
                            contraction_lookup[
                                ("qhflow3", family, kind, degree)
                            ]["stable_rank"]
                            - contraction_lookup[
                                ("nte", family, kind, degree)
                            ]["stable_rank"]
                        )
                    )
        lines.extend(
            [
                "## Main findings",
                "",
                f"- Before the final norm, QHFlow3 NodeBlock 3 is "
                f"`{qhf_l3_ratio:.1f}×` NTE at `l=3` and "
                f"`{qhf_l4_ratio:.1f}×` at `l=4`. Its `l=4` branch carries "
                f"`{qhf_l4_power:.1f}%` of raw node power versus "
                f"`{nte_l4_power:.1f}%` for NTE. Final RMS normalization "
                "returns both trunks to order-one scale, so this is internal "
                "dynamic range rather than an output explosion.",
                f"- After the 128→64 node projection, QHFlow3/NTE RMS ratios "
                f"for `l=0..4` are "
                f"`{', '.join(f'{value:.2f}' for value in node_output_ratios)}`. "
                "QHFlow3 shifts the delivered node representation toward "
                "`l=3,4`, while NTE is stronger at `l=1,2`.",
                f"- After the 128→64 edge projection, QHFlow3/NTE RMS ratios "
                f"for `l=0..4` are "
                f"`{', '.join(f'{value:.2f}' for value in edge_output_ratios)}`. "
                "The largest separation is again the high-degree edge signal.",
                f"- QHFlow3 PairBlock 2 has only "
                f"`{min(pair_branch_ratios):.2f}–{max(pair_branch_ratios):.2f}×` "
                "the RMS of PairBlock 1. The first independent pair branch "
                "dominates the residual sum in this checkpoint.",
                f"- The largest node-stack breakpoint is Block 2 at `l=4`: "
                f"QHFlow3's edgewise update is "
                f"`{qhf_node2_l4_relative_update:.2f}×` its Block-1 state, "
                f"whereas NTE's effective update is only "
                f"`{nte_node2_l4_relative_update:.2f}×`. NTE's degree scale "
                f"reduces that raw Block-2 update to "
                f"`{nte_node2_l4_scale:.4f}×`. Block 3 is not the main "
                f"separation: the corresponding ratios converge to "
                f"`{qhf_node3_l4_relative_update:.2f}×` and "
                f"`{nte_node3_l4_relative_update:.2f}×`.",
                f"- NTE's recurrent EdgeBlock 2 is strongly scalar-biased. Its "
                f"atomwise residual contributes `{nte_edge2_scalar_update:.3g}` "
                f"RMS at `l=0` but only `{nte_edge2_l4_update:.3g}` at `l=4`; "
                f"the complete edge state grows `{nte_edge_l0_growth:.2f}×` at "
                f"`l=0` but `{nte_edge_l4_growth:.2f}×` at `l=4`. QHFlow3 "
                "instead computes two independent normalized pair branches and "
                "sums them, so pair topology is a higher-priority ablation than "
                "adding more conditioning.",
                f"- Channel participation also separates the trunks: node "
                f"`l=1` after projection uses `{qhf_node_l1_fraction:.1%}` of "
                f"QHFlow3 channels effectively versus `{nte_node_l1_fraction:.1%}` "
                f"for NTE; edge `l=4` uses `{qhf_edge_l4_fraction:.1%}` versus "
                f"`{nte_edge_l4_fraction:.1%}`. Channel indices themselves are "
                "not aligned between independently trained models.",
                f"- Mean absolute stable-rank gap is "
                f"`{sum(rank_gaps['backbone_128_to_64']) / len(rank_gaps['backbone_128_to_64']):.2f}` "
                f"in the backbone projection but only "
                f"`{sum(rank_gaps['muon_head_semantic']) / len(rank_gaps['muon_head_semantic']):.2f}` "
                "in the common Muon semantic head. The dominant contraction "
                "difference is therefore before the shared head.",
                "",
            ]
        )
    lines.extend(["## Final 128→64 activation gain", ""])
    projection_rows = []
    for model in models:
        for kind in ("node", "edge"):
            for degree in range(5):
                before = lookup[
                    (model, kind, "pre_output_projection", degree)
                ]["rms"]
                after = lookup[(model, kind, "output_64", degree)]["rms"]
                projection_rows.append(
                    [
                        model,
                        kind,
                        degree,
                        f"{before:.5g}",
                        f"{after:.5g}",
                        f"{after / before:.4f}" if before else "inf",
                    ]
                )
    lines.extend(
        [
            markdown_table(
                ["model", "feature", "l", "pre RMS", "post RMS", "gain"],
                projection_rows,
            ),
            "",
            "## Channel contraction matrices",
            "",
        ]
    )
    matrix_rows = []
    for row in contraction_rows:
        matrix_rows.append(
            [
                row["model"],
                row["family"],
                row["feature_kind"],
                row["degree"],
                f"{row['output_channels']}×{row['input_channels']}",
                f"{row['spectral_norm']:.4g}",
                f"{row['stable_rank']:.3f}",
                f"{row['effective_rank']:.3f}",
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "model",
                    "matrix",
                    "feature",
                    "l",
                    "shape",
                    "spectral norm",
                    "stable rank",
                    "effective rank",
                ],
                matrix_rows,
            ),
            "",
        ]
    )
    if residual_rows:
        lines.extend(
            [
                "## NTE learned residual update scales",
                "",
                "These bounded degree scales have no direct QHFlow3 counterpart.",
                "",
                markdown_table(
                    ["feature", "block", "branch", "l", "scale"],
                    [
                        [
                            row["feature_kind"],
                            row["block"],
                            row["branch"],
                            row["degree"],
                            f"{row['scale']:.6g}",
                        ]
                        for row in residual_rows
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Files",
            "",
            "- `layer_degree_stats.csv`: degree RMS, power fraction, and channel summaries.",
            "- `layer_channel_rms.csv`: every channel's aggregated RMS.",
            "- `contraction_stats.csv`: backbone and Muon-head matrix spectra.",
            "- `contraction_singular_values.csv`: full singular-value spectra.",
            "- `residual_degree_scales.csv`: learned NTE residual scales.",
            "- `layer_degree_rms.png`: layer-wise degree scale.",
            "- `layer_channel_participation.png`: channel collapse/dispersion view.",
            "- `contraction_stable_rank.png`: contraction rank comparison.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    MODEL_LABELS["qhflow3"] = args.qhflow3_label
    MODEL_LABELS["nte"] = args.nte_label
    models = (
        ["qhflow3", "nte"]
        if args.models == "both"
        else [args.models]
    )
    checkpoint_validation = validate_inputs(args, models)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "models": models,
                    "num_molecules": args.num_molecules,
                    "output_dir": str(args.output_dir.resolve()),
                    "checkpoints": checkpoint_validation,
                },
                indent=2,
            )
        )
        return

    import torch
    from maloq.train_utils import utils_compute

    if not torch.cuda.is_available():
        raise RuntimeError("This analysis requires one CUDA GPU.")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)
    device = utils_compute.setup_env(0, 1, backend="nccl", local_rank=0)
    config_paths = {
        "qhflow3": args.qhflow3_config,
        "nte": args.nte_config,
    }
    configs = {
        model: load_config(config_paths[model])
        for model in models
    }
    reference_config = configs[models[0]]
    loader, required_irreps, _, orbital_basis, ls_list = make_loader(
        args, reference_config
    )
    collector = FeatureCollector()
    provenance: list[dict[str, Any]] = []
    contraction_rows: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    run_dirs = {"qhflow3": args.qhflow3_run, "nte": args.nte_run}

    try:
        for model in models:
            result = run_model(
                model,
                configs[model],
                run_dirs[model],
                loader,
                required_irreps,
                orbital_basis,
                ls_list,
                collector,
                device,
            )
            model_provenance, contractions, singular, residual = result
            model_provenance["config"] = str(
                (
                    args.qhflow3_config
                    if model == "qhflow3"
                    else args.nte_config
                ).resolve()
            )
            provenance.append(model_provenance)
            contraction_rows.extend(contractions)
            singular_rows.extend(singular)
            residual_rows.extend(residual)

        degree_rows, channel_rows = collector.finalize()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "layer_degree_stats.csv", degree_rows)
        write_csv(args.output_dir / "layer_channel_rms.csv", channel_rows)
        write_csv(args.output_dir / "contraction_stats.csv", contraction_rows)
        write_csv(
            args.output_dir / "contraction_singular_values.csv", singular_rows
        )
        write_csv(
            args.output_dir / "residual_degree_scales.csv", residual_rows
        )
        summary = {
            "models": models,
            "model_labels": {model: MODEL_LABELS[model] for model in models},
            "database": str(args.dbpath.resolve()),
            "validation_rows": [12081, 12081 + args.num_molecules],
            "batch_size": args.batch_size,
            "degree_normalization": (
                "sqrt(sum(x^2)/(items*(2l+1)*channels))"
            ),
            "power_fraction": "sum_l(x^2)/sum_all_degrees(x^2)",
            "effective_channels": "(sum(channel_power)^2)/sum(channel_power^2)",
            "source": source_provenance(),
            "provenance": provenance,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        plot_features(args.output_dir, degree_rows, models)
        plot_contractions(args.output_dir, contraction_rows, models)
        write_report(
            args.output_dir,
            degree_rows,
            contraction_rows,
            residual_rows,
            provenance,
            models,
            args.num_molecules,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_dir": str(args.output_dir.resolve()),
                    "degree_rows": len(degree_rows),
                    "channel_rows": len(channel_rows),
                    "contraction_rows": len(contraction_rows),
                },
                indent=2,
            )
        )
    finally:
        utils_compute.cleanup_process_group()


if __name__ == "__main__":
    main()
