# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Opt-in training workflow with atomic, distributed-safe epoch resume.

The legacy :mod:`training_workflow` remains the default execution path.  This
module adds a unified checkpoint that restores the complete optimizer and
scheduler trajectory and resumes at the next epoch.  Checkpoints are written
by rank zero, while every rank contributes and later restores its own random
number generator state.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from . import splittrainer
from . import training_workflow as legacy


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_NAME = "training_state.pt"
PREVIOUS_CHECKPOINT_NAME = "training_state.prev.pt"
CHECKPOINT_META_NAME = "training_state.meta.json"

_SIGNATURE_EXCLUDED_KEYS = {
    "backbone_checkpoint",
    "head_checkpoint",
    "output_folder",
    "resume_from",
    "restart_backbone",
    "restart_head",
    "restart_optimizer",
    "run_name",
    "save_frequency",
    "use_wandb",
    "wandb_entity",
    "wandb_group",
    "wandb_job_type",
    "wandb_log_every_n_steps",
    "wandb_mode",
    "wandb_project",
    "wandb_run_name",
    "wandb_tags",
}

_SIGNATURE_COMPATIBILITY_DEFAULTS = {
    "direct_atomwise_layers": [],
    "direct_edgewise_layers": [],
    "initial_edge_state_mode": "edge_degree",
    "qhflow3_muonize_output_projection": False,
    "node_stack_mode": "nte",
    "nte_output_projection_mode": "so3_linear",
    "output_norm_sharing": "shared",
    "qhflow3_layer_gaussian_width": 2.0,
    "qhflow3_layer_grid_ffn_chunk_size": 512,
    "qhflow3_exact_pair_rng_aligned": False,
}


def _normalise_signature_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _normalise_signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_signature_value(item) for item in value]
    if callable(value):
        module = getattr(value, "__module__", "")
        qualified_name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
        return f"{module}.{qualified_name}".strip(".")
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def resume_signature(config: dict[str, Any], world_size: int) -> dict[str, Any]:
    """Return stable, training-semantic configuration used for resume checks."""
    signature = {
        key: _normalise_signature_value(value)
        for key, value in sorted(config.items())
        if key not in _SIGNATURE_EXCLUDED_KEYS
        and key != "element_references"
        and not key.endswith("_fxn")
    }
    signature["world_size"] = int(world_size)
    return signature


def signature_digest(signature: dict[str, Any]) -> str:
    serialised = json.dumps(
        signature,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _migrate_stored_signature_defaults(
    signature: dict[str, Any],
) -> dict[str, Any]:
    """Add behavior-preserving defaults introduced after a checkpoint."""
    migrated = dict(signature)
    for key, value in _SIGNATURE_COMPATIBILITY_DEFAULTS.items():
        migrated.setdefault(key, value)
    return migrated


def _checkpoint_directory(source: Path) -> Path:
    return source if source.is_dir() else source.parent


def checkpoint_candidates(source: str | os.PathLike[str]) -> tuple[Path, ...]:
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        return (
            path / CHECKPOINT_NAME,
            path / PREVIOUS_CHECKPOINT_NAME,
        )
    candidates = [path]
    if path.name == CHECKPOINT_NAME:
        candidates.append(path.with_name(PREVIOUS_CHECKPOINT_NAME))
    return tuple(candidates)


def load_training_checkpoint(
    source: str | os.PathLike[str],
) -> tuple[dict[str, Any], Path]:
    """Load the newest valid checkpoint, falling back to the previous epoch."""
    errors = []
    for candidate in checkpoint_candidates(source):
        if not candidate.is_file():
            errors.append(f"{candidate}: missing")
            continue
        try:
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if not isinstance(state, dict):
                raise TypeError("checkpoint payload is not a dictionary")
            if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema version {state.get('schema_version')!r}"
                )
            required = {
                "backbone_state_dict",
                "completed_epoch",
                "config_signature_digest",
                "head_state_dict",
                "history",
                "optimizer_state_dict",
                "rng_states",
                "scheduler_state_dict",
                "world_size",
            }
            missing = sorted(required.difference(state))
            if missing:
                raise ValueError(f"missing checkpoint keys: {missing}")
            return state, candidate
        except Exception as exc:  # the previous generation may still be valid
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "No valid fixed-workflow checkpoint was found:\n- "
        + "\n- ".join(errors)
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text)
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint_write(state: dict[str, Any], output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / CHECKPOINT_NAME
    previous = output_directory / PREVIOUS_CHECKPOINT_NAME
    temporary = output_directory / f".{CHECKPOINT_NAME}.tmp-{os.getpid()}"
    try:
        torch.save(state, temporary)
        _fsync_file(temporary)
        if destination.exists():
            os.replace(destination, previous)
        os.replace(temporary, destination)
        _fsync_directory(output_directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available() and device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if (
        state.get("torch_cuda") is not None
        and torch.cuda.is_available()
        and device.type == "cuda"
    ):
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _model_state_dict(model: Any) -> dict[str, Any]:
    module = getattr(model, "module", model)
    return module.state_dict()


def _format_loss_history(history: dict[str, list[float]], validation: bool) -> str:
    node_key = "node_val" if validation else "node"
    edge_key = "edge_val" if validation else "edge"
    nodes = history.get(node_key, [])
    edges = history.get(edge_key)
    if edges is None:
        return "".join(f"{node:.10f}\n" for node in nodes)
    return "".join(
        f"{edge:.10f}\t{node:.10f}\n"
        for edge, node in zip(edges, nodes, strict=True)
    )


def _write_loss_histories(
    output_directory: Path,
    history: dict[str, list[float]],
) -> None:
    training_text = _format_loss_history(history, validation=False)
    validation_text = _format_loss_history(history, validation=True)
    for model_name in ("backbone", "head"):
        _atomic_text_write(
            output_directory / f"{model_name}_training_loss.txt",
            training_text,
        )
        _atomic_text_write(
            output_directory / f"{model_name}_validation_loss.txt",
            validation_text,
        )


def _read_resume_metadata(source: Path) -> dict[str, Any]:
    meta_path = _checkpoint_directory(source) / CHECKPOINT_META_NAME
    if not meta_path.is_file():
        return {}
    try:
        payload = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


class TrainingWorkflowFixed(legacy.TrainingWorkflow):
    """Training workflow with opt-in, epoch-boundary exact resume."""

    def __init__(self, config: dict[str, Any]):
        fixed_config = dict(config)
        environment_resume = os.environ.get("MALOQ_FIXED_RESUME_FROM")
        resume_source = fixed_config.pop("resume_from", None) or environment_resume
        self.resume_source = (
            Path(resume_source).expanduser().resolve()
            if resume_source
            else None
        )
        self.allow_config_mismatch = bool(
            fixed_config.pop("allow_resume_config_mismatch", False)
            or os.environ.get("MALOQ_FIXED_ALLOW_CONFIG_MISMATCH") == "1"
        )
        stop_after_epoch = (
            fixed_config.pop("fixed_stop_after_epoch", None)
            or os.environ.get("MALOQ_FIXED_STOP_AFTER_EPOCH")
        )
        self.stop_after_epoch = (
            int(stop_after_epoch) if stop_after_epoch is not None else None
        )
        if any(
            fixed_config.get(key, False)
            for key in ("restart_backbone", "restart_head", "restart_optimizer")
        ):
            raise ValueError(
                "TrainingWorkflowFixed does not combine legacy restart flags "
                "with unified resume; use resume_from instead."
            )
        fixed_config.update(
            restart_backbone=False,
            restart_head=False,
            restart_optimizer=False,
        )
        self.loaded_checkpoint_path: Path | None = None
        super().__init__(fixed_config)

    def setup_tracking(self):
        if (
            self.rank != 0
            or not self.config.get("use_wandb", False)
            or self.resume_source is None
        ):
            return super().setup_tracking()

        metadata = _read_resume_metadata(self.resume_source)
        wandb_run_id = metadata.get("wandb_run_id")
        if not wandb_run_id:
            return super().setup_tracking()

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking was requested, but wandb is not installed."
            ) from exc

        wandb_config = {
            key: (
                value
                if isinstance(value, (str, int, float, bool)) or value is None
                else value.__name__
                if hasattr(value, "__name__")
                else str(value)
            )
            for key, value in self.config.items()
        }
        run = wandb.init(
            project=self.config["wandb_project"],
            entity=self.config.get("wandb_entity"),
            id=str(wandb_run_id),
            resume="allow",
            name=self.config.get("wandb_run_name") or self.config["run_name"],
            group=self.config.get("wandb_group"),
            job_type=self.config.get("wandb_job_type"),
            tags=list(self.config.get("wandb_tags") or ()),
            dir=self.config["output_folder"],
            config=wandb_config,
            mode=self.config.get("wandb_mode", "online"),
        )
        print(f"W&B run resumed: {run.name} ({run.id})", flush=True)
        return run

    def _database(self):
        c = self.config
        if c["dataset_name"] == "QM7":
            target_property = (
                "density_matrix"
                if c["loss_target"] == "density_matrix"
                else "hamiltonian"
            )
            load_properties = [
                "energy",
                "forces",
                target_property,
                "overlap",
            ]
            if c.get("delta_learning", False):
                initial_target_property = (
                    "initial_density_matrix"
                    if target_property == "density_matrix"
                    else "initial_hamiltonian"
                )
                load_properties.append(initial_target_property)
                if (
                    c["backbone_type"] == "qhflow3_clean"
                    or c.get("nte_input_conditioning") == "qhflow3_exact"
                ):
                    auxiliary_property = (
                        "initial_hamiltonian"
                        if target_property == "density_matrix"
                        else "initial_density_matrix"
                    )
                    load_properties.append(auxiliary_property)
            database = legacy.ASEAtomsData(c["dbpath"])
            database.load_properties = load_properties
            if c["shuffle"]:
                print("Shuffling QM7 dataset for training...")
                indices = list(range(len(database)))
                random.shuffle(indices)
                database = [database[index] for index in indices]
            return database
        if c["dataset_name"] == "nablaDFT":
            return legacy.HamiltonianDatabase(c["dbpath"])
        if c["dataset_name"] in {"omol", "cp2k_material"}:
            return None
        raise ValueError(f"Unknown dataset name: {c['dataset_name']}")

    def _load_resume_state(self, backbone, head, optimizer, scheduler):
        if self.resume_source is None:
            return 0, None

        state, checkpoint_path = load_training_checkpoint(self.resume_source)
        self.loaded_checkpoint_path = checkpoint_path
        expected_signature = resume_signature(self.config, self.world_size)
        expected_digest = signature_digest(expected_signature)
        stored_digest = state["config_signature_digest"]
        stored_signature = state.get("config_signature")
        if isinstance(stored_signature, dict):
            if signature_digest(stored_signature) != stored_digest:
                raise ValueError(
                    "Checkpoint configuration signature is corrupted."
                )
            stored_digest = signature_digest(
                _migrate_stored_signature_defaults(stored_signature)
            )
        if stored_digest != expected_digest and not self.allow_config_mismatch:
            raise ValueError(
                "Resume configuration does not match the checkpoint. "
                f"checkpoint={stored_digest}, current={expected_digest}. "
                "Use a matching dataset/model/optimizer/world size."
            )
        if int(state["world_size"]) != self.world_size:
            raise ValueError(
                "Changing world size during resume is unsupported: "
                f"checkpoint={state['world_size']}, current={self.world_size}."
            )

        backbone.load_state_dict(state["backbone_state_dict"], strict=True)
        head.load_state_dict(state["head_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])

        rank_key = str(self.rank)
        rng_states = state["rng_states"]
        if rank_key not in rng_states:
            raise ValueError(
                f"Checkpoint has no RNG state for rank {self.rank}; "
                f"available ranks are {sorted(rng_states)}."
            )
        _restore_rng_state(rng_states[rank_key], self.device)

        start_epoch = int(state["completed_epoch"]) + 1
        if start_epoch >= int(self.config["num_epochs"]):
            raise ValueError(
                f"Checkpoint has already completed {start_epoch} epochs; "
                f"the configured target is {self.config['num_epochs']}."
            )
        if self.rank == 0:
            print(
                f"Resuming fixed workflow from {checkpoint_path}: "
                f"epoch {start_epoch}/{self.config['num_epochs']}, "
                f"optimizer step {state.get('optimizer_step')}.",
                flush=True,
            )
        return start_epoch, state["history"]

    def _checkpoint_callback(
        self,
        *,
        epoch,
        backbone,
        head,
        optimizer,
        scheduler,
        history,
        optimizer_steps_per_epoch,
    ) -> None:
        local_rng = _capture_rng_state(self.device)
        if dist.is_initialized() and self.world_size > 1:
            gathered_rng = [None] * self.world_size
            dist.all_gather_object(gathered_rng, local_rng)
        else:
            gathered_rng = [local_rng]

        error = None
        if self.rank == 0:
            try:
                signature = resume_signature(self.config, self.world_size)
                state = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "completed_epoch": int(epoch),
                    "optimizer_step": int(
                        (epoch + 1) * optimizer_steps_per_epoch
                    ),
                    "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
                    "planned_num_epochs": int(self.config["num_epochs"]),
                    "world_size": int(self.world_size),
                    "backbone_state_dict": _model_state_dict(backbone),
                    "head_state_dict": _model_state_dict(head),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_states": {
                        str(rank): rng
                        for rank, rng in enumerate(gathered_rng)
                    },
                    "history": history,
                    "config_signature": signature,
                    "config_signature_digest": signature_digest(signature),
                    "wandb_run_id": (
                        getattr(self.wandb_run, "id", None)
                        if self.wandb_run is not None
                        else None
                    ),
                    "resume_parent": (
                        str(self.loaded_checkpoint_path)
                        if self.loaded_checkpoint_path is not None
                        else None
                    ),
                }
                output_directory = Path(self.config["output_folder"])
                checkpoint_path = _atomic_checkpoint_write(
                    state,
                    output_directory,
                )
                _write_loss_histories(output_directory, history)
                _atomic_text_write(
                    output_directory / CHECKPOINT_META_NAME,
                    json.dumps(
                        {
                            "schema_version": CHECKPOINT_SCHEMA_VERSION,
                            "checkpoint": str(checkpoint_path),
                            "completed_epoch": int(epoch),
                            "optimizer_step": state["optimizer_step"],
                            "world_size": int(self.world_size),
                            "wandb_run_id": state["wandb_run_id"],
                            "resume_parent": state["resume_parent"],
                        },
                        indent=2,
                    )
                    + "\n",
                )
                print(
                    f"Atomic checkpoint saved: {checkpoint_path} "
                    f"(completed epoch {epoch + 1})",
                    flush=True,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        if dist.is_initialized() and self.world_size > 1:
            error_payload = [error]
            dist.broadcast_object_list(error_payload, src=0)
            error = error_payload[0]
        if error is not None:
            raise RuntimeError(f"Fixed-workflow checkpoint failed: {error}")

    def run(self):
        if self.config["train_or_eval"] != "train":
            if self.resume_source is not None:
                raise ValueError("resume_from is supported only for training.")
            return super().run()

        try:
            database = self._database()
            (
                loader,
                val_loader,
                irreps,
                basis_transform,
                orbital_basis,
                ls_list,
            ) = self.prepare_loaders(database)
            backbone, head, optimizer = self.build_model(
                irreps,
                orbital_basis,
                ls_list,
            )
            scheduler = self._get_scheduler(optimizer, loader)
            start_epoch, initial_history = self._load_resume_state(
                backbone,
                head,
                optimizer,
                scheduler,
            )

            trainer = splittrainer.SplitTrainer(
                backbone=backbone,
                head=head,
                head_irreps=irreps,
                run_name=self.config.get("run_name", "run"),
                save_frequency=self.config.get("save_frequency", 10),
                wandb_run=self.wandb_run,
            )
            target_map = {
                "fock_matrix": ("node_y", "y"),
                "density_matrix": ("node_y", "y"),
                "forces": ("forces", None),
                "energies": ("energies", None),
            }
            node_target, edge_target = target_map[self.config["loss_target"]]
            training_end_epoch = int(self.config["num_epochs"])
            if self.stop_after_epoch is not None:
                if not start_epoch < self.stop_after_epoch <= training_end_epoch:
                    raise ValueError(
                        "fixed_stop_after_epoch must be after the resume epoch "
                        "and no greater than the configured num_epochs: "
                        f"start={start_epoch}, stop={self.stop_after_epoch}, "
                        f"target={training_end_epoch}."
                    )
                training_end_epoch = self.stop_after_epoch
                if self.rank == 0:
                    print(
                        "Fixed-workflow diagnostic stop requested after "
                        f"epoch {training_end_epoch}/{self.config['num_epochs']}.",
                        flush=True,
                    )
            trainer.train(
                training_end_epoch,
                self.config["train_loss_fxn"],
                optimizer,
                scheduler,
                self.device,
                train_loader=loader,
                val_loader=val_loader,
                loss_target_string=self.config["loss_target"],
                node_target_name=node_target,
                edge_target_name=edge_target,
                output_folder=self.config["output_folder"],
                train_backbone=self.config["train_backbone"],
                train_head=self.config["train_head"],
                basis_transform=basis_transform,
                compute_uncoupled_loss=self.config.get(
                    "compute_uncoupled_loss",
                    False,
                ),
                step_every_epoch=self.config.get("step_every_epoch", True),
                element_references=self.config.get("element_references"),
                validation_matrix_metrics=self.config.get(
                    "validation_matrix_metrics",
                    False,
                ),
                validation_matrix_metrics_frequency=self.config.get(
                    "validation_matrix_metrics_frequency",
                    1,
                ),
                gradient_clip_val=self.config.get("gradient_clip_val"),
                gradient_accumulation_steps=self.config.get(
                    "gradient_accumulation_steps",
                    1,
                ),
                wandb_enabled=self.config.get("use_wandb", False),
                wandb_log_every_n_steps=self.config.get(
                    "wandb_log_every_n_steps",
                    10,
                ),
                start_epoch=start_epoch,
                initial_history=initial_history,
                checkpoint_callback=self._checkpoint_callback,
            )
        finally:
            self.close()
