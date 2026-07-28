"""Canonical SplitTrainer integration for joint node/edge endpoint flow."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from maloq.train_utils.splittrainer import DEFAULT_OUTPUT_FOLDER, SplitTrainer

from .conditioning import (
    CoupledAOCodec,
    CoupledBasisTransform,
    HamiltonianSymmetryProjector,
)
from .config import FlowMatchingConfig
from .objective import EndpointFlowMatcher
from .prior import build_coupled_prior
from .sampler import EndpointEulerResult, EndpointEulerSampler, EndpointPrediction


def _closed_shell_target(batch: Any, name: str) -> Tensor:
    target = getattr(batch, name)
    if not isinstance(target, Tensor):
        raise TypeError(f"batch.{name} must be a tensor.")
    if target.ndim == 3:
        if target.shape[0] != 1:
            raise ValueError("Endpoint flow currently supports closed shell only.")
        target = target[0]
    if target.ndim != 2 or not target.is_floating_point():
        raise ValueError(f"batch.{name} must have shape [entries, coupled_dim].")
    return target


def _closed_shell_prediction(value: Any, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} endpoint prediction must be a tensor.")
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError("Endpoint flow currently supports closed shell only.")
        value = value[0]
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} endpoint must have shape [entries, coupled_dim].")
    return value


def _closed_shell_mask(batch: Any, name: str, target: Tensor) -> Tensor:
    mask = getattr(batch, name)
    if not isinstance(mask, Tensor) or mask.dtype is not torch.bool:
        raise TypeError(f"batch.{name} must be a boolean tensor.")
    if mask.ndim == target.ndim + 1:
        if mask.shape[0] != 1:
            raise ValueError("Endpoint flow currently supports closed shell only.")
        mask = mask[0]
    if mask.shape != target.shape:
        raise ValueError(
            f"batch.{name} must match the coupled target shape "
            f"{tuple(target.shape)}; got {tuple(mask.shape)}."
        )
    return mask


def _install_closed_shell_target(batch: Any, name: str, target: Tensor) -> None:
    original = getattr(batch, name)
    setattr(batch, name, target.unsqueeze(0) if original.ndim == 3 else target)


def _node_graph_index(batch: Any, node_count: int, device: torch.device) -> Tensor:
    index = getattr(batch, "batch", None)
    if isinstance(index, Tensor) and index.numel() == node_count:
        return index.to(device=device, dtype=torch.long)
    ptr = getattr(batch, "ptr", None)
    if isinstance(ptr, Tensor) and ptr.ndim == 1 and ptr.numel() >= 2:
        counts = (ptr[1:] - ptr[:-1]).to(device=device, dtype=torch.long)
        if int(counts.sum().item()) == node_count:
            return torch.arange(
                counts.numel(), device=device, dtype=torch.long
            ).repeat_interleave(counts)
    raise ValueError("Batch must expose node-level batch indices or a valid ptr.")


def _edge_graph_index(
    batch: Any,
    *,
    node_graph_index: Tensor,
    edge_count: int,
) -> Tensor:
    edge_index = getattr(batch, "edge_index", None)
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.dtype not in {torch.int32, torch.int64}
        or edge_index.shape != (2, edge_count)
    ):
        observed_shape = (
            tuple(edge_index.shape) if isinstance(edge_index, Tensor) else None
        )
        observed_dtype = (
            str(edge_index.dtype) if isinstance(edge_index, Tensor) else None
        )
        raise ValueError(
            "batch.edge_index must be an integer tensor of shape [2, E]: "
            f"target_edges={edge_count}, observed_shape={observed_shape}, "
            f"observed_dtype={observed_dtype}."
        )
    if edge_index.dtype != torch.long:
        edge_index = edge_index.to(dtype=torch.long)
        batch.edge_index = edge_index
    if edge_index.device != node_graph_index.device:
        raise ValueError("batch.edge_index and graph indices must share a device.")
    if edge_count == 0:
        raise ValueError("Full-matrix endpoint flow requires directed edges.")
    if (
        int(edge_index.min().item()) < 0
        or int(edge_index.max().item()) >= node_graph_index.numel()
    ):
        raise ValueError("batch.edge_index contains an out-of-range node index.")
    source_graph = node_graph_index.index_select(0, edge_index[0])
    target_graph = node_graph_index.index_select(0, edge_index[1])
    if not torch.equal(source_graph, target_graph):
        raise ValueError("Every directed edge must stay within one molecular graph.")
    return source_graph


def _validate_explicit_source(
    source: Tensor,
    reference: Tensor,
    *,
    name: str,
) -> None:
    if (
        not isinstance(source, Tensor)
        or source.shape != reference.shape
        or source.dtype != reference.dtype
        or source.device != reference.device
    ):
        raise ValueError(
            f"Explicit {name} source must match target shape/dtype/device."
        )


class EndpointCorruptingLoader:
    """Install one symmetry-preserving node/edge flow state per batch."""

    def __init__(
        self,
        loader: Any,
        *,
        matcher: EndpointFlowMatcher,
        basis_transform: CoupledBasisTransform,
        device: torch.device | str,
    ):
        self.loader = loader
        self.matcher = matcher
        self.codec = CoupledAOCodec(basis_transform)
        self.projector = HamiltonianSymmetryProjector(basis_transform)
        self.prior = build_coupled_prior(matcher.config, basis_transform)
        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)

    def __iter__(self):
        for batch in self.loader:
            batch = batch.to(self.device)
            clean_node = _closed_shell_target(batch, "node_y")
            clean_edge = _closed_shell_target(batch, "y")
            node_mask = _closed_shell_mask(
                batch,
                "node_padding_mask",
                clean_node,
            ).to(device=clean_node.device)
            edge_mask = _closed_shell_mask(
                batch,
                "edge_padding_mask",
                clean_edge,
            ).to(device=clean_edge.device)
            node_graph_index = _node_graph_index(
                batch, clean_node.shape[0], clean_node.device
            )
            edge_graph_index = _edge_graph_index(
                batch,
                node_graph_index=node_graph_index,
                edge_count=clean_edge.shape[0],
            )
            num_graphs = int(node_graph_index.max().item()) + 1
            time = self.matcher.sample_time(
                num_graphs,
                reference=clean_node,
            )

            clean = self.projector(
                clean_node,
                clean_edge,
                edge_index=batch.edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
            source = self.projector(
                self.prior.sample(clean.node, mask=node_mask),
                self.prior.sample(clean.edge, mask=edge_mask),
                edge_index=batch.edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
            sample = self.matcher.corrupt_joint(
                clean.node,
                clean.edge,
                node_graph_index=node_graph_index,
                edge_graph_index=edge_graph_index,
                time=time,
                node_source=source.node,
                edge_source=source.edge,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
            state = self.projector(
                sample.node.state,
                sample.edge.state,
                edge_index=batch.edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
            _install_closed_shell_target(batch, "node_y", clean.node)
            _install_closed_shell_target(batch, "y", clean.edge)
            batch.node_flow_t = state.node
            batch.init_ham_t = self.codec.decode(state.node)
            batch.edge_flow_t = state.edge
            batch.t = sample.time
            yield batch


class EndpointFlowTrainer(SplitTrainer):
    """Reuse the complete canonical loop with flow-corrupted input batches."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        head_irreps: Any,
        config: FlowMatchingConfig,
        *,
        save_frequency: int = 10,
        run_name: str = "flow_matching",
        wandb_run: Any = None,
        validation_inference_seed: int = 42,
    ):
        super().__init__(
            backbone=backbone,
            head=head,
            head_irreps=head_irreps,
            save_frequency=save_frequency,
            run_name=run_name,
            wandb_run=wandb_run,
        )
        self.flow_config = config
        self.matcher = EndpointFlowMatcher(config)
        self.validation_inference_seed = int(validation_inference_seed)
        if self.validation_inference_seed < 0:
            raise ValueError("validation_inference_seed must be non-negative.")
        self._validation_inference_calls = 0
        self._validation_inference_num_batches = 1

    def sample_batch(
        self,
        batch: Any,
        *,
        basis_transform: CoupledBasisTransform,
        device: torch.device | str,
        node_source: Tensor | None = None,
        edge_source: Tensor | None = None,
        generator: torch.Generator | None = None,
        num_ode_steps: int | None = None,
    ) -> EndpointEulerResult:
        """Integrate a complete block-sparse Hamiltonian with joint Euler steps."""
        batch = batch.to(device)
        clean_node = _closed_shell_target(batch, "node_y")
        node_mask = _closed_shell_mask(
            batch,
            "node_padding_mask",
            clean_node,
        ).to(device=clean_node.device)
        clean_edge = _closed_shell_target(batch, "y")
        edge_mask = _closed_shell_mask(
            batch,
            "edge_padding_mask",
            clean_edge,
        ).to(device=clean_edge.device)
        node_graph_index = _node_graph_index(
            batch,
            clean_node.shape[0],
            clean_node.device,
        )
        edge_graph_index = _edge_graph_index(
            batch,
            node_graph_index=node_graph_index,
            edge_count=clean_edge.shape[0],
        )
        prior = build_coupled_prior(self.flow_config, basis_transform)

        if node_source is None:
            node_source = prior.sample(
                clean_node,
                mask=node_mask,
                generator=generator,
            )
        else:
            _validate_explicit_source(node_source, clean_node, name="node")
        if edge_source is None:
            edge_source = prior.sample(
                clean_edge,
                mask=edge_mask,
                generator=generator,
            )
        else:
            _validate_explicit_source(edge_source, clean_edge, name="edge")

        projector = HamiltonianSymmetryProjector(basis_transform)
        codec = CoupledAOCodec(basis_transform)
        source = projector(
            node_source,
            edge_source,
            edge_index=batch.edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )
        sampler_config = self.flow_config
        if num_ode_steps is not None:
            if isinstance(num_ode_steps, bool) or not isinstance(
                    num_ode_steps, int):
                raise TypeError("num_ode_steps override must be an integer.")
            if num_ode_steps < 1:
                raise ValueError("num_ode_steps override must be at least one.")
            sampler_config = self.flow_config.model_copy(
                update={"num_ode_steps": num_ode_steps}
            )
        sampler = EndpointEulerSampler(sampler_config)

        backbone_was_training = bool(getattr(self.backbone, "training", False))
        head_was_training = bool(getattr(self.head, "training", False))
        self.backbone.eval()
        self.head.eval()

        def project_state(
            node_state: Tensor,
            edge_state: Tensor,
        ) -> EndpointPrediction:
            projected = projector(
                node_state,
                edge_state,
                edge_index=batch.edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
            return EndpointPrediction(
                node=projected.node,
                edge=projected.edge,
            )

        def predict_endpoint(
            node_state: Tensor,
            edge_state: Tensor,
            graph_time: Tensor,
        ) -> EndpointPrediction:
            batch.node_flow_t = node_state
            batch.init_ham_t = codec.decode(node_state)
            batch.edge_flow_t = edge_state
            batch.t = graph_time
            features = self.backbone(batch)
            output = self.head(features, batch)
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError(
                    "Matrix head must return (node_endpoint, edge_endpoint)."
                )
            node = _closed_shell_prediction(output[0], name="node")
            edge = _closed_shell_prediction(output[1], name="edge")
            if node.shape != clean_node.shape or edge.shape != clean_edge.shape:
                raise ValueError("Head endpoint shapes must match canonical targets.")
            return project_state(node, edge)

        try:
            with torch.inference_mode():
                result = sampler.sample(
                    source.node,
                    source.edge,
                    node_graph_index=node_graph_index,
                    edge_graph_index=edge_graph_index,
                    predict_endpoint=predict_endpoint,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                    project_state=project_state,
                )
                final_state = projector(
                    result.node,
                    result.edge,
                    edge_index=batch.edge_index,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                )
                return EndpointEulerResult(
                    node=final_state.node,
                    edge=final_state.edge,
                    times=result.times,
                )
        finally:
            if backbone_was_training:
                self.backbone.train()
            if head_was_training:
                self.head.train()

    def _next_validation_inference_generator(
        self,
        device: torch.device | str,
    ) -> torch.Generator:
        """Return a rank/batch-stable generator without touching global RNG."""
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        batch_index = (
            self._validation_inference_calls % self._validation_inference_num_batches
        )
        self._validation_inference_calls += 1
        seed = self.validation_inference_seed + rank * 1_000_003 + batch_index
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(seed)
        return generator

    def compute_validation_matrix_error_sums(
        self,
        batch,
        node_output,
        edge_output,
        node_target,
        edge_target,
        basis_transform,
    ):
        """Evaluate canonical AO metrics from the joint ODE endpoint."""
        del node_output, edge_output
        device = _closed_shell_target(batch, "node_y").device
        return self._compute_endpoint_validation_matrix_error_sums(
            batch,
            node_target,
            edge_target,
            basis_transform,
            device=device,
            generator=self._next_validation_inference_generator(device),
        )

    def compute_validation_matrix_error_sums_with_variants(
        self,
        batch,
        node_output,
        edge_output,
        node_target,
        edge_target,
        basis_transform,
    ):
        """Record configured Euler and same-prior one-shot AO metrics."""
        del node_output, edge_output
        device = _closed_shell_target(batch, "node_y").device
        configured_generator = self._next_validation_inference_generator(device)
        initial_generator_state = configured_generator.get_state()
        configured_stats = self._compute_endpoint_validation_matrix_error_sums(
            batch,
            node_target,
            edge_target,
            basis_transform,
            device=device,
            generator=configured_generator,
        )

        one_shot_generator = torch.Generator(device=torch.device(device))
        one_shot_generator.set_state(initial_generator_state)
        one_shot_stats = self._compute_endpoint_validation_matrix_error_sums(
            batch,
            node_target,
            edge_target,
            basis_transform,
            device=device,
            generator=one_shot_generator,
            num_ode_steps=1,
        )
        return configured_stats, {
            "validation/flow_matching_configured_ode": configured_stats,
            "validation/flow_matching_one_shot": one_shot_stats,
        }

    def _compute_endpoint_validation_matrix_error_sums(
        self,
        batch,
        node_target,
        edge_target,
        basis_transform,
        *,
        device: torch.device | str,
        generator: torch.Generator,
        num_ode_steps: int | None = None,
    ):
        sample_kwargs = {
            "basis_transform": basis_transform,
            "device": device,
            "generator": generator,
        }
        if num_ode_steps is not None:
            sample_kwargs["num_ode_steps"] = num_ode_steps
        endpoint = self.sample_batch(batch, **sample_kwargs)
        return super().compute_validation_matrix_error_sums(
            batch,
            endpoint.node,
            endpoint.edge,
            node_target,
            edge_target,
            basis_transform,
        )

    def train(
        self,
        num_epochs,
        loss_fxn,
        optimizer,
        scheduler,
        device,
        train_loader,
        loss_target_string,
        node_target_name,
        val_loader=None,
        edge_target_name=None,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        num_warmup_epochs=0,
        train_backbone=True,
        train_head=True,
        basis_transform=None,
        compute_uncoupled_loss=False,
        element_references=None,
        step_every_epoch=False,
        validation_matrix_metrics=False,
        validation_matrix_metrics_frequency=1,
        gradient_clip_val=None,
        gradient_accumulation_steps=1,
        wandb_enabled=False,
        wandb_log_every_n_steps=10,
        start_epoch=0,
        initial_history=None,
        checkpoint_callback=None,
        min_lr=1e-10,
    ):
        if basis_transform is None:
            raise ValueError("Endpoint flow requires the canonical basis_transform.")
        if loss_target_string not in {"fock_matrix", "density_matrix"}:
            raise ValueError("Endpoint flow requires a matrix target.")
        if node_target_name != "node_y" or edge_target_name != "y":
            raise ValueError("Endpoint flow requires canonical node_y/y targets.")
        if compute_uncoupled_loss:
            raise ValueError("Endpoint flow loss is defined in coupled coordinates.")
        if int(validation_matrix_metrics_frequency) < 1:
            raise ValueError(
                "validation_matrix_metrics_frequency must be at least one."
            )

        train_loader = EndpointCorruptingLoader(
            train_loader,
            matcher=self.matcher,
            basis_transform=basis_transform,
            device=device,
        )
        if val_loader is not None:
            val_loader = EndpointCorruptingLoader(
                val_loader,
                matcher=self.matcher,
                basis_transform=basis_transform,
                device=device,
            )
        self._validation_inference_calls = 0
        self._validation_inference_num_batches = len(
            val_loader if val_loader is not None else train_loader
        )
        if self._validation_inference_num_batches <= 0:
            raise ValueError("Endpoint flow validation loader must be non-empty.")
        return super().train(
            num_epochs,
            loss_fxn,
            optimizer,
            scheduler,
            device,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_target_string=loss_target_string,
            node_target_name=node_target_name,
            edge_target_name=edge_target_name,
            output_folder=output_folder,
            num_warmup_epochs=num_warmup_epochs,
            train_backbone=train_backbone,
            train_head=train_head,
            basis_transform=basis_transform,
            compute_uncoupled_loss=False,
            element_references=element_references,
            step_every_epoch=step_every_epoch,
            validation_matrix_metrics=validation_matrix_metrics,
            validation_matrix_metrics_frequency=validation_matrix_metrics_frequency,
            gradient_clip_val=gradient_clip_val,
            gradient_accumulation_steps=gradient_accumulation_steps,
            wandb_enabled=wandb_enabled,
            wandb_log_every_n_steps=wandb_log_every_n_steps,
            start_epoch=start_epoch,
            initial_history=initial_history,
            checkpoint_callback=checkpoint_callback,
            min_lr=min_lr,
        )
