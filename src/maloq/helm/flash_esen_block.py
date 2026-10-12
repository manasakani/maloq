# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""FlashSO2-backed implementation of Maloq's eSEN message-passing block."""

from __future__ import annotations

import torch

try:
    from flash_so2 import (
        Precision,
        SO2_Convolution,
        auto_precision_mode,
    )
except ImportError as exc:  # pragma: no cover - depends on the optional package
    raise ImportError(
        "Flash_eSEN_Block requires the optional flash_so2 package"
    ) from exc

from .esen_block import eSEN_Block
from .nn.communication import ExchangeHaloNodes

_SUPPORTED_LMAX = frozenset({4, 6, 8})


def _resolve_precision(
    precision: str | Precision,
    *,
    dtype: torch.dtype | None = None,
) -> str:
    """Resolve FlashSO2 precision through Maloq's global dtype policy."""

    value = precision.value if isinstance(precision, Precision) else str(precision)
    value = value.strip().lower()
    if value == "auto":
        return auto_precision_mode(dtype or torch.get_default_dtype()).value

    try:
        return Precision(value).value
    except ValueError as exc:
        supported = ", ".join(["auto", *(mode.value for mode in Precision)])
        raise ValueError(
            f"flash_esen_block must be one of {supported}, or None to use "
            f"Maloq's native eSEN block; got {precision!r}"
        ) from exc


def _gate_activation_name(gate_module: torch.nn.Module) -> str:
    activation = getattr(gate_module, "gate_act", None)
    if isinstance(activation, torch.nn.Tanh):
        return "tanh"
    if isinstance(activation, torch.nn.Sigmoid):
        return "sigmoid"
    raise ValueError(f"FlashSO2 does not support gate activation {activation!r}")


def _halo_edge_index(partition, num_local: int) -> torch.Tensor:
    """This rank's edges renumbered into ExchangeHaloNodes' [local; received] rows.

    Built from the partition's existing communication plan and cached on it, so
    every block, layer and step hands FlashSO2 the same tensor -- its topology
    cache is keyed on that tensor's storage and version counter. It is built
    outside inference mode, as SplitTrainer.infer() runs the model under
    torch.inference_mode() and an inference tensor has no version counter.
    """

    edge_index = getattr(partition, "_flash_so2_edge_index", None)
    if edge_index is None:
        with torch.inference_mode(False):
            source = partition.expand_edge_0
            target_rows = partition.expand_edge_1["local_indices_torch"]  # targets are always local
            device = target_rows.device
            source_rows = torch.empty_like(target_rows)
            source_rows[torch.as_tensor(source["is_local"], device=device)] = source["local_indices_torch"]
            source_rows[torch.as_tensor(source["is_remote"], device=device)] = (
                num_local + source["remote_indices_torch"]
            )
            edge_index = torch.stack([source_rows, target_rows])
        partition._flash_so2_edge_index = edge_index
    return edge_index


class Flash_eSEN_Block(eSEN_Block):
    """Parallel eSEN block whose Edgewise path is executed by FlashSO2.

    The native eSEN block still owns every trainable module. FlashSO2 directly
    references those modules, preserving parameter names and the native
    ``state_dict`` layout.
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced,
        SO3_grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: str,
        act_type: str,
        mlp_type: str,
        message_type: str,
        include_edges=True,
        node_or_edge: str = "node",
        *,
        precision: str,
    ) -> None:
        model_dtype = torch.get_default_dtype()
        if model_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "FlashSO2 requires Maloq dtype float32 or bfloat16; "
                f"got {model_dtype}"
            )
        if lmax != mmax or lmax not in _SUPPORTED_LMAX:
            raise ValueError(
                "FlashSO2 requires lmax == mmax with lmax in "
                f"{sorted(_SUPPORTED_LMAX)}; got lmax={lmax}, mmax={mmax}"
            )
        if message_type != "source-target":
            raise ValueError(
                "FlashSO2 requires Maloq message_type='source-target'; "
                f"got {message_type!r}"
            )
        resolved_precision = _resolve_precision(precision, dtype=model_dtype)

        super().__init__(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            edge_channels_list=edge_channels_list,
            cutoff=cutoff,
            norm_type=norm_type,
            act_type=act_type,
            mlp_type=mlp_type,
            message_type=message_type,
            include_edges=include_edges,
            node_or_edge=node_or_edge,
        )

        radial_function = self.edge_wise.so2_conv_1.rad_func
        radial_mlp = getattr(radial_function, "net", None)
        if radial_mlp is None:
            raise ValueError("Maloq conv1 must provide rad_func.net for FlashSO2")

        convolution = SO2_Convolution(
            self.edge_wise.so2_conv_1,
            self.edge_wise.so2_conv_2,
            radial_mlp,
            compute_precision=resolved_precision,
            gate_activation=_gate_activation_name(self.edge_wise.act),
        )
        # The convolution references modules already registered under
        # edge_wise. Keep it out of Module._modules so state_dict paths are not
        # duplicated while still calling FlashSO2 directly.
        self.__dict__["_flash_so2_convolution"] = convolution

    def _run_edgewise(
        self,
        *,
        x_node: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        reduce: str,
        partition,
    ) -> torch.Tensor:
        if not x_node.is_cuda:
            raise RuntimeError("FlashSO2 execution requires CUDA tensors")

        num_local = x_node.shape[0]
        if partition is not None:
            # Distributed graph: this rank owns every edge whose target it owns,
            # but a source may live on another rank. FlashSO2 gathers node rows
            # itself, so append the received source nodes to the local ones and
            # renumber this rank's edges into that tensor (same edge order, so
            # x_edge and the Wigner rows still line up).
            local_edge_index = _halo_edge_index(partition, num_local)
            if local_edge_index.shape[1] != edge_index.shape[1]:
                raise ValueError(
                    "the partition's edge list does not match this batch's edges: "
                    f"{local_edge_index.shape[1]} vs {edge_index.shape[1]}"
                )
            x_node = ExchangeHaloNodes.apply(x_node, partition.expand_edge_0)
            edge_index = local_edge_index

        output = self._flash_so2_convolution(
            x_edge,
            x_node,
            wigner,
            wigner_inv,
            edge_index,
            reduce=reduce,
        )
        if reduce == "node" and output.shape[0] != num_local:
            # Targets are always local, so the received rows get no messages.
            output = output[:num_local]
        return output if output.dtype == x_node.dtype else output.to(x_node.dtype)

    def train(self, mode: bool = True):
        """Keep the unregistered execution view in the block's train/eval mode."""

        super().train(mode)
        self._flash_so2_convolution.train(mode)
        return self

    def forward(
        self,
        x_message_node,
        x_message_edge,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        node_or_edge,
        partition,
    ):
        """Run the native eSEN residual block around the fused Edgewise path."""

        if node_or_edge == "node":
            x_res = x_message_node
            x_message_node = self.norm_1(x_message_node)
            x_message_node = self._run_edgewise(
                x_node=x_message_node,
                x_edge=x_edge,
                edge_index=edge_index,
                wigner=wigner,
                wigner_inv=wigner_inv,
                reduce="node",
                partition=partition,
            )

            x_message_node = x_message_node + x_res
            x_res = x_message_node
            x_message_node = self.norm_2(x_message_node)
            x_message_node = self.atom_wise(x_message_node)
            return x_message_node + x_res

        if node_or_edge == "edge":
            x_res = x_message_edge

            x_message_edge = self._run_edgewise(
                x_node=x_message_node,
                x_edge=x_edge,
                edge_index=edge_index,
                wigner=wigner,
                wigner_inv=wigner_inv,
                reduce="edge",
                partition=partition,
            )
            x_message_edge = self.norm_1(x_message_edge)

            x_message_edge = x_message_edge + x_res
            x_res = x_message_edge
            x_message_edge = self.norm_2(x_message_edge)

            x_message_edge = self.atom_wise(x_message_edge)
            return x_message_edge + x_res

        raise ValueError(f"Unknown reduction target {node_or_edge!r}")


__all__ = [
    "Flash_eSEN_Block",
]
