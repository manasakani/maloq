#!/usr/bin/env python3
"""Overfit one real NablaDFT Hamiltonian through an operator callback."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from e3nn.o3 import Irreps
from torch_geometric.data import Batch, Data

from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
from maloq.experimental.op_projection import (
    OpProjectionBackbone,
    OpProjectionHead,
    bind_operator_callback,
    probe_matrix_mse,
    rademacher_probes,
    relative_action_error,
)
from maloq.fock_utils import basis_sets
from maloq.fock_utils.utils_tensor_decomp import e3TensorDecomp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db"
)
DEFAULT_ORBITAL_CACHE = PROJECT_ROOT / "orbital_cache_nablaDFT.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--orbital-cache", type=Path, default=DEFAULT_ORBITAL_CACHE)
    parser.add_argument("--row", type=int, default=189)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--train-probes", type=int, default=2)
    parser.add_argument("--heldout-probes", type=int, default=8)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--distance-basis", type=int, default=16)
    parser.add_argument("--pair-chunk-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-heldout-reduction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "op_projection_single_batch_row189",
    )
    return parser.parse_args()


def load_orbital_metadata(cache_path: Path, device: torch.device):
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    required_irreps = Irreps(cache["req_output_irreps"])
    transform = e3TensorDecomp(
        required_irreps,
        cache["out_js_list"],
        default_dtype_torch=torch.float32,
        if_sort=False,
        device_torch=device,
    )
    return cache, required_irreps, transform


def complete_graph_batch(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    overlap: np.ndarray,
    *,
    device: torch.device,
) -> tuple[Batch, float]:
    atomic_numbers_t = torch.tensor(
        np.array(atomic_numbers, copy=True), dtype=torch.long, device=device
    )
    positions_t = torch.tensor(
        np.array(positions, copy=True), dtype=torch.float32, device=device
    )
    num_atoms = positions_t.shape[0]
    mask = ~torch.eye(num_atoms, dtype=torch.bool, device=device)
    row_atoms, col_atoms = torch.where(mask)
    edge_index = torch.stack((row_atoms, col_atoms), dim=0)
    # Match Fock_Targets.make_atomic_graphs: H_ij uses row i/column j and
    # the stored displacement is position[i] - position[j].
    displacement = positions_t[row_atoms] - positions_t[col_atoms]
    distance = displacement.norm(dim=1)
    edge_attr = torch.cat((distance[:, None], displacement), dim=1)
    data = Data(
        pos=positions_t,
        z=atomic_numbers_t,
        atomic_numbers=atomic_numbers_t,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_atoms_in_molecule=torch.tensor([num_atoms], device=device),
        charge=torch.zeros(1, dtype=torch.long, device=device),
        spin_multiplicity=torch.ones(1, dtype=torch.long, device=device),
    )
    batch = Batch.from_data_list([data])
    batch.overlap_matrix = [
        torch.tensor(np.array(overlap, copy=True), dtype=torch.float32, device=device)
    ]
    cutoff = float(distance.max().item() + 1.0)
    return batch, cutoff


def detached_features(backbone: OpProjectionBackbone, batch: Batch):
    backbone.eval()
    with torch.no_grad():
        output = backbone(batch)
    return {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in output.items()
    }


def gradient_norm(module: torch.nn.Module) -> tuple[float, int]:
    total = 0.0
    tensors = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        tensors += 1
        total += float(parameter.grad.detach().square().sum().item())
    return math.sqrt(total), tensors


def pack_dense_blocks(
    hamiltonian: np.ndarray,
    atomic_numbers: np.ndarray,
    edge_index: torch.Tensor,
    orbital_basis: dict[int, list[int]],
    orbital_template: list,
    packed_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ao_counts = [
        sum(2 * (int(degree) % 10) + 1 for degree in orbital_basis[int(z)])
        for z in atomic_numbers
    ]
    ao_ptr = np.concatenate(([0], np.cumsum(ao_counts)))

    def pack(row_atoms: np.ndarray, col_atoms: np.ndarray) -> torch.Tensor:
        packed = np.zeros((len(row_atoms), packed_dim), dtype=np.float32)
        for index, (row_atom, col_atom) in enumerate(
            zip(row_atoms, col_atoms, strict=True)
        ):
            row_slice = slice(ao_ptr[row_atom], ao_ptr[row_atom + 1])
            col_slice = slice(ao_ptr[col_atom], ao_ptr[col_atom + 1])
            interaction = hamiltonian[row_slice, col_slice]
            key = 100 * int(atomic_numbers[row_atom]) + int(atomic_numbers[col_atom])
            for local_rows, local_cols, output_slice in orbital_template[key]:
                packed[index, output_slice] = interaction[
                    local_rows, local_cols
                ].reshape(-1)
        return torch.from_numpy(packed)

    node_atoms = np.arange(len(atomic_numbers))
    edges = edge_index.detach().cpu().numpy()
    return pack(node_atoms, node_atoms), pack(edges[0], edges[1])


@torch.no_grad()
def packed_matvec_oracle_error(
    head: OpProjectionHead,
    batch: Batch,
    target_hamiltonian: torch.Tensor,
    node_packed: torch.Tensor,
    edge_packed: torch.Tensor,
    probes: torch.Tensor,
) -> float:
    atomic_numbers = batch.atomic_numbers.to(target_hamiltonian.device)
    ao_ptr = head.block_matvec.make_ao_ptr(atomic_numbers)
    actual = torch.zeros_like(probes)
    nodes = torch.arange(len(atomic_numbers), device=target_hamiltonian.device)
    head.block_matvec.add_packed_blocks(
        actual,
        node_packed.to(target_hamiltonian.device),
        nodes,
        nodes,
        atomic_numbers,
        ao_ptr,
        probes,
    )
    edge_index = batch.edge_index.reshape(2, -1).to(target_hamiltonian.device)
    head.block_matvec.add_packed_blocks(
        actual,
        edge_packed.to(target_hamiltonian.device),
        edge_index[0],
        edge_index[1],
        atomic_numbers,
        ao_ptr,
        probes,
    )
    return float(relative_action_error(actual, target_hamiltonian @ probes).item())


@torch.no_grad()
def coupled_transform_reference_error(
    head: OpProjectionHead,
    basis_transform: e3TensorDecomp,
    *,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    generator = torch.Generator(device=device).manual_seed(seed)
    coupled = torch.randn(
        4,
        head.block_matvec.decode.coupled_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    expected = basis_transform.get_H(coupled)
    actual = head.block_matvec.decode(coupled)
    relative = (actual - expected).square().sum() / expected.square().sum().clamp_min(
        1.0e-12
    )
    return float(relative.item()), float((actual - expected).abs().max().item())


@torch.no_grad()
def evaluate(
    head: OpProjectionHead,
    features,
    batch: Batch,
    hamiltonian: torch.Tensor,
    probes: torch.Tensor,
) -> float:
    head.eval()
    callback = bind_operator_callback(features, batch, head)
    predicted = callback(probes)
    target = hamiltonian @ probes
    return float(relative_action_error(predicted, target).item())


def main() -> None:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"requested unavailable device {args.device!r}")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    database = HamiltonianDatabase(str(args.database))
    atomic_numbers, positions, _, _, hamiltonian, overlap, *_ = database[args.row]
    atomic_numbers = np.array(atomic_numbers, copy=True)
    positions = np.array(positions, copy=True)
    hamiltonian = np.array(hamiltonian, copy=True)
    overlap = np.array(overlap, copy=True)
    batch, cutoff = complete_graph_batch(
        atomic_numbers, positions, overlap, device=device
    )
    target_hamiltonian = torch.tensor(hamiltonian, dtype=torch.float32, device=device)

    cache, required_irreps, basis_transform = load_orbital_metadata(
        args.orbital_cache, device
    )
    orbital_basis = copy.deepcopy(basis_sets.orbital_basis_def2_svp_nabla)
    expected_ao = sum(
        sum(2 * (int(degree) % 10) + 1 for degree in orbital_basis[int(z)])
        for z in atomic_numbers
    )
    if target_hamiltonian.shape != (expected_ao, expected_ao):
        raise RuntimeError(
            f"basis gives {expected_ao} AO but target is {target_hamiltonian.shape}"
        )

    backbone = OpProjectionBackbone(
        required_irreps,
        sphere_channels=args.channels,
        hidden_channels=args.channels,
        lmax=required_irreps.lmax,
        mmax=required_irreps.lmax,
        cutoff=cutoff,
        edge_channels=args.channels,
        num_distance_basis=args.distance_basis,
        num_layers=1,
        output_sphere_channels=args.channels,
        conditioning_basis="def2-svp-nabla",
    ).to(device)
    features = detached_features(backbone, batch)
    head = OpProjectionHead(
        required_irreps=required_irreps,
        ls_list=cache["ls_list"],
        orbital_basis=orbital_basis,
        orbital_template=cache["orbital_template"],
        basis_transformation=basis_transform,
        sphere_channels=args.channels,
        lmax=required_irreps.lmax,
        mmax=required_irreps.lmax,
        hidden_channels=args.channels,
        edge_channels=args.channels,
        num_distance_basis=args.distance_basis,
        cutoff=cutoff,
        pair_chunk_size=args.pair_chunk_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    heldout = rademacher_probes(
        expected_ao,
        args.heldout_probes,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    initial_relative_error = evaluate(
        head, features, batch, target_hamiltonian, heldout
    )
    node_packed, edge_packed = pack_dense_blocks(
        hamiltonian,
        atomic_numbers,
        batch.edge_index.reshape(2, -1),
        orbital_basis,
        cache["orbital_template"],
        head.block_matvec.decode.packed_dim,
    )
    ao_matvec_error = packed_matvec_oracle_error(
        head,
        batch,
        target_hamiltonian,
        node_packed,
        edge_packed,
        heldout,
    )
    transform_error, transform_max_abs_error = coupled_transform_reference_error(
        head,
        basis_transform,
        device=device,
        seed=args.seed + 2,
    )
    history = []
    first_gradient_norm = None
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        head.train()
        probes = rademacher_probes(
            expected_ao,
            args.train_probes,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        target_action = target_hamiltonian @ probes
        optimizer.zero_grad(set_to_none=True)
        callback = bind_operator_callback(features, batch, head)
        predicted_action = callback(probes)
        loss = probe_matrix_mse(predicted_action, target_action)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm, grad_tensors = gradient_norm(head)
        if first_gradient_norm is None:
            first_gradient_norm = grad_norm
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=100.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            heldout_error = evaluate(head, features, batch, target_hamiltonian, heldout)
            entry = {
                "step": step,
                "train_probe_mse": float(loss.item()),
                "heldout_relative_action_error": heldout_error,
                "gradient_norm": grad_norm,
                "gradient_tensors": grad_tensors,
            }
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    elapsed = time.perf_counter() - started
    final_relative_error = evaluate(head, features, batch, target_hamiltonian, heldout)
    backbone.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    backbone.train()
    head.train()
    joint_probe = rademacher_probes(
        expected_ao,
        args.train_probes,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    joint_features = backbone(batch)
    joint_prediction = bind_operator_callback(joint_features, batch, head)(joint_probe)
    joint_loss = probe_matrix_mse(joint_prediction, target_hamiltonian @ joint_probe)
    joint_loss.backward()
    backbone_grad_norm, backbone_grad_tensors = gradient_norm(backbone)
    head_grad_norm, head_grad_tensors = gradient_norm(head)
    improvement = (initial_relative_error - final_relative_error) / max(
        initial_relative_error, 1.0e-12
    )
    result = {
        "experiment": "op_projection_single_batch_overfit",
        "dataset": str(args.database),
        "row": args.row,
        "seed": args.seed,
        "device": str(device),
        "steps": args.steps,
        "train_probes_resampled_each_step": args.train_probes,
        "heldout_probes_fixed": args.heldout_probes,
        "num_atoms": int(len(atomic_numbers)),
        "num_edges": int(batch.edge_index.shape[1]),
        "total_ao": expected_ao,
        "cutoff": cutoff,
        "channels": args.channels,
        "pair_chunk_size": args.pair_chunk_size,
        "projection_stats": head.last_projection_stats,
        "backbone_trainable_during_overfit": False,
        "head_trainable_during_overfit": True,
        "ao_packed_matvec_oracle_relative_action_error": ao_matvec_error,
        "coupled_to_packed_reference_relative_error": transform_error,
        "coupled_to_packed_reference_max_abs_error": transform_max_abs_error,
        "end_to_end_probe_loss": float(joint_loss.item()),
        "end_to_end_backbone_gradient_norm": backbone_grad_norm,
        "end_to_end_backbone_gradient_tensors": backbone_grad_tensors,
        "end_to_end_head_gradient_norm": head_grad_norm,
        "end_to_end_head_gradient_tensors": head_grad_tensors,
        "dense_matrix_role": (
            "target oracle H@Z plus one non-training AO packing diagnostic"
        ),
        "prediction_constructs_dense_matrix": False,
        "initial_heldout_relative_action_error": initial_relative_error,
        "final_heldout_relative_action_error": final_relative_error,
        "heldout_error_reduction_fraction": improvement,
        "minimum_heldout_reduction_fraction": args.min_heldout_reduction,
        "first_gradient_norm": first_gradient_norm,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "passed_learning_smoke": bool(
            first_gradient_norm is not None
            and math.isfinite(first_gradient_norm)
            and first_gradient_norm > 0.0
            and ao_matvec_error < 1.0e-10
            and transform_error < 1.0e-12
            and math.isfinite(backbone_grad_norm)
            and backbone_grad_norm > 0.0
            and improvement >= args.min_heldout_reduction
        ),
        "history": history,
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "head_state_dict": head.state_dict(),
            "result": result,
        },
        args.output_dir / "checkpoint.pt",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["passed_learning_smoke"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
