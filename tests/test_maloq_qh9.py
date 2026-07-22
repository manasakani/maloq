from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from e3nn.o3 import Irreps
from torch_geometric.data import Batch, Data

from maloq.core.config import MaloqConfig
from maloq.dataset_utils.get_loader import _qm7_matrix_target
from maloq.helm.esen_block import DegreeLayerScale, GridAtomwise
from maloq.helm.esen_osh import eSEN_Backbone
from maloq.helm.nn.activation import GateActivation
from maloq.helm.qhflow3_clean import (
    GridAtomwise as QHFlow3GridAtomwise,
    QHFlow3MaloqBackbone,
    _orbital_masks_for_basis,
)
from maloq.train_utils.splittrainer import SplitTrainer


def _load_comparison_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "_my_script/experiment/2026-07-21/compare_maloq_qh9.py"
    )
    spec = importlib.util.spec_from_file_location("compare_maloq_qh9", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_activation_supports_qhflow3_sigmoid_gate():
    sigmoid = GateActivation(1, 1, 2, gate_act_type="sigmoid")
    tanh = GateActivation(1, 1, 2, gate_act_type="tanh")
    gating = torch.zeros(1, 2)
    features = torch.ones(1, 4, 2)

    sigmoid_out = sigmoid(gating, features)
    tanh_out = tanh(gating, features)

    torch.testing.assert_close(sigmoid_out[:, 1:], torch.full((1, 3, 2), 0.5))
    torch.testing.assert_close(tanh_out[:, 1:], torch.zeros(1, 3, 2))


def test_bounded_degree_layerscale_has_reference_initialization_and_bounds():
    scale = DegreeLayerScale(
        lmax=4,
        mode="bounded_degree",
        init=1.0 / 64.0,
        log_range=math.log(64.0),
    )
    torch.testing.assert_close(
        scale.degree_scales(),
        torch.full((5,), 1.0 / 64.0),
    )

    with torch.no_grad():
        scale.raw.copy_(torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0]))
    values = scale.degree_scales()
    assert values.min().item() >= (1.0 / 4096.0) * (1.0 - 1.0e-6)
    assert values.max().item() <= 1.0 + 1.0e-6


def test_maloq_qh9_backbone_contract_matches_reference_axes():
    model = eSEN_Backbone(
        Irreps("1x0e"),
        sphere_channels=8,
        hidden_channels=8,
        lmax=4,
        mmax=4,
        cutoff=15.0,
        edge_channels=8,
        num_layers=3,
        num_edge_layers=2,
        num_distance_basis=16,
        gate_act_type="sigmoid",
        mlp_type="grid",
        message_passing_schedule="node_then_edge",
        output_sphere_channels=4,
        use_edge_envelope=True,
        use_edge_scalar_modulation=True,
        residual_update_scale_mode="bounded_degree",
        residual_update_scale_init=1.0 / 64.0,
        residual_update_scale_log_range=math.log(64.0),
    )

    assert len(model.node_blocks) == 3
    assert len(model.edge_blocks) == 2
    assert model.output_sphere_channels == 4
    assert model.message_passing_schedule == "node_then_edge"
    assert all(isinstance(block.atom_wise, GridAtomwise) for block in model.node_blocks)
    assert all(isinstance(block.atom_wise, GridAtomwise) for block in model.edge_blocks)
    assert all(block.edge_wise.use_edge_envelope for block in model.node_blocks)
    assert all(
        block.edge_wise.use_edge_scalar_modulation for block in model.edge_blocks
    )

    scale_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("_update_scale.raw")
    ]
    assert len(scale_parameters) == 10
    assert sum(parameter.numel() for parameter in scale_parameters) == 50


def test_maloq_qh9_config_round_trip_preserves_model_recipe():
    config = MaloqConfig(
        model={
            "model_variant": "maloq-qh9",
            "gate_act_type": "sigmoid",
            "mlp_type": "grid",
            "message_passing_schedule": "node_then_edge",
            "num_edge_layers": 2,
            "output_l_embedding_dim": 64,
            "use_edge_envelope": True,
            "use_edge_scalar_modulation": True,
            "residual_update_scale_mode": "bounded_degree",
            "residual_update_scale_init": 1.0 / 64.0,
            "residual_update_scale_log_range": math.log(64.0),
        },
        optimization={
            "scheduler_type": "warmup_polynomial",
            "warmup_steps": 1000,
            "gradient_clip_val": 1.0,
        },
        runtime={"seed": 44},
    )
    workflow = config.to_workflow_config()

    assert workflow["model_variant"] == "maloq-qh9"
    assert workflow["message_passing_schedule"] == "node_then_edge"
    assert workflow["output_l_embedding_dim"] == 64
    assert workflow["scheduler_type"] == "warmup_polynomial"
    assert workflow["gradient_clip_val"] == 1.0
    assert workflow["seed"] == 44


def test_delta_learning_config_round_trip_for_hamiltonian_and_density():
    for loss_target in ("fock_matrix", "density_matrix"):
        workflow = MaloqConfig(
            loss={"loss_target": loss_target, "delta_learning": True}
        ).to_workflow_config()

        assert workflow["loss_target"] == loss_target
        assert workflow["delta_learning"] is True


def test_wandb_tracking_defaults_to_ten_optimizer_steps():
    workflow = MaloqConfig().to_workflow_config()

    assert workflow["wandb_log_every_n_steps"] == 10
    assert SplitTrainer._should_log_wandb_step(10, 9, 11, 10)
    assert not SplitTrainer._should_log_wandb_step(9, 8, 11, 10)
    # The epoch summary owns the final step so W&B never receives that step twice.
    assert not SplitTrainer._should_log_wandb_step(10, 9, 10, 10)


def test_gradient_accumulation_config_and_optimizer_step_count():
    workflow = MaloqConfig(
        optimization={"gradient_accumulation_steps": 2}
    ).to_workflow_config()

    assert workflow["gradient_accumulation_steps"] == 2
    assert SplitTrainer.optimizer_steps_per_epoch(1208, 2) == 604
    assert SplitTrainer.optimizer_steps_per_epoch(3, 2) == 2

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        SplitTrainer.optimizer_steps_per_epoch(1, 0)


def test_comparison_loss_reader_uses_persisted_edge_node_order(tmp_path):
    module = _load_comparison_module()
    (tmp_path / "head_training_loss.txt").write_text("4.0\t2.0\n")
    (tmp_path / "head_validation_loss.txt").write_text("3.0\t1.0\n")

    assert module.last_losses(tmp_path) == {
        "train_node_loss": 2.0,
        "train_edge_loss": 4.0,
        "validation_node_loss": 1.0,
        "validation_edge_loss": 3.0,
    }


def test_qm7_loader_selects_explicit_density_target():
    hamiltonian = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    density = torch.tensor([[0.3, 0.1], [0.1, 0.7]])
    row = {"hamiltonian": hamiltonian, "density_matrix": density}

    torch.testing.assert_close(
        torch.from_numpy(_qm7_matrix_target(row, "fock_matrix")),
        hamiltonian,
    )
    torch.testing.assert_close(
        torch.from_numpy(_qm7_matrix_target(row, "density_matrix")),
        density,
    )


def test_qm7_loader_rejects_missing_density_target():
    row = {"hamiltonian": torch.eye(2)}
    try:
        _qm7_matrix_target(row, "density_matrix")
    except KeyError as error:
        assert "density_matrix" in str(error)
    else:
        raise AssertionError("Missing density target should fail explicitly")


def test_qhflow3_native_overlap_bridge_uses_def2_svp_padding():
    batch = SimpleNamespace(
        overlap_matrix=[torch.eye(19).numpy()],
        ptr=torch.tensor([0, 2]),
        atomic_numbers=torch.tensor([1, 6]),
        pos=torch.zeros(2, 3),
    )

    backbone = QHFlow3MaloqBackbone.__new__(QHFlow3MaloqBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.basis = "def2-svp"
    backbone._orbital_masks, backbone.output_matrix_dim = _orbital_masks_for_basis(
        backbone.basis
    )
    blocks = backbone._overlap_blocks(batch)

    assert blocks.shape == (2, 14, 14)
    hydrogen_mask = torch.tensor([0, 1, 3, 4, 5])
    torch.testing.assert_close(
        blocks[0][hydrogen_mask[:, None], hydrogen_mask[None, :]],
        torch.eye(5),
    )
    torch.testing.assert_close(blocks[1], torch.eye(14))


def test_qhflow3_native_matrix_bridge_handles_delta_inputs():
    initial_density = torch.diag(torch.arange(1, 20, dtype=torch.float32))
    batch = SimpleNamespace(
        initial_density_matrix=[initial_density.numpy()],
        ptr=torch.tensor([0, 2]),
        atomic_numbers=torch.tensor([1, 6]),
        pos=torch.zeros(2, 3),
    )

    backbone = QHFlow3MaloqBackbone.__new__(QHFlow3MaloqBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.basis = "def2-svp"
    backbone._orbital_masks, backbone.output_matrix_dim = _orbital_masks_for_basis(
        backbone.basis
    )
    blocks = backbone._matrix_blocks(
        batch,
        "initial_density_matrix",
        "initial density",
    )

    assert blocks.shape == (2, 14, 14)
    hydrogen_mask = torch.tensor([0, 1, 3, 4, 5])
    torch.testing.assert_close(
        blocks[0][hydrogen_mask[:, None], hydrogen_mask[None, :]],
        initial_density[:5, :5],
    )
    torch.testing.assert_close(blocks[1], initial_density[5:, 5:])


def test_qhflow3_nabladft_bridge_uses_32_ao_padding_for_heavy_elements():
    masks, matrix_dim = _orbital_masks_for_basis("def2-svp-nabla")

    assert matrix_dim == 32
    assert masks[1].tolist() == [0, 1, 5, 6, 7]
    assert masks[6].numel() == 14
    assert masks[16].numel() == 18
    assert masks[17].numel() == 18
    assert masks[35].tolist() == list(range(32))

    orbital_count = 5 + 14 + 18 + 32
    batch = SimpleNamespace(
        overlap_matrix=[torch.eye(orbital_count).numpy()],
        ptr=torch.tensor([0, 4]),
        atomic_numbers=torch.tensor([1, 6, 16, 35]),
        pos=torch.zeros(4, 3),
    )
    backbone = QHFlow3MaloqBackbone.__new__(QHFlow3MaloqBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.basis = "def2-svp-nabla"
    backbone._orbital_masks = masks
    backbone.output_matrix_dim = matrix_dim

    blocks = backbone._overlap_blocks(batch)

    assert blocks.shape == (4, 32, 32)
    for atom_index, atomic_number in enumerate((1, 6, 16, 35)):
        mask = masks[atomic_number]
        torch.testing.assert_close(
            blocks[atom_index][mask[:, None], mask[None, :]],
            torch.eye(mask.numel()),
        )


def test_splittrainer_adds_delta_baseline_to_closed_shell_outputs():
    node_delta = torch.tensor([[[0.1, -0.2], [0.3, 0.4]]])
    edge_delta = torch.tensor([[[0.5, -0.6]]])
    batch = SimpleNamespace(
        delta_node_base=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        delta_edge_base=torch.tensor([[5.0, 6.0]]),
    )

    node_final, edge_final = SplitTrainer.apply_delta_baseline(
        node_delta, edge_delta, batch
    )

    torch.testing.assert_close(
        node_final,
        torch.tensor([[[1.1, 1.8], [3.3, 4.4]]]),
    )
    torch.testing.assert_close(edge_final, torch.tensor([[[5.5, 5.4]]]))


def test_qhflow3_config_selects_headless_native_bridge():
    workflow = MaloqConfig(
        model={
            "model_variant": "qhflow3-maloq-head",
            "backbone_type": "qhflow3_clean",
            "output_l_embedding_dim": 64,
            "num_edge_layers": 2,
        }
    ).to_workflow_config()

    assert workflow["backbone_type"] == "qhflow3_clean"
    assert workflow["output_l_embedding_dim"] == 64
    assert workflow["qhflow3_grid_resolution"] == 48
    assert workflow["qhflow3_grid_ffn_chunk_size"] == 512


def _qhflow3_equivariance_batch(
    positions: torch.Tensor,
    overlap: torch.Tensor,
    device: torch.device,
    atomic_number: int = 6,
) -> Batch:
    atomic_numbers = torch.tensor(
        [atomic_number, atomic_number], dtype=torch.long, device=device
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    positions = positions.to(device)
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    edge_attr = torch.cat(
        [
            displacement.norm(dim=-1, keepdim=True),
            displacement[:, [2, 0, 1]],
        ],
        dim=-1,
    )
    data = Data(
        pos=positions,
        z=atomic_numbers,
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_atoms_in_molecule=torch.tensor([2], device=device),
        charge=torch.zeros(1, dtype=torch.long, device=device),
        spin_multiplicity=torch.ones(1, dtype=torch.long, device=device),
    )
    batch = Batch.from_data_list([data]).to(device)
    batch.overlap_matrix = [overlap.cpu().numpy()]
    return batch


def test_qhflow3_matrix_conditioning_is_atom_aligned():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(44)
    model = (
        QHFlow3MaloqBackbone(
            sh_lmax=2,
            hidden_size=8,
            bottle_hidden_size=4,
            num_gnn_layers=1,
            num_ham_gnn_layers=1,
            max_radius=12.0,
            radius_embed_dim=8,
            escn_edge_channels=8,
            escn_num_distance_basis=8,
            esen_max_radius=15.0,
            basis="def2-svp",
            grid_resolution=12,
            grid_ffn_chunk_size=None,
            delta_learning=True,
            delta_target="density_matrix",
        )
        .to(device)
        .eval()
    )
    positions = torch.tensor(
        [[0.13, -0.21, 0.08], [0.74, 0.35, -0.42]],
        dtype=torch.float32,
    )
    atom_block = torch.eye(14)
    matrices = {
        "overlap_matrix": torch.block_diag(atom_block, 2.0 * atom_block),
        "initial_density_matrix": torch.block_diag(
            3.0 * atom_block,
            4.0 * atom_block,
        ),
        "initial_hamiltonian": torch.block_diag(
            5.0 * atom_block,
            6.0 * atom_block,
        ),
    }
    captured_mix_inputs = []

    def capture_mix_input(_module, inputs):
        captured_mix_inputs.append(inputs[0].detach().clone())

    handle = model.node_attr_backbone.mix_matrix.register_forward_pre_hook(
        capture_mix_input
    )
    try:
        with torch.no_grad():
            for changed_attribute in (None, *matrices):
                batch = _qhflow3_equivariance_batch(
                    positions,
                    matrices["overlap_matrix"],
                    device,
                )
                for attribute, matrix in matrices.items():
                    value = matrix.clone()
                    if attribute == changed_attribute:
                        value[14:, 14:] *= 1.5
                    setattr(batch, attribute, [value.cpu().numpy()])
                model(batch)
    finally:
        handle.remove()

    reference = captured_mix_inputs[0]
    assert len(captured_mix_inputs) == 4
    for changed in captured_mix_inputs[1:]:
        torch.testing.assert_close(changed[0], reference[0])
        assert torch.max(torch.abs(changed[1] - reference[1])).item() > 1.0e-5


@pytest.mark.parametrize(
    ("basis", "atomic_number", "basis_irreps", "matrix_dim"),
    (
        ("def2-svp", 6, "3x0e + 2x1e + 1x2e", 14),
        ("def2-svp-nabla", 35, "5x0e + 4x1e + 3x2e", 32),
    ),
)
def test_qhflow3_grid48_backbone_is_equivariant_for_general_rotation(
    basis,
    atomic_number,
    basis_irreps,
    matrix_dim,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(44)
    model = (
        QHFlow3MaloqBackbone(
            sh_lmax=4,
            hidden_size=8,
            bottle_hidden_size=4,
            num_gnn_layers=1,
            num_ham_gnn_layers=1,
            max_radius=12.0,
            radius_embed_dim=8,
            escn_edge_channels=8,
            escn_num_distance_basis=8,
            esen_max_radius=15.0,
            basis=basis,
        )
        .to(device)
        .eval()
    )
    assert model.grid_resolution == 48
    assert model.grid_ffn_chunk_size == 512
    grid_modules = [
        module for module in model.modules() if isinstance(module, QHFlow3GridAtomwise)
    ]
    assert all(
        getattr(module, "grid_ffn_chunk_size", None) == 512 for module in grid_modules
    )
    # Force the checkpointed branch on this two-node test graph; production
    # keeps the same math but uses chunks of 512 nodes/pairs.
    for module in grid_modules:
        module.grid_ffn_chunk_size = 1

    positions = torch.tensor(
        [[0.13, -0.21, 0.08], [0.74, 0.35, -0.42]],
        dtype=torch.float32,
    )
    basis_irreps = Irreps(basis_irreps)
    generator = torch.Generator().manual_seed(145)
    overlap_blocks = []
    for _ in range(2):
        raw = torch.randn(matrix_dim, matrix_dim, generator=generator)
        overlap_blocks.append(raw @ raw.T / matrix_dim + 0.25 * torch.eye(matrix_dim))
    overlap = torch.block_diag(*overlap_blocks)

    rotation = o3.angles_to_matrix(
        torch.tensor(0.37),
        torch.tensor(1.11),
        torch.tensor(-0.62),
    )
    basis_rotation = basis_irreps.D_from_matrix(rotation)
    rotated_overlap = torch.block_diag(
        *(basis_rotation @ block @ basis_rotation.T for block in overlap_blocks)
    )

    with torch.no_grad():
        reference = model(
            _qhflow3_equivariance_batch(positions, overlap, device, atomic_number)
        )
        observed = model(
            _qhflow3_equivariance_batch(
                positions @ rotation.T,
                rotated_overlap,
                device,
                atomic_number,
            )
        )

    for embedding_name in ("node_embeddings", "edge_embeddings"):
        for degree in range(5):
            component_slice = slice(degree**2, (degree + 1) ** 2)
            degree_rotation = o3.Irrep(degree, 1).D_from_matrix(rotation).to(device)
            expected = torch.einsum(
                "ij,njc->nic",
                degree_rotation,
                reference[embedding_name][:, component_slice, :],
            )
            torch.testing.assert_close(
                observed[embedding_name][:, component_slice, :],
                expected,
                atol=1.0e-4,
                rtol=1.0e-4,
            )

    model.train()
    training_output = model(
        _qhflow3_equivariance_batch(positions, overlap, device, atomic_number)
    )
    training_loss = sum(
        embedding.square().mean() for embedding in training_output.values()
    )
    training_loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
