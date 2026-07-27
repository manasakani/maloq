from __future__ import annotations

import torch
from e3nn.o3 import Irreps
from torch_geometric.data import Batch, Data

from maloq.core.config import MaloqConfig
from maloq.helm.esen_osh import eSEN_Backbone


def _small_backbone(**kwargs) -> eSEN_Backbone:
    options = {
        "sphere_channels": 4,
        "hidden_channels": 4,
        "lmax": 1,
        "mmax": 1,
        "cutoff": 8.0,
        "edge_channels": 4,
        "num_layers": 1,
        "num_distance_basis": 4,
        "open_shell": False,
    }
    options.update(kwargs)
    torch.manual_seed(7)
    return eSEN_Backbone(Irreps("1x0e"), **options).eval()


def _two_atom_batch(
    charge: int = 0,
    spin: int = 1,
    *,
    include_system_metadata: bool = True,
) -> Batch:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.2, -0.1]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    fields = {
        "pos": positions,
        "atomic_numbers": torch.tensor([6, 6], dtype=torch.long),
        "edge_index": edge_index,
        "edge_attr": torch.cat(
            (displacement.norm(dim=-1, keepdim=True), displacement),
            dim=-1,
        ),
        "num_atoms_in_molecule": torch.tensor([2], dtype=torch.long),
    }
    if include_system_metadata:
        fields.update(
            charge=torch.tensor([charge], dtype=torch.long),
            spin_multiplicity=torch.tensor([spin], dtype=torch.long),
        )
    return Batch.from_data_list([Data(**fields)])


def _forward_embeddings(model, batch):
    with torch.no_grad():
        output = model(batch)
    return output["node_embeddings"], output["edge_embeddings"]


def test_element_only_is_independent_of_charge_and_spin_metadata() -> None:
    model = _small_backbone(atom_scalar_embedding_mode="element_only")
    reference = _forward_embeddings(model, _two_atom_batch(0, 1))
    corrupt = _forward_embeddings(model, _two_atom_batch(10, 17))
    missing = _forward_embeddings(
        model,
        _two_atom_batch(include_system_metadata=False),
    )

    for actual_pair in (corrupt, missing):
        for actual, expected in zip(actual_pair, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)

    assert model.charge_embedding is None
    assert model.spin_embedding is None
    assert model.scalar_node_embedding is None
    assert not any(
        key.startswith(
            (
                "charge_embedding.",
                "spin_embedding.",
                "scalar_node_embedding.",
            )
        )
        for key in model.state_dict()
    )


def test_default_embedding_mode_still_uses_charge_and_spin() -> None:
    model = _small_backbone()
    reference_node, _ = _forward_embeddings(model, _two_atom_batch(0, 1))
    charged_node, _ = _forward_embeddings(model, _two_atom_batch(1, 1))
    high_spin_node, _ = _forward_embeddings(model, _two_atom_batch(0, 3))
    assert not torch.equal(reference_node, charged_node)
    assert not torch.equal(reference_node, high_spin_node)


def test_config_round_trip_preserves_omol_paper_contract() -> None:
    config = MaloqConfig(
        dataset={
            "dataset_name": "omol",
            "dataset_format": "omol_csh_h5",
            "omol_csh_metadata_policy": "paper_contract",
            "open_shell": False,
        },
        model={
            "backbone_type": "esen",
            "atom_scalar_embedding_mode": "element_only",
        },
    ).to_workflow_config()
    assert config["dataset_format"] == "omol_csh_h5"
    assert config["omol_csh_metadata_policy"] == "paper_contract"
    assert config["atom_scalar_embedding_mode"] == "element_only"
