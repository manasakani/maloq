from __future__ import annotations

import importlib
from types import MethodType

import numpy as np
import torch

from maloq.fock_utils.fock_targets_batched import Fock_Targets
from maloq.fock_utils.utils_orca_out import sort_by_m


def test_nabladft_loader_leaves_qhflow3_optional_initial_matrices_absent(
    monkeypatch,
):
    get_loader_module = importlib.import_module("maloq.dataset_utils.get_loader")

    class FakeFockTargets:
        def __init__(
            self,
            atomic_numbers,
            positions,
            _rcut,
            orbital_basis,
            _hamiltonians,
            **_kwargs,
        ):
            self.atomic_positions_list = positions
            self.atomic_numbers_list = atomic_numbers
            self.neighbour_list_list = [
                np.asarray([[0, 1], [1, 0]], dtype=np.int64)
            ]
            self.edge_dist_list = [torch.zeros(2, 4)]
            self.edge_labels_list = [torch.zeros(1, 2, 1)]
            self.node_labels_list = [torch.zeros(1, 2, 1)]
            self.edge_unpadding_mask_list = [
                torch.ones(1, 2, 1, dtype=torch.bool)
            ]
            self.node_unpadding_mask_list = [
                torch.ones(1, 2, 1, dtype=torch.bool)
            ]
            self.orbital_basis = orbital_basis
            self.ls_list = torch.zeros(1, dtype=torch.long)
            self.req_output_irreps = "1x0e"
            self.basis_transformation = object()
            self.orbital_starts = torch.zeros(1, dtype=torch.long)

    monkeypatch.setattr(get_loader_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(get_loader_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(get_loader_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(
        get_loader_module.fock_targets_batched,
        "Fock_Targets",
        FakeFockTargets,
    )

    atomic_numbers = np.asarray([1, 1])
    positions = np.asarray([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    matrix = np.eye(10, dtype=np.float32)
    database = [
        (
            atomic_numbers,
            positions,
            0.0,
            np.zeros((2, 3), dtype=np.float32),
            matrix,
            matrix,
            None,
            0,
            0,
        )
    ]

    loader, *_ = get_loader_module.get_loader(
        database=database,
        start_idx=0,
        end_idx=1,
        dataset_name="nablaDFT",
        rcut=8.0,
        batch_size=1,
        load_delta_auxiliary_matrix=True,
    )

    data = loader.dataset[0]
    assert getattr(data, "initial_density_matrix", None) is None
    assert getattr(data, "initial_hamiltonian", None) is None
    np.testing.assert_array_equal(data.overlap_matrix, matrix)


def _reference_sort_by_m(
    matrix: np.ndarray,
    orbital_basis: dict[int, list[int]],
    atomic_numbers: np.ndarray,
    direction: str,
) -> np.ndarray:
    if direction in {"orca_to_e3nn", "e3nn_to_orca"}:
        conversions = {
            0: [0],
            1: [2, 0, 1],
            2: [4, 2, 0, 1, 3],
            3: [6, 4, 2, 0, 1, 3, 5],
            4: [8, 6, 4, 2, 0, 1, 3, 5, 7],
        }
    else:
        conversions = {
            0: [0],
            1: [2, 0, 1],
            2: [0, 1, 2, 3, 4],
            3: [0, 1, 2, 3, 4, 5, 6],
            4: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    reflections = {
        0: [1],
        1: [1, 1, 1],
        2: [1, 1, 1, 1, 1],
        3: [-1, 1, 1, 1, 1, 1, -1],
        4: [-1, -1, 1, 1, 1, 1, 1, -1, -1],
    }
    full_orbitals = np.hstack(
        [orbital_basis[int(z)] for z in atomic_numbers]
    )
    result = matrix.copy()
    start = 0
    for orbital_l in full_orbitals:
        orbital_l = int(orbital_l) % 10
        size = 2 * orbital_l + 1
        stop = start + size
        permutation = np.zeros((size, size))
        for i, j in enumerate(conversions[orbital_l]):
            permutation[i, j] = 1
        reflection = np.diag(reflections[orbital_l])
        if direction in {"orca_to_e3nn", "e3nn_to_pyscf"}:
            transform = reflection @ permutation
        else:
            transform = permutation.T @ reflection
        result[start:stop, :] = transform @ result[start:stop, :]
        result[:, start:stop] = result[:, start:stop] @ transform.T
        start = stop
    return result


def test_sort_by_m_signed_permutation_matches_original_algorithm():
    orbital_basis = {
        1: [0, 1],
        6: [0, 0, 0, 1, 1, 2],
    }
    atomic_numbers = np.asarray([1, 6, 1])
    size = sum(
        2 * orbital_l + 1
        for z in atomic_numbers
        for orbital_l in orbital_basis[int(z)]
    )
    rng = np.random.default_rng(44)
    matrix = rng.normal(size=(size, size))

    for direction in (
        "orca_to_e3nn",
        "e3nn_to_orca",
        "e3nn_to_pyscf",
        "pyscf_to_e3nn",
    ):
        np.testing.assert_array_equal(
            sort_by_m(matrix, orbital_basis, atomic_numbers, direction),
            _reference_sort_by_m(
                matrix, orbital_basis, atomic_numbers, direction
            ),
        )


def test_vectorized_label_masks_match_element_pair_slices_and_stay_on_cpu():
    target = Fock_Targets.__new__(Fock_Targets)
    target.max_num_elements = 100
    target.target_len = 8
    target.distribute_graphs = False
    target.atomic_numbers_list = [np.asarray([1, 6])]
    target.neighbour_list_list = [np.asarray([[0, 1], [1, 0]])]
    target.orbital_template = [[] for _ in range(607)]
    target.orbital_template[101] = [(slice(0, 1), slice(0, 1), slice(0, 2))]
    target.orbital_template[106] = [(slice(0, 1), slice(0, 1), slice(2, 4))]
    target.orbital_template[601] = [(slice(0, 1), slice(0, 1), slice(4, 6))]
    target.orbital_template[606] = [(slice(0, 1), slice(0, 1), slice(6, 8))]

    target.create_label_unpadding_mask()

    expected_edges = torch.tensor(
        [[[False, False, True, True, False, False, False, False],
          [False, False, False, False, True, True, False, False]]]
    )
    expected_nodes = torch.tensor(
        [[[True, True, False, False, False, False, False, False],
          [False, False, False, False, False, False, True, True]]]
    )
    torch.testing.assert_close(target.edge_unpadding_mask_list[0], expected_edges)
    torch.testing.assert_close(target.node_unpadding_mask_list[0], expected_nodes)
    assert target.edge_unpadding_mask_list[0].device.type == "cpu"
    assert target.node_unpadding_mask_list[0].device.type == "cpu"


def test_additional_targets_restore_primary_target_state():
    target = Fock_Targets.__new__(Fock_Targets)
    primary_nodes = [torch.tensor([1.0])]
    primary_edges = [torch.tensor([2.0])]
    target.node_labels_list = primary_nodes
    target.edge_labels_list = primary_edges
    target.target_len = 4

    def fake_make_targets(self, matrices):
        assert matrices == ["initial"]
        self.node_labels_list = [torch.tensor([3.0])]
        self.edge_labels_list = [torch.tensor([4.0])]
        self.target_len = 9

    target.make_targets = MethodType(fake_make_targets, target)
    extra_nodes, extra_edges = target.make_additional_targets(["initial"])

    torch.testing.assert_close(extra_nodes[0], torch.tensor([3.0]))
    torch.testing.assert_close(extra_edges[0], torch.tensor([4.0]))
    assert target.node_labels_list is primary_nodes
    assert target.edge_labels_list is primary_edges
    assert target.target_len == 4
