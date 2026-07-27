import numpy as np

from dft_dataset.conventions import (
    AOShellSpec,
    build_orca_raw_density_layout_transform,
    build_orca_json_signs_for_layout,
    build_layout_reorder_indices,
    build_orca_shell_specs_from_cache,
    invert_reorder_indices,
    layout_matrix_to_orca_raw_density_layout,
    pyscf_overlap_to_orca_raw_density_layout,
    reorder_matrix,
    signed_reorder_matrix,
)


def test_orca_layout_reorder_roundtrip_with_shell_order_change():
    # ORCA raw order can place a diffuse shell after higher-l shells.
    src_orca = [
        AOShellSpec(0, "1s", 0, 1),
        AOShellSpec(0, "2p", 1, 3),
        AOShellSpec(0, "3d", 2, 5),
        AOShellSpec(0, "2s", 0, 1),
    ]
    dst_pyscf = [
        AOShellSpec(0, "1s", 0, 1),
        AOShellSpec(0, "2s", 0, 1),
        AOShellSpec(0, "2p", 1, 3),
        AOShellSpec(0, "3d", 2, 5),
    ]

    idx = build_layout_reorder_indices(src_orca, dst_pyscf, "orca", "pyscf")
    inv = invert_reorder_indices(idx)

    rng = np.random.default_rng(0)
    M = rng.normal(size=(10, 10))
    M = 0.5 * (M + M.T)

    assert np.allclose(reorder_matrix(reorder_matrix(M, idx), inv), M)


def test_orca_to_e3nn_is_distinct_from_orca_to_pyscf_for_p_shells():
    src_orca = [AOShellSpec(0, "2p", 1, 3)]
    dst = [AOShellSpec(0, "2p", 1, 3)]

    idx_pyscf = build_layout_reorder_indices(src_orca, dst, "orca", "pyscf")
    idx_e3nn = build_layout_reorder_indices(src_orca, dst, "orca", "e3nn")

    # ORCA p=(z,x,y), PySCF p=(x,y,z), e3nn p=(y,z,x) in this m encoding.
    assert idx_pyscf.tolist() == [1, 2, 0]
    assert idx_e3nn.tolist() == [2, 0, 1]


def test_orca_shell_specs_from_cache_reorders_diffuse_shells():
    class FakeMol:
        def ao_labels(self):
            labels = []
            for shell_id, suffixes in [
                ("1s", [""]),
                ("2s", [""]),
                ("3s", [""]),
                ("4s", [""]),
                ("5s", [""]),
                ("6s", [""]),
                ("2p", ["x", "y", "z"]),
                ("3p", ["x", "y", "z"]),
                ("4p", ["x", "y", "z"]),
                ("3d", ["xy", "yz", "z^2", "xz", "x2-y2"]),
                ("4d", ["xy", "yz", "z^2", "xz", "x2-y2"]),
                ("5d", ["xy", "yz", "z^2", "xz", "x2-y2"]),
                ("4f", ["-3", "-2", "-1", "0", "+1", "+2", "+3"]),
            ]:
                for suffix in suffixes:
                    labels.append(f"0 C {shell_id}{suffix}")
            return labels

        def atom_charges(self):
            return np.asarray([6])

    specs = build_orca_shell_specs_from_cache(FakeMol(), "def2-tzvpd")
    assert [s.shell_id for s in specs] == [
        "1s", "2s", "3s", "4s", "5s",
        "2p", "3p", "4p",
        "3d", "4d", "4f",
        "6s", "5d",
    ]


def test_orca_json_signs_flip_f_extreme_m_components():
    shells = [
        AOShellSpec(0, "1s", 0, 1),
        AOShellSpec(0, "4f", 3, 7),
    ]

    signs = build_orca_json_signs_for_layout(shells, "pyscf")

    assert signs.tolist() == [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]


def test_signed_reorder_preserves_trace_when_density_and_overlap_transform_together():
    shells = [
        AOShellSpec(0, "1s", 0, 1),
        AOShellSpec(0, "4f", 3, 7),
    ]
    idx = np.arange(8)
    signs = build_orca_json_signs_for_layout(shells, "pyscf")

    rng = np.random.default_rng(1)
    D = rng.normal(size=(8, 8))
    D = 0.5 * (D + D.T)
    A = rng.normal(size=(8, 8))
    S = A.T @ A + np.eye(8)

    D_t = signed_reorder_matrix(D, idx, signs)
    S_t = signed_reorder_matrix(S, idx, signs)

    assert np.allclose(np.trace(D_t @ S_t), np.trace(D @ S))


def test_pyscf_overlap_to_orca_raw_density_layout_matches_density_sign():
    class FakeMol:
        def __init__(self, overlap):
            self._overlap = overlap

        def ao_labels(self):
            return [
                "0 C 1s",
                "0 C 4f-3",
                "0 C 4f-2",
                "0 C 4f-1",
                "0 C 4f+0",
                "0 C 4f+1",
                "0 C 4f+2",
                "0 C 4f+3",
            ]

        def intor(self, name):
            assert name == "int1e_ovlp"
            return self._overlap

        def atom_charges(self):
            return np.asarray([6])

    shells = [
        AOShellSpec(0, "1s", 0, 1),
        AOShellSpec(0, "4f", 3, 7),
    ]
    signs = build_orca_json_signs_for_layout(shells, "pyscf")

    rng = np.random.default_rng(2)
    D = rng.normal(size=(8, 8))
    D = 0.5 * (D + D.T)
    A = rng.normal(size=(8, 8))
    S = A.T @ A + np.eye(8)
    mol = FakeMol(S)

    S_raw = pyscf_overlap_to_orca_raw_density_layout(
        mol,
        "def2-tzvpd",
        dst_convention="pyscf",
    )
    S_raw_from_layout = layout_matrix_to_orca_raw_density_layout(
        S,
        mol,
        "def2-tzvpd",
        src_convention="pyscf",
        dst_convention="pyscf",
    )
    D_raw = (signs[:, None] * D) * signs[None, :]

    assert np.allclose(S_raw, S_raw_from_layout)
    assert np.allclose(S_raw, (signs[:, None] * S) * signs[None, :])
    assert np.allclose(np.trace(D_raw @ S_raw), np.trace(D @ S))


def test_orca_raw_density_layout_applies_be_def2_tzvpd_p_signed_swap():
    class FakeBeMol:
        def ao_labels(self):
            labels = []
            for shell_id in ("2p", "3p", "4p", "5p"):
                for suffix in ("x", "y", "z"):
                    labels.append(f"0 Be {shell_id}{suffix}")
            return labels

        def atom_charges(self):
            return np.asarray([4])

    mol = FakeBeMol()
    radial = np.asarray([
        [1.0, -0.232, -0.598, -0.311],
        [-0.232, 1.0, 0.032, 0.013],
        [-0.598, 0.032, 1.0, 0.830],
        [-0.311, 0.013, 0.830, 1.0],
    ])
    M = np.zeros((12, 12), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for m in range(3):
                M[3 * i + m, 3 * j + m] = radial[i, j]

    idx, signs = build_orca_raw_density_layout_transform(
        mol,
        "def2-tzvpd",
        src_convention="e3nn",
        dst_convention="e3nn",
    )
    out = signed_reorder_matrix(M, idx, signs)

    expected_radial = np.asarray([
        [1.0, 0.232, -0.032, -0.013],
        [0.232, 1.0, -0.598, -0.311],
        [-0.032, -0.598, 1.0, 0.830],
        [-0.013, -0.311, 0.830, 1.0],
    ])
    expected = np.zeros((12, 12), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for m in range(3):
                expected[3 * i + m, 3 * j + m] = expected_radial[i, j]

    assert idx.tolist() == [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11]
    assert signs.tolist() == [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert np.allclose(out, expected)
