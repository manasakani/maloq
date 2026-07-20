# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

"""
Tests for the Wigner D-matrix Triton kernel (wigner_backend="triton").
Validates correctness against PyTorch reference for lmax 0-8.
"""

import sys
import os
import pytest
import torch

from maloq.helm.common.rotation import (
    init_edge_rot_euler_angles,
    eulers_to_wigner,
    wigner_D,
)
from maloq.helm.triton_kernels import edge_vec_to_wigner_fused, extract_euler_angles

import maloq.helm
helm_path = os.path.dirname(maloq.helm.__file__)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def jd_list(device):
    jd = torch.load(os.path.join(helm_path, "Jd.pt"), weights_only=True)
    return [j.to(device).float() for j in jd]


def _make_edge_vecs(n, device, seed=42):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(n, 3, device=device, dtype=torch.float32, generator=gen)


def _pytorch_wigner(edge_vecs, jd_list, lmax):
    """PyTorch path (same as _get_rotmat_and_wigner with wigner_backend='torch')."""
    euler_angles = init_edge_rot_euler_angles(edge_vecs)
    wigner = eulers_to_wigner(euler_angles, 0, lmax, jd_list)
    return wigner


# ---------------------------------------------------------------------------
# 1. Direct per-L numerical comparison: Triton vs PyTorch wigner_D
#    This is the gold-standard correctness test. If each block matches
#    wigner_D element-wise, orthogonality / determinant / structure follow.
# ---------------------------------------------------------------------------
class TestNumericalCorrectness:

    @pytest.mark.parametrize("lmax", list(range(0, 9)))
    @pytest.mark.parametrize("num_edges", [1, 16, 128])
    def test_per_L_block_matches_pytorch(self, lmax, num_edges, device, jd_list):
        """Every L block must match PyTorch wigner_D to high precision."""
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries, need {lmax+1}")

        seed = 12345
        edge_vecs = _make_edge_vecs(num_edges, device, seed=lmax * 77 + num_edges)
        wigner = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=lmax, seed=seed)

        eulers = extract_euler_angles(edge_vecs, seed=seed)
        alpha, beta, gamma = eulers[:, 0], eulers[:, 1], eulers[:, 2]

        start = 0
        for l in range(lmax + 1):
            sz = 2 * l + 1
            end = start + sz
            triton_block = wigner[:, start:end, start:end]
            pytorch_block = wigner_D(l, alpha, beta, gamma, jd_list)
            err = (triton_block - pytorch_block).abs().max().item()
            assert err < 1e-4, \
                f"L={l} mismatch (lmax={lmax}, n={num_edges}): max_err={err:.6f}"
            start = end

    @pytest.mark.parametrize("l_val", list(range(0, 9)))
    def test_individual_L_isolated(self, l_val, device, jd_list):
        """Test each L in isolation (lmax = l_val)."""
        if len(jd_list) <= l_val:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries, need {l_val+1}")

        seed = 9999
        edge_vecs = _make_edge_vecs(64, device, seed=l_val * 31)
        wigner = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=l_val, seed=seed)
        eulers = extract_euler_angles(edge_vecs, seed=seed)
        alpha, beta, gamma = eulers[:, 0], eulers[:, 1], eulers[:, 2]

        start = l_val * l_val
        sz = 2 * l_val + 1
        triton_block = wigner[:, start:start+sz, start:start+sz]
        pytorch_block = wigner_D(l_val, alpha, beta, gamma, jd_list)
        err = (triton_block - pytorch_block).abs().max().item()
        assert err < 1e-4, f"L={l_val} (isolated) mismatch: max_err={err:.6f}"


# ---------------------------------------------------------------------------
# 2. Both-paths equivalence (Triton vs PyTorch end-to-end)
# ---------------------------------------------------------------------------
class TestBothPaths:

    @pytest.mark.parametrize("lmax", [2, 4, 6, 8])
    def test_shape_and_dtype_match(self, lmax, device, jd_list):
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries")
        edge_vecs = _make_edge_vecs(32, device)
        w_pt = _pytorch_wigner(edge_vecs, jd_list, lmax)
        w_tr = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=lmax)
        assert w_pt.shape == w_tr.shape
        assert w_pt.dtype == w_tr.dtype

    @pytest.mark.parametrize("lmax", [2, 4, 6, 8])
    def test_wigner_times_inv_is_identity(self, lmax, device, jd_list):
        """wigner @ wigner^T == I for both paths."""
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries")
        edge_vecs = _make_edge_vecs(64, device)
        out_dim = (lmax + 1) ** 2
        eye = torch.eye(out_dim, device=device).unsqueeze(0)

        for label, w in [
            ("torch", _pytorch_wigner(edge_vecs, jd_list, lmax)),
            ("triton", edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=lmax)),
        ]:
            product = torch.bmm(w, w.transpose(1, 2))
            err = (product - eye).abs().max().item()
            assert err < 1e-4, f"[{label}] wigner @ wigner^T != I, err={err:.6f}"

    def test_l0_exact_match_between_paths(self, device, jd_list):
        """L=0 is gamma-independent, so both paths must match exactly."""
        edge_vecs = _make_edge_vecs(64, device)
        w_pt = _pytorch_wigner(edge_vecs, jd_list, lmax=4)
        w_tr = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=4)
        assert torch.allclose(w_pt[:, 0:1, 0:1], w_tr[:, 0:1, 0:1], atol=1e-5), \
            "L=0 block should match exactly between paths"


# ---------------------------------------------------------------------------
# 3. Seed determinism
# ---------------------------------------------------------------------------
class TestSeedDeterminism:

    @pytest.mark.parametrize("lmax", [0, 4, 8])
    def test_same_seed_same_output(self, lmax, device, jd_list):
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries, need {lmax+1}")
        edge_vecs = _make_edge_vecs(64, device)
        w1 = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=lmax, seed=42)
        w2 = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=lmax, seed=42)
        assert torch.allclose(w1, w2, atol=1e-6)

    @pytest.mark.parametrize("lmax", [4, 8])
    def test_different_seed_different_output(self, lmax, device, jd_list):
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries, need {lmax+1}")
        edge_vecs = _make_edge_vecs(64, device)
        w1 = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=lmax, seed=42)
        w2 = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=lmax, seed=99)
        assert not torch.allclose(w1, w2, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:

    @pytest.mark.parametrize("lmax", list(range(0, 9)))
    def test_shape(self, lmax, device, jd_list):
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries, need {lmax+1}")
        edge_vecs = _make_edge_vecs(50, device)
        wigner = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=lmax, seed=1)
        expected_dim = (lmax + 1) ** 2
        assert wigner.shape == (50, expected_dim, expected_dim)
        assert wigner.dtype == edge_vecs.dtype

    def test_empty_input(self, device, jd_list):
        edge_vecs = torch.empty(0, 3, device=device, dtype=torch.float32)
        wigner = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=4, seed=1)
        assert wigner.shape == (0, 25, 25)

    def test_single_edge(self, device, jd_list):
        wigner = edge_vec_to_wigner_fused(_make_edge_vecs(1, device), jd_list, lmax=4, seed=1)
        assert wigner.shape == (1, 25, 25)

    def test_lmax_zero(self, device, jd_list):
        wigner = edge_vec_to_wigner_fused(_make_edge_vecs(10, device), jd_list, lmax=0, seed=1)
        assert wigner.shape == (10, 1, 1)
        assert (wigner - 1.0).abs().max() < 1e-5

    def test_invalid_lmax(self, device, jd_list):
        edge_vecs = _make_edge_vecs(10, device)
        with pytest.raises(AssertionError):
            edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=9, seed=1)
        with pytest.raises(AssertionError):
            edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=-1, seed=1)

    def test_unit_vectors(self, device, jd_list):
        """Axis-aligned unit vectors should produce valid Wigner matrices."""
        edge_vecs = torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ], device=device, dtype=torch.float32)
        wigner = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=4, seed=42)
        product = torch.bmm(wigner, wigner.transpose(1, 2))
        eye = torch.eye(25, device=device).unsqueeze(0).expand_as(product)
        assert (product - eye).abs().max() < 1e-4
