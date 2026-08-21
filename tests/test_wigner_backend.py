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
from maloq.helm.triton import (
    edge_vec_to_wigner_fused,
    edge_vec_to_wigner_packed,
    extract_euler_angles,
    packed_wigner_width,
    wigner_fused_dense_and_packed,
)

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


# ---------------------------------------------------------------------------
# Packed (compact block-diagonal) kernel
#
#    The packed kernel computes the same tiles as the dense one and stores them
#    straight into the layout FlashSO2 consumes. Because the algebra is shared
#    verbatim, "close" is not good enough here -- for one seed the two outputs
#    must be bit-identical, and that is what pins the two layouts together.
# ---------------------------------------------------------------------------
def _block_indices(lmax, device):
    """Dense flat indices of the per-degree diagonal blocks, row-major."""
    indices, start = [], 0
    coefficients = (lmax + 1) ** 2
    for l in range(lmax + 1):
        size = 2 * l + 1
        for row in range(start, start + size):
            indices.extend(range(row * coefficients + start,
                                 row * coefficients + start + size))
        start += size
    return torch.tensor(indices, dtype=torch.long, device=device)


class TestPackedLayout:

    @pytest.mark.parametrize("lmax", list(range(0, 9)))
    def test_matches_gathered_dense(self, lmax, device, jd_list):
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries, need {lmax+1}")

        seed = 4242
        edge_vecs = _make_edge_vecs(256, device, seed=lmax * 13 + 5)
        dense = edge_vec_to_wigner_fused(edge_vecs, jd_list, lmax=lmax, seed=seed)
        packed, packed_inv = edge_vec_to_wigner_packed(
            edge_vecs, jd_list, lmax=lmax, seed=seed
        )

        indices = _block_indices(lmax, device)
        assert torch.equal(packed, dense.flatten(1).index_select(1, indices))
        assert torch.equal(
            packed_inv,
            dense.transpose(1, 2).contiguous().flatten(1).index_select(1, indices),
        )

    @pytest.mark.parametrize("lmax", [2, 4, 8])
    def test_dense_holds_nothing_outside_the_blocks(self, lmax, device, jd_list):
        """The compact form is lossless only if the rest is exactly zero."""
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries, need {lmax+1}")

        dense = edge_vec_to_wigner_fused(
            _make_edge_vecs(64, device, seed=7), jd_list, lmax=lmax, seed=11
        ).flatten(1)
        dense[:, _block_indices(lmax, device)] = 0.0
        assert dense.abs().max().item() == 0.0

    @pytest.mark.parametrize("lmax", [0, 4, 7, 8])
    def test_width(self, lmax, device, jd_list):
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries, need {lmax+1}")
        packed, packed_inv = edge_vec_to_wigner_packed(
            _make_edge_vecs(8, device), jd_list, lmax=lmax, seed=1
        )
        expected = sum((2 * l + 1) ** 2 for l in range(lmax + 1))
        assert packed_wigner_width(lmax) == expected
        assert packed.shape == packed_inv.shape == (8, expected)

    def test_caller_buffers_are_written_in_place(self, device, jd_list):
        """Every element is written, so the buffers need no pre-zeroing."""
        lmax = 4
        edge_vecs = _make_edge_vecs(32, device, seed=3)
        width = packed_wigner_width(lmax)
        buf = torch.full((32, width), float("nan"), device=device)
        buf_inv = torch.full((32, width), float("nan"), device=device)

        packed, packed_inv = edge_vec_to_wigner_packed(
            edge_vecs, jd_list, lmax=lmax, seed=5, out=buf, out_inv=buf_inv
        )
        assert packed.data_ptr() == buf.data_ptr()
        assert packed_inv.data_ptr() == buf_inv.data_ptr()
        assert buf.isfinite().all() and buf_inv.isfinite().all()

    def test_inverse_is_the_per_degree_transpose(self, device, jd_list):
        """W_l^T W_l = I for every degree, read straight out of the packed rows."""
        lmax = 6
        if len(jd_list) <= lmax:
            pytest.skip("Jd.pt too short")
        packed, packed_inv = edge_vec_to_wigner_packed(
            _make_edge_vecs(16, device, seed=9), jd_list, lmax=lmax, seed=13
        )
        offset = 0
        for l in range(lmax + 1):
            size = 2 * l + 1
            block = packed[:, offset:offset + size * size].view(-1, size, size)
            block_inv = packed_inv[:, offset:offset + size * size].view(-1, size, size)
            assert torch.equal(block_inv, block.transpose(1, 2))
            identity = torch.bmm(block_inv, block)
            expected = torch.eye(size, device=device).expand_as(identity)
            assert (identity - expected).abs().max().item() < 1e-4
            offset += size * size

    @pytest.mark.parametrize("lmax", [4, 6, 8])
    def test_dense_and_packed_share_one_gauge(self, lmax, device, jd_list):
        """The flash path rotates with both layouts, so they must agree.

        The edge-degree embedding uses the dense wigner_inv while the blocks use
        the packed pair. Two independent gamma draws would put them in different
        frames -- silently, since each is internally consistent -- so they are
        produced by one call from one draw. This is that guarantee.
        """
        if len(jd_list) <= lmax:
            pytest.skip(f"Jd.pt has only {len(jd_list)} entries, need {lmax+1}")

        edge_vecs = _make_edge_vecs(128, device, seed=lmax * 3)
        dense, packed, packed_inv = wigner_fused_dense_and_packed(
            edge_vecs, jd_list, lmax=lmax
        )
        indices = _block_indices(lmax, device)
        assert torch.equal(packed, dense.flatten(1).index_select(1, indices))
        assert torch.equal(
            packed_inv,
            dense.transpose(1, 2).contiguous().flatten(1).index_select(1, indices),
        )
