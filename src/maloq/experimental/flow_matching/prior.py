"""Coupled-coordinate priors for full-matrix endpoint flow."""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol

import torch
from torch import Tensor

from .conditioning import CoupledAOCodec, CoupledBasisTransform
from .objective import broadcast_mask


class CoupledPrior(Protocol):
    """Sample a target-shaped prior in coupled Hamiltonian coordinates."""

    def sample(
        self,
        reference: Tensor,
        *,
        mask: Tensor | None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


def _validate_reference(reference: Tensor, *, coupled_dim: int | None = None) -> None:
    if (
        not isinstance(reference, Tensor)
        or reference.ndim != 2
        or not reference.is_floating_point()
    ):
        raise ValueError(
            "Coupled prior reference must be a floating-point [entries, dim] tensor."
        )
    if coupled_dim is not None and reference.shape[-1] != coupled_dim:
        raise ValueError(
            f"Coupled prior width must be {coupled_dim}, "
            f"got {reference.shape[-1]}."
        )


class CoupledIrrepGaussianPrior:
    """Independent isotropic Gaussian coefficients for every expansion path."""

    def __init__(self, sigma: float):
        if sigma <= 0.0:
            raise ValueError("Prior sigma must be positive.")
        self.sigma = float(sigma)

    def sample(
        self,
        reference: Tensor,
        *,
        mask: Tensor | None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        _validate_reference(reference)
        valid = broadcast_mask(mask, reference, entry_dim=0)
        source = torch.randn(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )
        source = source * self.sigma
        return torch.where(valid, source, torch.zeros_like(source))


class TensorExpansionPrior:
    r"""QHFlow2 unit-path Tensor Expansion prior in coupled coordinates.

    The pinned QHFlow2 prior samples one feature tensor for every orbital
    degree, sums its shell-copy channels through all-one path weights, and
    reuses that sum for every compatible shell-pair path. MALOQ already stores
    the coefficients immediately before the same Wigner-3j expansion, so the
    construction can be represented exactly without materializing AO blocks.

    This intentionally preserves QHFlow2's unit-path *sum* normalization:
    for ``C_l`` copies of an orbital degree ``l``, each shared coefficient has
    standard deviation ``sigma * sqrt(C_l)``. Coupled output degrees that are
    absent from the orbital basis receive zero. For def2-SVP NablaDFT this is
    therefore a rank-9 (l=0,1,2) prior before Hamiltonian projection, just as
    in the original 14-AO QH9 construction, while the shell multiplicities are
    generalized from the active padded basis layout.
    """

    def __init__(self, basis_transform: CoupledBasisTransform, sigma: float):
        if sigma <= 0.0:
            raise ValueError("Prior sigma must be positive.")
        codec = CoupledAOCodec(basis_transform)
        irreps = getattr(basis_transform, "required_irreps_out", None)
        slices = getattr(irreps, "slices", None)
        coupled_dim = getattr(irreps, "dim", None)
        if irreps is None or not callable(slices) or coupled_dim is None:
            raise TypeError(
                "Tensor Expansion prior requires basis_transform.required_irreps_out."
            )

        self.sigma = float(sigma)
        self.irreps = irreps
        self.coupled_dim = int(coupled_dim)
        groups = tuple(
            (
                (int(mul_ir.ir.l), int(mul_ir.ir.p)),
                int(mul_ir.mul),
                int(mul_ir.ir.dim),
                irrep_slice,
            )
            for mul_ir, irrep_slice in zip(irreps, irreps.slices())
        )
        if not groups or groups[-1][3].stop != self.coupled_dim:
            raise ValueError("Coupled output irreps do not cover their declared width.")

        expected_path_keys = tuple(
            (coupled_degree, 1)
            for left_degree, right_degree in basis_transform.out_js_list
            for coupled_degree in range(
                abs(int(left_degree) % 10 - int(right_degree) % 10),
                int(left_degree) % 10 + int(right_degree) % 10 + 1,
            )
        )
        observed_path_keys = tuple(
            key
            for key, multiplicity, _irrep_dim, _irrep_slice in groups
            for _ in range(multiplicity)
        )
        if observed_path_keys != expected_path_keys:
            raise ValueError(
                "Tensor Expansion prior requires the canonical unsorted "
                "shell-pair coupled-irrep layout."
            )

        shell_multiplicities = Counter(int(shell) for shell in codec.shells)
        self._latent_specs = tuple(
            ((degree, 1), multiplicity, 2 * degree + 1)
            for degree, multiplicity in sorted(shell_multiplicities.items())
        )
        self.input_feature_dim = sum(
            multiplicity * irrep_dim
            for _key, multiplicity, irrep_dim in self._latent_specs
        )
        self.latent_rank = sum(
            irrep_dim for _key, _multiplicity, irrep_dim in self._latent_specs
        )
        self._groups = groups

    def sample(
        self,
        reference: Tensor,
        *,
        mask: Tensor | None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        _validate_reference(reference, coupled_dim=self.coupled_dim)
        valid = broadcast_mask(mask, reference, entry_dim=0)
        for _key, _multiplicity, _irrep_dim, irrep_slice in self._groups:
            path_valid = valid[:, irrep_slice]
            if not torch.equal(path_valid, path_valid[:, :1].expand_as(path_valid)):
                raise ValueError(
                    "Tensor Expansion masks must keep or remove complete irrep paths."
                )
        entry_count = reference.shape[0]
        # QHFlow2 calls Irreps.randn(entries, -1), whose component-normalized
        # implementation is one flat torch.randn draw. Preserve that ordering
        # as well as the distribution so explicit generators are reproducible.
        input_features = torch.randn(
            (entry_count, self.input_feature_dim),
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )
        latent_by_degree = {}
        feature_start = 0
        for key, multiplicity, irrep_dim in self._latent_specs:
            feature_stop = feature_start + multiplicity * irrep_dim
            latent_by_degree[key] = (
                input_features[:, feature_start:feature_stop]
                .reshape(entry_count, multiplicity, irrep_dim)
                .sum(dim=1)
                * self.sigma
            )
            feature_start = feature_stop

        source = torch.zeros_like(reference)
        for key, multiplicity, irrep_dim, irrep_slice in self._groups:
            shared = latent_by_degree.get(key)
            if shared is None:
                continue
            source[:, irrep_slice] = (
                shared[:, None, :]
                .expand(entry_count, multiplicity, irrep_dim)
                .reshape(entry_count, multiplicity * irrep_dim)
            )
        return torch.where(valid, source, torch.zeros_like(source))


def build_coupled_prior(
    config: Any,
    basis_transform: CoupledBasisTransform,
) -> CoupledPrior:
    """Construct the configured feature-owned prior."""
    prior_type = getattr(config, "prior_type", None)
    sigma = float(getattr(config, "sigma"))
    if prior_type == "coupled_irrep_gaussian":
        return CoupledIrrepGaussianPrior(sigma)
    if prior_type == "tensor_expansion":
        normalization = getattr(config, "tensor_expansion_normalization", None)
        if normalization != "qhflow2_unit_path_sum":
            raise ValueError(
                "Tensor Expansion prior requires qhflow2_unit_path_sum "
                f"normalization, got {normalization!r}."
            )
        return TensorExpansionPrior(basis_transform, sigma)
    raise ValueError(f"Unsupported coupled prior type: {prior_type!r}.")
