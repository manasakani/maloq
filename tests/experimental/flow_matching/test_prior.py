from __future__ import annotations

from itertools import product

import pytest
import torch
from e3nn import o3
from pydantic import ValidationError
from torch import nn

import maloq.experimental.flow_matching.trainer as flow_trainer
from maloq.experimental.flow_matching import (
    CoupledAOCodec,
    CoupledIrrepGaussianPrior,
    EndpointCorruptingLoader,
    EndpointFlowMatcher,
    EndpointFlowTrainer,
    FlowMatchingConfig,
    TensorExpansionPrior,
    build_coupled_prior,
)
from maloq.fock_utils.utils_tensor_decomp import e3TensorDecomp


def _basis(shells: list[int]) -> e3TensorDecomp:
    return e3TensorDecomp(
        net_irreps_out=None,
        out_js_list=list(product(shells, repeat=2)),
        default_dtype_torch=torch.float64,
    )


class _ScalarFlowBatch:
    def __init__(self) -> None:
        self.node_y = torch.arange(1, 17, dtype=torch.float64).reshape(1, 4, 4)
        self.y = torch.arange(21, 37, dtype=torch.float64).reshape(1, 4, 4)
        self.node_padding_mask = torch.ones((1, 4, 4), dtype=torch.bool)
        self.edge_padding_mask = torch.ones((1, 4, 4), dtype=torch.bool)
        self.batch = torch.tensor([0, 0, 1, 1])
        self.ptr = torch.tensor([0, 2, 4])
        self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])

    def to(self, device: torch.device | str):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


def test_default_gaussian_prior_keeps_the_previous_seeded_draw() -> None:
    reference = torch.zeros(3, 7, dtype=torch.float64)
    mask = torch.tensor([True, False, True])
    expected_generator = torch.Generator().manual_seed(31)
    expected = 0.1 * torch.randn(
        reference.shape,
        dtype=reference.dtype,
        generator=expected_generator,
    )
    expected[1] = 0.0
    global_rng_before = torch.random.get_rng_state().clone()

    prior = CoupledIrrepGaussianPrior(0.1)
    observed = prior.sample(
        reference,
        mask=mask,
        generator=torch.Generator().manual_seed(31),
    )

    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before)


def test_tensor_expansion_matches_qhflow2_unit_path_sum_exactly() -> None:
    shells = [0, 0, 0, 1, 1, 2]
    basis = _basis(shells)
    prior = TensorExpansionPrior(basis, sigma=0.1)
    reference = torch.zeros(2, basis.required_irreps_out.dim, dtype=torch.float64)
    generator = torch.Generator().manual_seed(17)

    source = prior.sample(reference, mask=None, generator=generator)

    expected_generator = torch.Generator().manual_seed(17)
    original_input = torch.randn(
        (2, 14),
        dtype=torch.float64,
        generator=expected_generator,
    )
    shared = {}
    feature_start = 0
    for degree in sorted(set(shells)):
        multiplicity = shells.count(degree)
        irrep_dim = 2 * degree + 1
        feature_stop = feature_start + multiplicity * irrep_dim
        shared[degree] = (
            0.1
            * original_input[:, feature_start:feature_stop]
            .reshape(2, multiplicity, irrep_dim)
            .sum(dim=1)
        )
        feature_start = feature_stop
    for mul_ir, path_slice in zip(
        basis.required_irreps_out,
        basis.required_irreps_out.slices(),
    ):
        degree = int(mul_ir.ir.l)
        expected = shared.get(degree)
        if expected is None:
            torch.testing.assert_close(
                source[:, path_slice], torch.zeros_like(source[:, path_slice])
            )
        else:
            torch.testing.assert_close(source[:, path_slice], expected)

    # The original input contains only l=0,1,2. The l=3,4 d-d output sectors
    # are consequently zero, giving covariance rank 1+3+5=9.
    assert prior.input_feature_dim == 14
    assert prior.latent_rank == 9


def test_nabla_tensor_expansion_generalizes_shell_counts_but_stays_rank_nine() -> None:
    # Largest active def2-SVP Nabla layout: 5s + 4p + 3d = 32 AOs.
    shells = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    basis = _basis(shells)
    prior = TensorExpansionPrior(basis, sigma=0.1)

    assert CoupledAOCodec(basis).ao_dim == 32
    assert basis.required_irreps_out.dim == 32 * 32
    assert prior.input_feature_dim == 32
    assert prior.latent_rank == 9


def test_tensor_expansion_decoder_is_so3_equivariant() -> None:
    basis = _basis([0, 1])
    codec = CoupledAOCodec(basis)
    prior = TensorExpansionPrior(basis, sigma=0.1)
    source = prior.sample(
        torch.zeros(2, basis.required_irreps_out.dim, dtype=torch.float64),
        mask=None,
        generator=torch.Generator().manual_seed(5),
    )
    rotation = o3.angles_to_matrix(
        torch.tensor(0.41, dtype=torch.float64),
        torch.tensor(0.83, dtype=torch.float64),
        torch.tensor(-0.27, dtype=torch.float64),
    )
    coupled_action = basis.required_irreps_out.D_from_matrix(rotation)
    orbital_action = o3.Irreps("1x0e + 1x1e").D_from_matrix(rotation)

    decoded = codec.decode(source)
    decoded_rotated = codec.decode(source @ coupled_action.T)
    expected_rotated = orbital_action @ decoded @ orbital_action.T

    torch.testing.assert_close(
        decoded_rotated,
        expected_rotated,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_tensor_expansion_rejects_partial_irrep_masks() -> None:
    basis = _basis([1])
    prior = TensorExpansionPrior(basis, sigma=0.1)
    reference = torch.zeros(1, basis.required_irreps_out.dim, dtype=torch.float64)
    mask = torch.ones_like(reference, dtype=torch.bool)
    degree_one_slice = basis.required_irreps_out.slices()[1]
    mask[0, degree_one_slice.start] = False

    with pytest.raises(ValueError, match="complete irrep paths"):
        prior.sample(reference, mask=mask)


def test_prior_factory_is_strict_and_tensor_expansion_is_explicit() -> None:
    basis = _basis([0, 1])
    gaussian_config = FlowMatchingConfig()
    tensor_config = FlowMatchingConfig(prior_type="tensor_expansion")

    assert isinstance(build_coupled_prior(gaussian_config, basis), CoupledIrrepGaussianPrior)
    assert isinstance(build_coupled_prior(tensor_config, basis), TensorExpansionPrior)
    assert tensor_config.tensor_expansion_normalization == "qhflow2_unit_path_sum"
    with pytest.raises(ValidationError):
        FlowMatchingConfig(
            prior_type="tensor_expansion",
            tensor_expansion_normalization="variance_matched_coupled_component",
        )


def test_training_loader_uses_tensor_expansion_for_node_and_edge_sources() -> None:
    basis = _basis([0, 0])
    batch = next(
        iter(
            EndpointCorruptingLoader(
                [_ScalarFlowBatch()],
                matcher=EndpointFlowMatcher(
                    FlowMatchingConfig(prior_type="tensor_expansion")
                ),
                basis_transform=basis,
                device="cpu",
            )
        )
    )
    node_time = batch.t.index_select(0, batch.batch).unsqueeze(-1)
    edge_time = batch.t.index_select(0, batch.batch[batch.edge_index[0]]).unsqueeze(-1)
    node_source = (batch.node_flow_t - node_time * batch.node_y[0]) / (1.0 - node_time)
    edge_source = (batch.edge_flow_t - edge_time * batch.y[0]) / (1.0 - edge_time)

    # Four s-s shell paths reuse the same summed l=0 latent per entry.
    torch.testing.assert_close(node_source, node_source[:, :1].expand_as(node_source))
    torch.testing.assert_close(edge_source, edge_source[:, :1].expand_as(edge_source))


def test_validation_sampler_constructs_the_same_configured_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _IdentityBackbone(nn.Module):
        def forward(self, batch):
            return batch

    class _StateEndpointHead(nn.Module):
        def forward(self, _features, batch):
            return batch.node_flow_t, batch.edge_flow_t

    basis = _basis([0, 0])
    configured_prior = TensorExpansionPrior(basis, sigma=0.1)
    observed_configs: list[FlowMatchingConfig] = []

    def fake_factory(config, observed_basis):
        observed_configs.append(config)
        assert observed_basis is basis
        return configured_prior

    monkeypatch.setattr(flow_trainer, "build_coupled_prior", fake_factory)
    trainer = EndpointFlowTrainer(
        backbone=_IdentityBackbone(),
        head=_StateEndpointHead(),
        head_irreps="irreps",
        config=FlowMatchingConfig(prior_type="tensor_expansion"),
    )

    result = trainer.sample_batch(
        _ScalarFlowBatch(),
        basis_transform=basis,
        device="cpu",
        generator=torch.Generator().manual_seed(9),
    )

    assert observed_configs == [trainer.flow_config]
    assert result.node.shape == (4, 4)
    assert result.edge.shape == (4, 4)
