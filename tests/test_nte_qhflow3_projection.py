from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from e3nn import o3
from e3nn.o3 import Irreps
from pydantic import ValidationError

from maloq.core.config import MaloqConfig
from maloq.helm.esen_osh import (
    QHFlow3IrrepLinear,
    _qhflow3_irrep_projection_with_legacy_rng,
    eSEN_Backbone,
)
from maloq.helm.nn.so3_layers import SO3_Linear
from maloq.helm.qhflow3_clean import MuonVisibleIrrepLinear
from maloq.train_utils.training_workflow import TrainingWorkflow


def _small_backbone(
    *,
    projection_mode: str | None = None,
    output_channels: int = 2,
) -> eSEN_Backbone:
    kwargs = {}
    if projection_mode is not None:
        kwargs["nte_output_projection_mode"] = projection_mode
    return eSEN_Backbone(
        Irreps("1x0e"),
        sphere_channels=4,
        hidden_channels=4,
        lmax=2,
        mmax=2,
        cutoff=8.0,
        edge_channels=4,
        num_layers=1,
        num_edge_layers=1,
        num_distance_basis=4,
        output_sphere_channels=output_channels,
        **kwargs,
    )


def _native_to_flat(features: torch.Tensor, lmax: int) -> torch.Tensor:
    return torch.cat(
        [
            features[:, degree**2 : (degree + 1) ** 2, :]
            .transpose(1, 2)
            .reshape(features.shape[0], -1)
            for degree in range(lmax + 1)
        ],
        dim=1,
    )


def _flat_to_native(
    features: torch.Tensor,
    *,
    lmax: int,
    channels: int,
) -> torch.Tensor:
    blocks = []
    offset = 0
    for degree in range(lmax + 1):
        multiplicity = 2 * degree + 1
        width = channels * multiplicity
        blocks.append(
            features[:, offset : offset + width]
            .reshape(features.shape[0], channels, multiplicity)
            .transpose(1, 2)
        )
        offset += width
    return torch.cat(blocks, dim=1)


def _alternating_irreps(channels: int, lmax: int) -> Irreps:
    return Irreps(
        [
            (channels, (degree, 1 if degree % 2 == 0 else -1))
            for degree in range(lmax + 1)
        ]
    )


def test_default_projection_path_preserves_state_and_rng_bitwise() -> None:
    torch.manual_seed(260726)
    implicit_default = _small_backbone()
    implicit_rng = torch.get_rng_state().clone()

    torch.manual_seed(260726)
    explicit_default = _small_backbone(projection_mode="so3_linear")
    explicit_rng = torch.get_rng_state().clone()

    assert torch.equal(explicit_rng, implicit_rng)
    assert set(explicit_default.state_dict()) == set(implicit_default.state_dict())
    for name, expected in implicit_default.state_dict().items():
        torch.testing.assert_close(
            explicit_default.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_qhflow3_projection_keeps_legacy_full_backbone_post_init_rng() -> None:
    seed = 260727
    torch.manual_seed(seed)
    reference = _small_backbone(projection_mode="so3_linear")
    reference_rng = torch.get_rng_state().clone()

    torch.manual_seed(seed)
    qhflow3 = _small_backbone(projection_mode="qhflow3_irrep_linear")
    qhflow3_rng = torch.get_rng_state().clone()

    assert torch.equal(qhflow3_rng, reference_rng)

    # The two QHFlow3 tensors are normal-initialized rather than copied from
    # the legacy uniform projection, while downstream RNG remains controlled.
    assert tuple(qhflow3.node_output_projection.weight.shape) == (3, 2, 4)
    assert tuple(qhflow3.edge_output_projection.weight.shape) == (3, 2, 4)
    assert not torch.equal(
        qhflow3.node_output_projection.weight,
        reference.node_output_projection.weight,
    )


def test_rng_aligned_helper_preserves_direct_qhflow3_start_weight_exactly() -> None:
    seed = 260729
    irreps_in = _alternating_irreps(4, 2)
    irreps_out = _alternating_irreps(2, 2)

    torch.manual_seed(seed)
    direct = MuonVisibleIrrepLinear(irreps_in, irreps_out)
    expected_weight = direct.weight.detach().clone()

    torch.manual_seed(seed)
    adapter = _qhflow3_irrep_projection_with_legacy_rng(4, 2, 2)
    adapter_rng = torch.get_rng_state().clone()

    torch.manual_seed(seed)
    SO3_Linear(4, 2, lmax=2, bias=False)
    legacy_rng = torch.get_rng_state().clone()

    torch.testing.assert_close(
        adapter.weight,
        expected_weight,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(adapter_rng, legacy_rng)


def test_native_ordering_sentinel_matches_degreewise_qhflow3_map() -> None:
    projection = QHFlow3IrrepLinear(2, 3, lmax=2).double()
    features = (
        torch.arange(2 * 9 * 2, dtype=torch.float64).reshape(2, 9, 2)
        + 0.125
    )
    with torch.no_grad():
        projection.weight.copy_(
            torch.arange(3 * 3 * 2, dtype=torch.float64).reshape(3, 3, 2)
            + 1.0
        )

    expected = torch.empty(2, 9, 3, dtype=torch.float64)
    for degree in range(3):
        degree_slice = slice(degree**2, (degree + 1) ** 2)
        expected[:, degree_slice, :] = torch.einsum(
            "nmi,oi->nmo",
            features[:, degree_slice, :],
            projection.weight[degree] / math.sqrt(2.0),
        )

    torch.testing.assert_close(
        projection(features),
        expected,
        rtol=5.0e-16,
        atol=5.0e-13,
    )
    assert projection.linear.linear.irreps_in == _alternating_irreps(2, 2)
    assert projection.linear.linear.irreps_out == _alternating_irreps(3, 2)


def test_adapter_matches_direct_wrapper_forward_and_all_gradients_exactly() -> None:
    lmax = 3
    in_channels = 5
    out_channels = 3
    torch.manual_seed(260728)
    adapter = QHFlow3IrrepLinear(in_channels, out_channels, lmax).double()
    direct = MuonVisibleIrrepLinear(
        _alternating_irreps(in_channels, lmax),
        _alternating_irreps(out_channels, lmax),
    ).double()
    with torch.no_grad():
        direct.weight.copy_(adapter.weight)

    features = torch.randn(
        2,
        (lmax + 1) ** 2,
        in_channels,
        dtype=torch.float64,
    )
    adapter_features = features.clone().requires_grad_(True)
    direct_features = _native_to_flat(features, lmax).detach().requires_grad_(True)
    probe = torch.randn(
        2,
        (lmax + 1) ** 2,
        out_channels,
        dtype=torch.float64,
    )

    adapter_output = adapter(adapter_features)
    direct_output = _flat_to_native(
        direct(direct_features),
        lmax=lmax,
        channels=out_channels,
    )
    torch.testing.assert_close(adapter_output, direct_output, rtol=0.0, atol=0.0)

    (adapter_output * probe).sum().backward()
    (direct_output * probe).sum().backward()
    torch.testing.assert_close(
        adapter_features.grad,
        _flat_to_native(
            direct_features.grad,
            lmax=lmax,
            channels=in_channels,
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        adapter.weight.grad,
        direct.weight.grad,
        rtol=0.0,
        atol=0.0,
    )


def test_projection_is_so3_covariant_for_every_degree() -> None:
    lmax = 4
    projection = QHFlow3IrrepLinear(4, 3, lmax).double()
    features = torch.randn(3, (lmax + 1) ** 2, 4, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)

    def rotate(values: torch.Tensor) -> torch.Tensor:
        rotated = values.clone()
        for degree in range(lmax + 1):
            degree_slice = slice(degree**2, (degree + 1) ** 2)
            representation = o3.Irrep(
                degree,
                1 if degree % 2 == 0 else -1,
            ).D_from_matrix(rotation)
            rotated[:, degree_slice, :] = torch.einsum(
                "ij,njc->nic",
                representation,
                values[:, degree_slice, :],
            )
        return rotated

    torch.testing.assert_close(
        projection(rotate(features)),
        rotate(projection(features)),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_backbone_uses_projection_for_node_and_edge_with_native_shapes() -> None:
    backbone = _small_backbone(projection_mode="qhflow3_irrep_linear")
    assert isinstance(backbone.node_output_projection, QHFlow3IrrepLinear)
    assert isinstance(backbone.edge_output_projection, QHFlow3IrrepLinear)

    node_features = torch.randn(3, 9, 4)
    edge_features = torch.randn(7, 9, 4)
    assert backbone.node_output_projection(node_features).shape == (3, 9, 2)
    assert backbone.edge_output_projection(edge_features).shape == (7, 9, 2)


@pytest.mark.parametrize(
    "projection_mode",
    ["so3_linear", "qhflow3_irrep_linear"],
)
def test_each_projection_mode_has_strict_state_dict_round_trip(
    projection_mode: str,
) -> None:
    torch.manual_seed(260730)
    source = _small_backbone(projection_mode=projection_mode)
    torch.manual_seed(91)
    target = _small_backbone(projection_mode=projection_mode)
    result = target.load_state_dict(source.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    features = torch.randn(3, 9, 4)
    torch.testing.assert_close(
        target.node_output_projection(features),
        source.node_output_projection(features),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        target.edge_output_projection(features),
        source.edge_output_projection(features),
        rtol=0.0,
        atol=0.0,
    )


def test_projection_weights_route_once_to_shape_muon_and_can_be_excluded() -> None:
    backbone = _small_backbone(projection_mode="qhflow3_irrep_linear")
    weights = [
        backbone.node_output_projection.weight,
        backbone.edge_output_projection.weight,
    ]
    assert [tuple(weight.shape) for weight in weights] == [(3, 2, 4), (3, 2, 4)]

    routed = TrainingWorkflow._collect_muon_parameters(
        backbone,
        torch.nn.Identity(),
    )
    routed_ids = [id(parameter) for parameter in routed]
    for weight in weights:
        assert routed_ids.count(id(weight)) == 1

    adamw_projection_parameters = (
        TrainingWorkflow._collect_nte_output_projection_parameters(backbone)
    )
    assert [id(parameter) for parameter in adamw_projection_parameters] == [
        id(weight) for weight in weights
    ]
    excluded_ids = {
        id(parameter) for parameter in adamw_projection_parameters
    }
    shape_muon_after_adamw_policy = [
        parameter for parameter in routed if id(parameter) not in excluded_ids
    ]
    assert all(
        id(weight)
        not in {id(parameter) for parameter in shape_muon_after_adamw_policy}
        for weight in weights
    )
    assert sum(
        id(parameter) in {id(weight) for weight in weights}
        for parameter in backbone.parameters()
    ) == 2


def test_nabladft_projection_parameter_count_and_unique_optimizer_ids() -> None:
    class ProjectionPair(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_output_projection = QHFlow3IrrepLinear(128, 64, lmax=4)
            self.edge_output_projection = QHFlow3IrrepLinear(128, 64, lmax=4)

    pair = ProjectionPair()
    projection_parameters = (
        TrainingWorkflow._collect_nte_output_projection_parameters(pair)
    )
    assert [tuple(parameter.shape) for parameter in projection_parameters] == [
        (5, 64, 128),
        (5, 64, 128),
    ]
    assert sum(parameter.numel() for parameter in projection_parameters) == 81_920

    routed = TrainingWorkflow._collect_muon_parameters(
        pair,
        torch.nn.Identity(),
    )
    routed_ids = [id(parameter) for parameter in routed]
    assert len(routed_ids) == len(set(routed_ids)) == 2
    assert routed_ids == [id(parameter) for parameter in projection_parameters]


def test_identity_width_and_invalid_mode_contracts() -> None:
    identity = _small_backbone(
        projection_mode="qhflow3_irrep_linear",
        output_channels=4,
    )
    assert isinstance(identity.node_output_projection, torch.nn.Identity)
    assert isinstance(identity.edge_output_projection, torch.nn.Identity)

    with pytest.raises(ValueError, match="nte_output_projection_mode"):
        _small_backbone(projection_mode="not-a-mode")
    with pytest.raises(ValidationError, match="nte_output_projection_mode"):
        MaloqConfig(model={"nte_output_projection_mode": "not-a-mode"})


def test_training_workflow_validates_projection_mode_and_backbone_scope() -> None:
    workflow = object.__new__(TrainingWorkflow)
    workflow.config = MaloqConfig().to_workflow_config()
    workflow.config["nte_output_projection_mode"] = "not-a-mode"
    with pytest.raises(ValueError, match="nte_output_projection_mode"):
        workflow.check_input_config()

    workflow.config = MaloqConfig().to_workflow_config()
    workflow.config.update(
        backbone_type="qhflow3_clean",
        nte_output_projection_mode="qhflow3_irrep_linear",
    )
    with pytest.raises(ValueError, match="requires backbone_type='esen'"):
        workflow.check_input_config()


def test_config_round_trip_and_nabladft_tracking_identity() -> None:
    default = MaloqConfig().to_workflow_config()
    assert default["nte_output_projection_mode"] == "so3_linear"
    configured = MaloqConfig(
        model={"nte_output_projection_mode": "qhflow3_irrep_linear"}
    ).to_workflow_config()
    assert configured["nte_output_projection_mode"] == "qhflow3_irrep_linear"

    runner_path = (
        Path(__file__).resolve().parents[1]
        / "_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_nabladft_qh9_density_projection_test",
        runner_path,
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    identity = runner.nabladft_tracking_identity(
        "maloq-nte",
        {
            "l_embedding_dim": 128,
            "output_l_embedding_dim": 64,
            "num_mp_layers": 2,
            "num_edge_layers": 2,
            "head_type": "maloq_muon",
            "optimizer_type": "muon",
            "scale_and_shift": False,
            "experiment_version": 1,
            "seed": 44,
            "nte_input_conditioning": "qhflow3_exact",
            "nte_output_projection_mode": "qhflow3_irrep_linear",
        },
        smoke=False,
    )
    assert identity["experiment_id"].endswith("-qcond-qhfproj-v1")
    assert identity["display_name"].endswith(
        "QHFcond | QHFProj | V1"
    )
    assert "output-projection:qhflow3-irrep-linear" in identity["tags"]
    assert (
        "output-projection-rng:legacy-so3-linear-aligned"
        in identity["tags"]
    )
