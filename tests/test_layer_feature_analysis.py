from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from e3nn.o3 import Irreps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "_auto_script/layer_feature_analysis/analyze_nabladft_qhf_vs_nte.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sc26_layer_feature_analysis",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


class _CapturedBackbone:
    args: tuple[object, ...]
    kwargs: dict[str, object]

    def __init__(self, *args: object, **kwargs: object) -> None:
        type(self).args = args
        type(self).kwargs = kwargs

    def to(self, _device: object) -> "_CapturedBackbone":
        return self


def _fixed_payload(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    *,
    epoch: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backbone_state_dict": backbone.state_dict(),
        "head_state_dict": head.state_dict(),
        "completed_epoch": epoch,
        "config_signature_digest": "test",
        "history": {},
        "optimizer_state_dict": {},
        "rng_states": [],
        "scheduler_state_dict": {},
        "world_size": 2,
    }


def _fill(module: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _assert_same_state(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
) -> None:
    for key, expected_value in expected.state_dict().items():
        torch.testing.assert_close(
            actual.state_dict()[key],
            expected_value,
            rtol=0.0,
            atol=0.0,
        )


def test_defaults_are_optimizer_fair_qhflow3_and_qhfcond_nte() -> None:
    assert ANALYSIS.DEFAULT_QHF_CONFIG.name == (
        "qhflow3_ov0_ntegrid_projection_muon_nabladft.yaml"
    )
    assert ANALYSIS.DEFAULT_NTE_CONFIG.name == (
        "nte64e2_qhflow3_conditioning_nabladft.yaml"
    )
    assert ANALYSIS.DEFAULT_QHF_RUN == (
        PROJECT_ROOT / "outputs/nabla-qhf3-projmuon-ov0-ntegrid-v3/run"
    )
    assert ANALYSIS.DEFAULT_NTE_RUN == (
        PROJECT_ROOT / "outputs/nabla-nte64e2-muon-ss0-qcond-v1/run"
    )
    assert "ProjMuon" in ANALYSIS.MODEL_LABELS["qhflow3"]
    assert "QHFcond" in ANALYSIS.MODEL_LABELS["nte"]


def test_qhflow3_constructor_propagates_projection_parameterization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maloq.helm import qhflow3_clean

    config = ANALYSIS.load_config(ANALYSIS.DEFAULT_QHF_CONFIG)
    monkeypatch.setattr(
        qhflow3_clean,
        "QHFlow3MaloqBackbone",
        _CapturedBackbone,
    )
    result = ANALYSIS.build_backbone(
        "qhflow3",
        config,
        SimpleNamespace(lmax=4),
        torch.device("cpu"),
    )

    assert result is not None
    assert _CapturedBackbone.kwargs["muonize_output_projection"] is True
    assert _CapturedBackbone.kwargs["grid_resolution"] == (
        config["qhflow3_grid_resolution"]
    )
    assert _CapturedBackbone.kwargs["grid_ffn_chunk_size"] == (
        config["qhflow3_grid_ffn_chunk_size"]
    )
    assert _CapturedBackbone.kwargs["use_block_S"] is False


@pytest.mark.parametrize(
    (
        "config_name",
        "expected_projection",
        "expected_norm",
        "expected_direct",
        "expected_initial_edge",
    ),
    [
        (
            "nte64e2_qcond_qhflow3_irrep_projection_nabladft.yaml",
            "qhflow3_irrep_linear",
            "post_edgewise",
            [],
            "edge_degree",
        ),
        (
            "nte64e2_qcond_edgepre_edge1direct_nabladft.yaml",
            "so3_linear",
            "pre_node",
            [1],
            "edge_degree",
        ),
        (
            "nte64e2_qcond_edgepre_qhflow3_irrep_projection_nabladft.yaml",
            "qhflow3_irrep_linear",
            "pre_node",
            [],
            "edge_degree",
        ),
        (
            "nte64e2_qcond_edgepre_edge1_qhflow3_irrep_projection_nabladft.yaml",
            "qhflow3_irrep_linear",
            "pre_node",
            [1],
            "edge_degree",
        ),
        (
            "nte64e2_qcond_edgepre_initial_edge_zero_nabladft.yaml",
            "so3_linear",
            "pre_node",
            [],
            "zero",
        ),
    ],
)
def test_nte_constructor_matches_current_structural_variant_configs(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    expected_projection: str,
    expected_norm: str,
    expected_direct: list[int],
    expected_initial_edge: str,
) -> None:
    from maloq.helm import esen_osh

    config_path = (
        PROJECT_ROOT / "_my_script/experiment/2026-07-26" / config_name
    )
    config = ANALYSIS.load_config(config_path)
    monkeypatch.setattr(esen_osh, "eSEN_Backbone", _CapturedBackbone)
    result = ANALYSIS.build_backbone(
        "nte",
        config,
        SimpleNamespace(lmax=4),
        torch.device("cpu"),
    )

    assert result is not None
    kwargs = _CapturedBackbone.kwargs
    propagated = (
        "nte_output_projection_mode",
        "repeat_system_embedding_each_node_block",
        "node_stack_mode",
        "edge_stack_mode",
        "qhflow3_layer_gaussian_width",
        "qhflow3_layer_grid_ffn_chunk_size",
        "qhflow3_exact_pair_rng_aligned",
        "edge_atom_norm_type",
        "edge_post_residual_norm_type",
        "direct_edgewise_layers",
        "edge_atomwise_output_mode",
        "edge_norm1_position",
        "initial_edge_state_mode",
    )
    for key in propagated:
        assert kwargs[key] == config[key]
    assert kwargs["nte_output_projection_mode"] == expected_projection
    assert kwargs["edge_norm1_position"] == expected_norm
    assert list(kwargs["direct_edgewise_layers"]) == expected_direct
    assert kwargs["initial_edge_state_mode"] == expected_initial_edge


def test_default_nte_constructor_supplies_legacy_message_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maloq.helm import esen_osh

    config = ANALYSIS.load_config(ANALYSIS.DEFAULT_NTE_CONFIG)
    config.pop("message_type", None)
    monkeypatch.setattr(esen_osh, "eSEN_Backbone", _CapturedBackbone)

    ANALYSIS.build_backbone(
        "nte",
        config,
        SimpleNamespace(lmax=4),
        torch.device("cpu"),
    )

    assert _CapturedBackbone.kwargs["message_type"] == "source-target"
    assert _CapturedBackbone.kwargs["initial_edge_state_mode"] == "edge_degree"


def test_fixed_checkpoint_uses_valid_previous_generation(
    tmp_path: Path,
) -> None:
    expected_backbone = torch.nn.Linear(3, 2)
    expected_head = torch.nn.Linear(2, 1)
    _fill(expected_backbone, 0.25)
    _fill(expected_head, -0.75)
    torch.save(
        _fixed_payload(expected_backbone, expected_head, epoch=19),
        tmp_path / "training_state.prev.pt",
    )
    (tmp_path / "training_state.pt").write_bytes(b"truncated")

    target_backbone = torch.nn.Linear(3, 2)
    target_head = torch.nn.Linear(2, 1)
    validation = ANALYSIS.validate_checkpoint_bundle(tmp_path)
    backbone_meta, head_meta = ANALYSIS.load_run_checkpoints(
        target_backbone,
        target_head,
        tmp_path,
    )

    assert validation["format"] == "fixed"
    assert Path(validation["path"]).name == "training_state.prev.pt"
    assert backbone_meta["format"] == "fixed"
    assert head_meta["epoch"] == 19
    _assert_same_state(target_backbone, expected_backbone)
    _assert_same_state(target_head, expected_head)


def test_invalid_fixed_checkpoint_falls_back_to_legacy_pair(
    tmp_path: Path,
) -> None:
    expected_backbone = torch.nn.Linear(3, 2)
    expected_head = torch.nn.Linear(2, 1)
    _fill(expected_backbone, 1.5)
    _fill(expected_head, -2.5)
    (tmp_path / "training_state.pt").write_bytes(b"invalid")
    torch.save(
        expected_backbone.state_dict(),
        tmp_path / "backbone_state_dic.pt",
    )
    torch.save(
        expected_head.state_dict(),
        tmp_path / "head_state_dic.pt",
    )

    target_backbone = torch.nn.Linear(3, 2)
    target_head = torch.nn.Linear(2, 1)
    validation = ANALYSIS.validate_checkpoint_bundle(tmp_path)
    backbone_meta, head_meta = ANALYSIS.load_run_checkpoints(
        target_backbone,
        target_head,
        tmp_path,
    )

    assert validation["format"] == "legacy"
    assert "fixed_checkpoint_error" in validation
    assert backbone_meta["format"] == "legacy"
    assert "fixed_checkpoint_error" in head_meta
    _assert_same_state(target_backbone, expected_backbone)
    _assert_same_state(target_head, expected_head)


def test_validate_checkpoint_bundle_rejects_corrupt_legacy_pair(
    tmp_path: Path,
) -> None:
    (tmp_path / "backbone_state_dic.pt").write_bytes(b"truncated")
    torch.save(
        torch.nn.Linear(2, 1).state_dict(),
        tmp_path / "head_state_dic.pt",
    )

    with pytest.raises(
        RuntimeError,
        match="Unable to load legacy checkpoint",
    ):
        ANALYSIS.validate_checkpoint_bundle(tmp_path)


def _alternating_irreps(channels: int, lmax: int) -> Irreps:
    return Irreps(
        [
            (channels, (degree, 1 if degree % 2 == 0 else -1))
            for degree in range(lmax + 1)
        ]
    )


@pytest.mark.parametrize("adapter_kind", ["nte", "qhflow3"])
def test_effective_projection_matrices_include_e3nn_path_weights(
    adapter_kind: str,
) -> None:
    from maloq.helm.esen_osh import QHFlow3IrrepLinear
    from maloq.helm.qhflow3_clean import MuonVisibleIrrepLinear

    in_channels = 4
    out_channels = 2
    lmax = 2
    if adapter_kind == "nte":
        projection = QHFlow3IrrepLinear(
            in_channels,
            out_channels,
            lmax,
        ).double()
    else:
        projection = MuonVisibleIrrepLinear(
            _alternating_irreps(in_channels, lmax),
            _alternating_irreps(out_channels, lmax),
        ).double()
    with torch.no_grad():
        projection.weight.copy_(
            torch.arange(
                (lmax + 1) * out_channels * in_channels,
                dtype=torch.float64,
            ).reshape(lmax + 1, out_channels, in_channels)
            + 1.0
        )

    matrices = ANALYSIS.effective_projection_matrices(projection, lmax)

    assert set(matrices) == {0, 1, 2}
    for degree in range(lmax + 1):
        torch.testing.assert_close(
            matrices[degree],
            projection.weight[degree] / math.sqrt(in_channels),
            rtol=0.0,
            atol=0.0,
        )
