from __future__ import annotations

import torch
from e3nn.o3 import Irreps, Linear

from maloq.helm.qhflow3 import MuonVisibleIrrepLinear
from maloq.train_utils.training_workflow import TrainingWorkflow


def _escn_irreps(channels: int, lmax: int) -> Irreps:
    return Irreps(
        "+".join(
            f"{channels}x{degree}{'e' if degree % 2 == 0 else 'o'}"
            for degree in range(lmax + 1)
        )
    )


def test_qhflow3_muon_projection_preserves_forward_and_gradients_exactly():
    irreps_in = _escn_irreps(8, 3)
    irreps_out = _escn_irreps(4, 3)

    torch.manual_seed(260726)
    reference = Linear(irreps_in, irreps_out, biases=False)
    torch.manual_seed(260726)
    muon_visible = MuonVisibleIrrepLinear(irreps_in, irreps_out)

    mapped_weight = muon_visible.weight.transpose(1, 2).reshape(-1)
    torch.testing.assert_close(mapped_weight, reference.weight, rtol=0.0, atol=0.0)

    features = torch.randn(3, irreps_in.dim)
    reference_features = features.clone().requires_grad_(True)
    muon_features = features.clone().requires_grad_(True)
    probe = torch.randn(3, irreps_out.dim)

    reference_output = reference(reference_features)
    muon_output = muon_visible(muon_features)
    torch.testing.assert_close(muon_output, reference_output, rtol=0.0, atol=0.0)

    (reference_output * probe).sum().backward()
    (muon_output * probe).sum().backward()
    torch.testing.assert_close(
        muon_features.grad,
        reference_features.grad,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        muon_visible.weight.grad.transpose(1, 2).reshape(-1),
        reference.weight.grad,
        rtol=0.0,
        atol=0.0,
    )


def test_qhflow3_muon_projection_is_shape_routed_to_muon():
    projection = MuonVisibleIrrepLinear(
        _escn_irreps(8, 3),
        _escn_irreps(4, 3),
    )

    routed = TrainingWorkflow._collect_muon_parameters(
        projection,
        torch.nn.Identity(),
    )

    assert tuple(projection.weight.shape) == (4, 4, 8)
    assert [id(parameter) for parameter in routed] == [id(projection.weight)]
