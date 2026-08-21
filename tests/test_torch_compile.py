# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

"""
Tests for torch.compile support (the `compile` config key).

maloq compiles a bound method rather than the module -- see maloq.helm.common.compile
for why. These tests pin the properties that design is meant to guarantee: the
model traces to a single graph with no breaks, the compiled model produces the
same numbers and gradients as eager, and the checkpoint is unaffected by
whether compile was on.

The backbone is built directly from a synthetic graph rather than through
TrainingWorkflow, so these run without a dataset.
"""

import random
import sys

import pytest
import torch

from e3nn.o3 import Irreps

from maloq.helm.common.compile import neutralise_nvtx_for_dynamo
from maloq.helm.esen_osh import eSEN_Backbone


LMAX = 2
CHANNELS = 16


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _backbone(device, wigner_backend="torch", num_layers=1):
    torch.manual_seed(0)
    return eSEN_Backbone(
        Irreps("1x0e"),                       # unused by the backbone itself
        sphere_channels=CHANNELS,
        hidden_channels=CHANNELS,
        edge_channels=CHANNELS,
        num_distance_basis=CHANNELS,          # must match sphere_channels
        lmax=LMAX,
        mmax=LMAX,
        cutoff=6.0,
        num_layers=num_layers,
        act_type="gate",
        mlp_type="spectral",
        include_edges=True,
        wigner_backend=wigner_backend,
        message_type="source-target",
    ).to(device)


def _graph(device, num_molecules=2, atoms_per_molecule=5, seed=0):
    """A synthetic batch carrying exactly the fields the backbone reads."""
    from torch_geometric.data import Data

    gen = torch.Generator(device="cpu").manual_seed(seed)
    natoms = num_molecules * atoms_per_molecule
    pos = torch.randn(natoms, 3, generator=gen)

    # Dense graph minus self-loops, which is what a small-molecule cutoff gives.
    src, dst = torch.meshgrid(
        torch.arange(natoms), torch.arange(natoms), indexing="ij"
    )
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]])

    vec = pos[edge_index[1]] - pos[edge_index[0]]
    dist = vec.norm(dim=1, keepdim=True)
    # edge_attr layout the backbone assumes: [distance, x, y, z].
    edge_attr = torch.cat([dist, vec], dim=1)

    data = Data(
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        atomic_numbers=torch.randint(1, 10, (natoms,), generator=gen),
        charge=torch.zeros(num_molecules, dtype=torch.long),
        spin_multiplicity=torch.ones(num_molecules, dtype=torch.long),
        num_atoms_in_molecule=torch.full(
            (num_molecules,), atoms_per_molecule, dtype=torch.long
        ),
    )
    return data.to(device)


def _seed_gamma(seed=1):
    """Put both gamma-drawing RNGs in a known state.

    The Wigner gauge is an arbitrary roll: the triton backend draws it with
    random.randint, the torch backend with torch.rand_like. Equivariance makes
    it cancel, but only to within fp32 noise -- comparing eager against
    compiled across two different gauges measures that cancellation, not the
    compiler, and was observed to exceed atol on one element in 2304.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def _loss(out):
    return sum(
        v.float().pow(2).sum()
        for v in out.values()
        if torch.is_tensor(v) and v.is_floating_point()
    )


# ---------------------------------------------------------------------------
# 1. The core traces to a single graph
# ---------------------------------------------------------------------------
class TestNoGraphBreaks:

    @pytest.mark.parametrize("wigner_backend", ["torch", "triton"])
    def test_traces_without_breaks(self, wigner_backend, device):
        """A break here is a silent performance cliff, so pin it at zero.

        The PyG graph object is inside the traced region on purpose: it costs
        nothing (measured), so there is no reason to hoist it out. This is the
        test that would catch that changing.
        """
        backbone = _backbone(device, wigner_backend)
        batch = _graph(device)

        # explain() traces _forward_impl directly, so it never passes through
        # enable_compile and never gets the NVTX shim a compiled run installs.
        # Without it the torch.cuda.nvtx annotations in the message-passing
        # path break the graph 23 times (gb0208), which measures the harness,
        # not the model.
        neutralise_nvtx_for_dynamo()

        torch._dynamo.reset()
        explanation = torch._dynamo.explain(backbone._forward_impl)(batch)

        reasons = "\n".join(str(r.reason) for r in explanation.break_reasons)
        assert explanation.graph_break_count == 0, (
            f"{explanation.graph_break_count} graph break(s) in the backbone "
            f"({wigner_backend} wigner):\n{reasons}"
        )
        assert explanation.graph_count == 1

    def test_fullgraph_compile_runs(self, device):
        """fullgraph=True must not raise -- it is unconditional, not a setting."""
        backbone = _backbone(device, "triton")
        spec = backbone.enable_compile(True)
        assert spec is not None
        out = backbone(_graph(device))
        assert out["node_embeddings"].isfinite().all()


# ---------------------------------------------------------------------------
# 2. Compiled must equal eager, forward and backward
# ---------------------------------------------------------------------------
class TestNumericsMatchEager:

    @pytest.mark.parametrize("wigner_backend", ["torch", "triton"])
    def test_forward_matches(self, wigner_backend, device):
        backbone = _backbone(device, wigner_backend)
        batch = _graph(device)

        # The triton wigner draws a gamma per call from Python's `random`, not
        # from torch's RNG, so seed that module -- otherwise the two calls pick
        # different gauges and the comparison measures the gauge, not compile.
        # Both RNGs: the triton backend draws its gamma seed with
        # random.randint, the torch backend with torch.rand_like. Seeding only
        # one leaves the two passes on different Wigner gauges, and the
        # comparison then measures gauge cancellation rather than compilation.
        _seed_gamma()
        eager = backbone(batch)["node_embeddings"].detach().clone()
        backbone.enable_compile(True)
        _seed_gamma()
        compiled = backbone(batch)["node_embeddings"].detach().clone()

        torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)

    def test_gradients_match(self, device):
        """Compiled gradients must match eager to within fp32 accumulation noise.

        Deliberately not element-wise. Message aggregation accumulates with
        atomics, so the *same* eager code run twice already differs by up to
        4.9e-4 absolute on the largest gradient -- one ulp of a value near
        4e3 -- and an element-wise assert_close at atol=1e-5 measures that
        rather than the compiler (it failed on 1 element in 2304, intermittently
        depending on test ordering).

        Per-tensor relative L2 is stable. Measured on this graph: 3.1-3.7e-7
        eager-vs-eager, 2.9e-7 compiled-vs-compiled, 5.5-5.8e-7
        eager-vs-compiled. The 1e-5 threshold leaves more than an order of
        magnitude of headroom while still catching any structural divergence --
        for scale, the L=8 Wigner bug this suite caught was O(1).
        """

        backbone = _backbone(device, "torch")
        batch = _graph(device)

        _seed_gamma()
        _loss(backbone(batch)).backward()
        eager = {n: p.grad.detach().clone()
                 for n, p in backbone.named_parameters() if p.grad is not None}
        backbone.zero_grad(set_to_none=True)

        backbone.enable_compile(True)
        _seed_gamma()
        _loss(backbone(batch)).backward()
        compiled = {n: p.grad.detach().clone()
                    for n, p in backbone.named_parameters() if p.grad is not None}

        assert set(compiled) == set(eager) and eager
        for name, grad in eager.items():
            rel = ((compiled[name] - grad).norm()
                   / grad.norm().clamp_min(1e-12)).item()
            assert rel < 1e-5, (
                f"gradient mismatch for {name}: relative L2 {rel:.3e} "
                "(eager-vs-eager noise floor is ~4e-7)"
            )


# ---------------------------------------------------------------------------
# 3. Compiling must not leak into the checkpoint
# ---------------------------------------------------------------------------
class TestCheckpointUnaffected:

    def test_state_dict_identical(self, device):
        """torch.compile(module) would prefix every key with _orig_mod."""
        backbone = _backbone(device, "torch")
        before = {k: v.clone() for k, v in backbone.state_dict().items()}

        backbone.enable_compile(True)
        backbone(_graph(device))
        after = backbone.state_dict()

        assert not [k for k in after if "_orig_mod" in k]
        assert set(after) == set(before)
        for key, value in before.items():
            torch.testing.assert_close(after[key], value)

    def test_compiled_checkpoint_loads_into_eager(self, device):
        compiled_model = _backbone(device, "torch")
        compiled_model.enable_compile(True)
        compiled_model(_graph(device))

        eager_model = _backbone(device, "torch")
        eager_model.load_state_dict(compiled_model.state_dict())  # must not raise


# ---------------------------------------------------------------------------
# 4. Varying molecule sizes must not trigger endless recompilation
# ---------------------------------------------------------------------------
class TestDynamicShapes:

    def test_new_atom_count_reuses_the_graph(self, device):
        backbone = _backbone(device, "torch")
        backbone.enable_compile(True)

        backbone(_graph(device, atoms_per_molecule=5))
        backbone(_graph(device, atoms_per_molecule=6, seed=1))  # marks dynamic
        before = torch._dynamo.utils.counters["stats"]["unique_graphs"]

        # A third distinct size must now reuse the dynamic graph.
        backbone(_graph(device, atoms_per_molecule=7, seed=2))
        after = torch._dynamo.utils.counters["stats"]["unique_graphs"]

        assert after == before, (
            f"a third atom count compiled {after - before} more graph(s); "
            "dynamic shapes are not taking effect"
        )


# ---------------------------------------------------------------------------
# 5. Unsupported configurations must be refused, not silently degraded
# ---------------------------------------------------------------------------
class TestRefusals:

    def test_distributed_is_refused(self, device):
        backbone = _backbone(device, "torch")
        backbone.distributed_graph_training = True
        with pytest.raises(ValueError, match="distribute_graphs"):
            backbone.enable_compile(True)

    def test_unknown_mode_is_refused(self, device):
        backbone = _backbone(device, "torch")
        with pytest.raises(ValueError, match="compile must be"):
            backbone.enable_compile("turbo")

    @pytest.mark.parametrize("value", [False, None, "off"])
    def test_off_stays_eager(self, value, device):
        backbone = _backbone(device, "torch")
        assert backbone.enable_compile(value) is None
        assert backbone._core_fn == backbone._forward_impl


# ---------------------------------------------------------------------------
# 6. The molecule_indices invariant is enforced in the compiled graph too
# ---------------------------------------------------------------------------
# molecule_indices is built with repeat_interleave(..., output_size=natoms),
# and that argument is a promise torch is only sometimes in a position to
# check: eager validates it, inductor does so only while it keeps the aten
# kernel as a fallback. _forward_impl states the invariant itself with
# torch._assert_async so a batch whose per-molecule counts do not sum to the
# node count can never reach the gather that attaches charges and spins.
#
# This pins the guard and its message, not the underlying kernel: torch's own
# assert also fires today, with a message that says nothing about molecules.
#
# The failure is a CUDA device-side assert, which leaves the context unusable
# for the rest of the process, so the check has to run in a child.
_MALFORMED_BATCH_CHILD = """
import sys, torch
sys.path.insert(0, {tests_dir!r})
from test_torch_compile import _backbone, _graph

device = torch.device("cuda")
backbone = _backbone(device, "torch")
backbone.enable_compile(True)

good = _graph(device, num_molecules=3, atoms_per_molecule=4)
backbone(good)
torch.cuda.synchronize()
print("WARMED", flush=True)

bad = _graph(device, num_molecules=3, atoms_per_molecule=4)
# 12 atoms, but the batch now claims 9 -- the malformed-collate case.
bad.num_atoms_in_molecule = torch.tensor([4, 4, 1], device=device)
backbone(bad)
torch.cuda.synchronize()
print("NO ERROR", flush=True)
"""


class TestBatchInvariant:

    def test_malformed_counts_are_caught_when_compiled(self, device):
        import os
        import subprocess

        script = _MALFORMED_BATCH_CHILD.format(
            tests_dir=os.path.dirname(os.path.abspath(__file__))
        )
        child = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=1800,
        )
        assert "WARMED" in child.stdout, (
            f"child never got a valid batch through:\n{child.stderr[-3000:]}"
        )
        assert "NO ERROR" not in child.stdout, \
            "the compiled forward accepted counts that do not sum to natoms"
        assert child.returncode != 0
        assert "num_atoms_in_molecule does not sum" in child.stderr, \
            f"failed for some other reason:\n{child.stderr[-3000:]}"
