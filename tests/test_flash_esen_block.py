# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

"""
Tests for the FlashSO2-backed eSEN block (the `flash_esen_block` config key).

Flash_eSEN_Block replaces the Edgewise path with FlashSO2 while leaving every
trainable module owned by the native block. These tests pin what that design
has to guarantee:

  - the checkpoint cannot tell whether flash was on,
  - the numbers and the gradients match the native block,
  - the FlashSO2 view stays wired to the parameters through the module
    operations training performs (.to(), train/eval, an optimizer step),
  - configurations FlashSO2 cannot honour are refused at construction rather
    than silently computing something else.

The tests are written against behaviour, not structure: they compare a flash
backbone with a native one holding the same weights, so they keep their meaning
if the block's internals are refactored.

Backbones are built directly from a synthetic graph rather than through
TrainingWorkflow, so these run without a dataset.
"""

import copy
import random

import pytest
import torch

from e3nn.o3 import Irreps

from maloq.helm.esen_osh import eSEN_Backbone


flash_so2 = pytest.importorskip(
    "flash_so2", reason="Flash_eSEN_Block requires the optional flash_so2 package"
)

# FlashSO2 supports lmax in {4, 6, 8}; 4 is what the nablaDFT basis produces.
LMAX = 4
CHANNELS = 32


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _backbone(device, flash="fp32", num_layers=1, **overrides):
    torch.manual_seed(0)
    kwargs = dict(
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
        # FlashSO2 consumes the packed block-diagonal Wigner form, which only
        # the Triton backend writes.
        wigner_backend="triton",
        message_type="source-target",
        flash_esen_block=flash,
    )
    kwargs.update(overrides)
    return eSEN_Backbone(Irreps("1x0e"), **kwargs).to(device)


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
    """Put the gamma-drawing RNGs in a known state.

    The Wigner gauge is an arbitrary roll drawn per call. Equivariance makes it
    cancel, but only to within fp32 noise -- comparing the flash path against
    the native one across two different gauges measures that cancellation
    rather than the block.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def _loss(out):
    return sum(
        v.float().pow(2).sum()
        for v in out.values()
        if torch.is_tensor(v) and v.is_floating_point()
    )


def _worst_rel(a, b):
    worst = 0.0
    for key, value in a.items():
        if torch.is_tensor(value) and value.is_floating_point() and value.numel():
            diff = (value - b[key]).abs().max().item()
            worst = max(worst, diff / max(value.abs().max().item(), 1e-12))
    return worst


def _blocks(backbone):
    return list(backbone.node_blocks) + list(backbone.edge_blocks)


@pytest.fixture(scope="module")
def pair(device):
    """A flash backbone and a native one holding identical weights."""
    native = _backbone(device, flash=None)
    flash = _backbone(device, flash="fp32")
    flash.load_state_dict(native.state_dict())
    return native, flash


# ---------------------------------------------------------------------------
# 1. The checkpoint must not record whether flash was on
# ---------------------------------------------------------------------------
# Flash_eSEN_Block keeps the FlashSO2 convolution out of Module._modules
# precisely so that state_dict stays the native layout. If that ever slipped,
# checkpoints would stop moving between a flash run and a native one -- which
# is how the two are compared in the first place.
class TestCheckpointUnaffected:

    def test_state_dict_keys_match_native(self, pair):
        native, flash = pair
        assert list(flash.state_dict().keys()) == list(native.state_dict().keys())

    def test_flash_convolution_is_not_in_the_state_dict(self, pair):
        _, flash = pair
        assert not [k for k in flash.state_dict() if "flash" in k.lower()]

    def test_native_checkpoint_loads_into_flash(self, device, pair):
        native, _ = pair
        _backbone(device, flash="fp32").load_state_dict(
            native.state_dict(), strict=True
        )

    def test_flash_checkpoint_loads_into_native(self, device, pair):
        _, flash = pair
        _backbone(device, flash=None).load_state_dict(flash.state_dict(), strict=True)

    def test_parameters_are_not_duplicated(self, pair):
        """The convolution references modules already registered under edge_wise.

        If it were registered too, every conv weight would reach the optimizer
        twice and be stepped twice per iteration.
        """
        native, flash = pair
        ids = [id(p) for p in flash.parameters()]
        assert len(ids) == len(set(ids))
        assert len(ids) == len(list(native.parameters()))


# ---------------------------------------------------------------------------
# 2. Flash must equal native, forward and backward
# ---------------------------------------------------------------------------
class TestNumericsMatchNative:

    def test_forward_matches(self, pair):
        native, flash = pair
        native.eval()
        flash.eval()
        batch = _graph(next(native.parameters()).device)

        _seed_gamma()
        with torch.no_grad():
            expected = native(batch)
        _seed_gamma()
        with torch.no_grad():
            got = flash(batch)

        assert _worst_rel(expected, got) < 2e-4

    def test_gradients_match(self, pair):
        native, flash = pair
        native.train()
        flash.train()
        batch = _graph(next(native.parameters()).device)

        _seed_gamma()
        native.zero_grad()
        _loss(native(batch)).backward()
        expected = {
            k: p.grad.detach().clone()
            for k, p in native.named_parameters()
            if p.grad is not None
        }

        _seed_gamma()
        flash.zero_grad()
        _loss(flash(batch)).backward()
        got = {
            k: p.grad.detach().clone()
            for k, p in flash.named_parameters()
            if p.grad is not None
        }

        assert set(expected) == set(got)
        for key in expected:
            scale = max(expected[key].abs().max().item(), 1e-12)
            assert (expected[key] - got[key]).abs().max().item() / scale < 5e-3, key

    def test_every_parameter_receives_a_gradient(self, pair):
        """A detached parameter trains silently and forever at its init value."""
        _, flash = pair
        missing = [
            k for k, p in flash.named_parameters() if p.requires_grad and p.grad is None
        ]
        assert not missing

    @pytest.mark.parametrize(
        "num_molecules,atoms_per_molecule",
        [(1, 6), (3, 1), (4, 9)],
        ids=["single-molecule", "single-atom-molecules", "larger-batch"],
    )
    def test_matches_native_on_other_batch_shapes(
        self, pair, num_molecules, atoms_per_molecule
    ):
        native, flash = pair
        native.eval()
        flash.eval()
        batch = _graph(
            next(native.parameters()).device,
            num_molecules=num_molecules,
            atoms_per_molecule=atoms_per_molecule,
        )

        _seed_gamma()
        with torch.no_grad():
            expected = native(batch)
        _seed_gamma()
        with torch.no_grad():
            got = flash(batch)

        assert _worst_rel(expected, got) < 2e-4

    def test_matches_native_with_several_layers(self, device):
        """One layer cannot show an error that only compounds across blocks."""
        native = _backbone(device, flash=None, num_layers=3)
        flash = _backbone(device, flash="fp32", num_layers=3)
        flash.load_state_dict(native.state_dict())
        native.eval()
        flash.eval()
        batch = _graph(device)

        _seed_gamma()
        with torch.no_grad():
            expected = native(batch)
        _seed_gamma()
        with torch.no_grad():
            got = flash(batch)

        assert _worst_rel(expected, got) < 2e-4


# ---------------------------------------------------------------------------
# 3. The unregistered convolution must survive what training does to a module
# ---------------------------------------------------------------------------
# Keeping the convolution out of _modules is what protects the state_dict, but
# it also means nn.Module's recursive operations do not reach it. Each of these
# is an operation maloq actually performs on the backbone.
class TestModuleProtocol:

    def test_train_and_eval_reach_the_convolution(self, pair):
        _, flash = pair
        blocks = _blocks(flash)
        assert blocks

        flash.eval()
        assert all(b._flash_so2_convolution.training is False for b in blocks)
        flash.train()
        assert all(b._flash_so2_convolution.training is True for b in blocks)

    def test_to_device_after_cpu_construction(self, device):
        """build_model constructs on CPU and then moves, so this is the real path.

        The convolution holds module references rather than weight tensors, so
        an in-place .to() has to leave it looking at the moved parameters.
        """
        flash = _backbone(torch.device("cpu"), flash="fp32").to(device)

        block = flash.node_blocks[0]
        assert block._flash_so2_convolution.conv1_block is block.edge_wise.so2_conv_1
        assert next(block._flash_so2_convolution.parameters()).is_cuda

        _seed_gamma()
        flash.eval()
        with torch.no_grad():
            flash(_graph(device))

    def test_optimizer_step_is_visible(self, device):
        """FlashSO2 caches prepared weights, so a stale cache would freeze training.

        The cache is fingerprinted on the source weights; this pins that an
        in-place optimizer update actually invalidates it.
        """
        flash = _backbone(device, flash="fp32")
        batch = _graph(device)

        flash.eval()
        _seed_gamma()
        with torch.no_grad():
            before = _loss(flash(batch)).item()

        optimizer = torch.optim.SGD(flash.parameters(), lr=1e-2)
        flash.train()
        _seed_gamma()
        optimizer.zero_grad()
        _loss(flash(batch)).backward()
        optimizer.step()

        flash.eval()
        _seed_gamma()
        with torch.no_grad():
            after = _loss(flash(batch)).item()

        assert abs(after - before) > 1e-6

    def test_deepcopy_keeps_the_convolution_with_its_own_modules(self, pair):
        """A copy whose convolution still pointed at the original's weights
        would train one model and evaluate another."""
        _, flash = pair
        clone = copy.deepcopy(flash)
        block = clone.node_blocks[0]
        assert block._flash_so2_convolution.conv1_block is block.edge_wise.so2_conv_1


# ---------------------------------------------------------------------------
# 4. Unsupported configurations must be refused, not silently degraded
# ---------------------------------------------------------------------------
class TestRefusals:

    def test_requires_the_triton_wigner_backend(self, device):
        """The dense torch backend never writes the packed form FlashSO2 reads."""
        with pytest.raises(ValueError, match="wigner_backend"):
            _backbone(device, flash="fp32", wigner_backend="torch")

    @pytest.mark.parametrize("precision", ["float32", "fp16", "emu5", "bogus", ""])
    def test_unknown_precision_is_refused(self, device, precision):
        with pytest.raises(ValueError, match="must be one of"):
            _backbone(device, flash=precision)

    @pytest.mark.parametrize("precision", ["FP32", "Fp32", " fp32 "])
    def test_precision_spelling_is_normalised(self, device, precision):
        """The block normalises case and surrounding space; the CLI config does not.

        ModelConfig.flash_esen_block is a lowercase Literal, so a CLI run is
        held to the exact spelling. TrainingWorkflow does not validate through
        MaloqConfig, so a script config reaches the block directly -- and the
        block accepting "FP32" is what keeps that from being a crash.
        """
        _backbone(device, flash=precision)

    @pytest.mark.parametrize("lmax", [2, 3, 5])
    def test_unsupported_lmax_is_refused(self, device, lmax):
        with pytest.raises(ValueError, match="lmax"):
            _backbone(device, flash="fp32", lmax=lmax, mmax=lmax)

    def test_source_target_message_is_refused(self, device):
        """This guard is load-bearing, not defensive.

        Edgewise concatenates x_message_edge into the message only when
        message_type is 'source-target-message'. The FlashSO2 path does not
        take x_message_edge at all, so without this refusal that configuration
        would train quietly on a different message.
        """
        with pytest.raises(ValueError, match="message_type"):
            _backbone(device, flash="fp32", message_type="source-target-message")

    def test_distributed_graph_training_is_refused(self, device):
        with pytest.raises(ValueError, match="distribut"):
            _backbone(device, flash="fp32", distributed_graph_training=True)

    @pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
    def test_supported_precisions_construct(self, device, precision):
        _backbone(device, flash=precision)


# ---------------------------------------------------------------------------
# 5. Flash and torch.compile must compose
# ---------------------------------------------------------------------------
# Nothing forces these two features to be used together, but the config permits
# it and eSEN_Backbone._check_compilable deliberately does not refuse it.
class TestComposesWithCompile:

    def test_compiled_flash_matches_eager_flash(self, device):
        flash = _backbone(device, flash="fp32")
        flash.eval()
        batch = _graph(device)

        _seed_gamma()
        with torch.no_grad():
            eager = flash(batch)

        flash.enable_compile(True)
        _seed_gamma()
        with torch.no_grad():
            compiled = flash(batch)

        assert _worst_rel(eager, compiled) < 2e-4

    def test_compiling_leaves_the_checkpoint_clean(self, device, pair):
        native, _ = pair
        flash = _backbone(device, flash="fp32")
        flash.enable_compile(True)

        keys = list(flash.state_dict().keys())
        assert not [k for k in keys if "_orig_mod" in k or "flash" in k.lower()]
        assert keys == list(native.state_dict().keys())
