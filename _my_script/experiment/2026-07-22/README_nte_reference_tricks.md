# QH9Stable NTE reference-trick audit and launch

The closest locally auditable quasar/ml-dft artifact is:

`qh9_b3lyp5_maloq0713_nte_qhflow3_parity_degree_rms_physical_p1_layerscale_s64_full_seed42`

It uses NTE 3-node/2-edge blocks, sigmoid grid MLPs, edge envelope/scalar
modulation, 64 output channels, bounded-degree LayerScale initialized to 1/64,
a zero-initialized coefficient head, and MuonAdamW routing every trainable
parameter with `ndim >= 2` through Muon.

SC26 now has one fixed Muon routing rule: every trainable parameter with
`ndim >= 2` uses Muon. Consequently backbone, static head, and L3/L4 gate
matrices use Muon; biases, normalization vectors, and one-dimensional
LayerScale parameters use AdamW. There is no routing-policy option.

The corrected static head uses semantic `(path row, channel)` matrices,
`path_offsets` output scattering, and an explicit `1 / channels` mean. With
`static_te_init_mode: zero`, both node and edge coefficients start at zero.
The optional degree gate is computed only from invariant L0 channels and
applied as a scalar to all m-components of L3/L4, so it remains equivariant.
Its `residual_tanh` form starts at identity when `static_te_gate_init: 1.0`.

## Presets

| Preset | Residual scaling | Static head | Degree gate |
|---|---|---|---|
| `zero-channel-mean-layerscale-s64` | bounded-degree, init 1/64 | zero-init, channel mean | none |
| `degreewise-l34-gate` | none | zero-init, channel mean | invariant L3/L4 residual-tanh, init 1 |

Both train QH9Stable B3LYP5 density residuals from `initial_density_matrix`,
batch 32, 80 epochs, seed 42, and log every ten optimizer steps plus epoch
summaries to `kaist-korea/maloq-qh9`. Initial Hamiltonian and overlap are
present in the converted DB but the SC26 static head does not yet condition on
them; this is a remaining difference from the ml-dft head.

## Commands

```bash
cd /dataset/seongsu/shared-home/workspace/project

# Read-only data/config validation for both presets.
./_my_script/experiment/2026-07-22/07_qh9stable_nte_reference_tricks.sh validate all

# CUDA smoke on GPUs 0 and 1. Successful smoke outputs are deleted.
./_my_script/experiment/2026-07-22/07_qh9stable_nte_reference_tricks.sh smoke all

# Full 80-epoch runs with W&B enabled. Override lanes if desired.
ZERO_GPU=0 GATE_GPU=1 \
  ./_my_script/experiment/2026-07-22/07_qh9stable_nte_reference_tricks.sh full all
```

Full outputs are timestamped below `outputs/`. Full training is not launched
by repository setup or validation commands.
