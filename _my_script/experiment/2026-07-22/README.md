# QH9Stable three-model training commands

NablaDFT full 20-epoch Muon commands are documented separately in
`README_nabladft_muon_20epoch.md` and
`03_nabladft_muon_20epoch_script.sh`.

The Muon-compatible native MALOQ head comparison is documented in
`README_nabladft_muon_head_3x2gpu.md` and launched with
`06_nabladft_three_models_muon_head_3x2gpu_mb5_ga2.sh`.

The QHFlow3-only version that explicitly pins
`src/maloq/helm/qhflow3_clean.py::QHFlow3MaloqBackbone` together with the
Muon-compatible head is documented in
`README_nabladft_qhflow3_clean_muon_head.md` and launched with
`08_nabladft_qhflow3_clean_muon_head_2gpu_mb5_ga2.sh`.

Individual MALOQ and MALOQ-NTE two-GPU versions with the same Muon-compatible
head are documented in `README_nabladft_maloq_muon_head_2gpu.md` and launched
with `09_nabladft_maloq_muon_head_2gpu_mb5_ga2.sh` and
`10_nabladft_maloq_nte_muon_head_2gpu_mb5_ga2.sh`.

QHFlow3 is integrated into the existing NablaDFT launchers:
`03_nabladft_muon_20epoch_script.sh` for full runs and
`04_nabladft_2gpu_effective_batch20.sh` for directly executable smoke/full
runs. W&B step losses use a 10-optimizer-step cadence.

For concurrent dedicated pairs (MALOQ on 0,1; MALOQ-NTE on 2,3; QHFlow3 on
4,5), use `05_nabladft_three_models_3x2gpu_mb5_ga2.sh`. It uses per-rank
micro-batch 5 and gradient accumulation 2 for effective batch 20.

Successful smoke runs are ephemeral by default: local outputs are deleted and
W&B is disabled. Failed smoke outputs remain available for diagnosis. Full-run
outputs and W&B records are always retained.

## QH9Stable target-separated delta comparison on scp-gpu-2

`06_qh9stable_delta_scp_gpu2_bs32.sh` compares MALOQ, MALOQ-NTE, and QHFlow3
for both target-separated delta tasks. Hamiltonian delta uses
`QH9Stable_random.db`; density delta uses `QH9StableMatrices_random.db`.
Each lane uses one H100 with batch/effective batch 32. Full runs use Muon,
80 epochs by default, W&B project `kaist-korea/maloq-qh9`, ten-step logging,
and epoch summaries. Outputs stay below `outputs/`.

Run from `scp-gpu-2` after validation and the non-persistent smoke pass:

```bash
cd /dataset/seongsu/shared-home/workspace/project
./_my_script/experiment/2026-07-22/06_qh9stable_delta_scp_gpu2_bs32.sh both validate all
./_my_script/experiment/2026-07-22/06_qh9stable_delta_scp_gpu2_bs32.sh both smoke all
./_my_script/experiment/2026-07-22/06_qh9stable_delta_scp_gpu2_bs32.sh both full all
```

GPU lanes are H/MALOQ=0, H/MALOQ-NTE=1, H/QHFlow3=2,
D/MALOQ=3, D/MALOQ-NTE=4, and D/QHFlow3=5. Override the full epoch count with
`NUM_EPOCHS=<n>` if needed. Smoke artifacts are removed after all lanes pass.

## Purpose

This command sheet launches the three matched QH9Stable production lanes:

| Model | Runner variant | Backbone |
|---|---|---|
| MALOQ | `maloq` | native interleaved 3 node + 3 edge blocks |
| MALOQ-NTE | `maloq-nte` | node-then-edge 3 node + 2 edge blocks |
| QHFlow3 | `qhflow3` | equivariant grid48 QHFlow3 3 node + 2 pair blocks |

There are no user-facing `baseline`, `both`, or `maloq-vs-nte` variants.
`all` means exactly the three rows above, trained sequentially under matched
conditions. Each completed comparison writes both `comparison.json` and
`comparison.csv` at the output root.

All lanes use the original QM7 loader contract, native MALOQ coupled-irrep
head, Muon/AdamW recipe, 80 epochs, batch size 32, gradient clipping 1.0,
seed 44, `shuffle=false`, and `distribute_graphs=false`.

## Command sheet

File:
`_my_script/experiment/2026-07-22/01_qh9stable_maloq_nte_qhflow3_script.sh`

Inspect it and copy exactly one complete `(a)`-`(d)` block into the shell:

```bash
cd /dataset/seongsu/shared-home/workspace/project
sed -n '1,260p' \
  _my_script/experiment/2026-07-22/01_qh9stable_maloq_nte_qhflow3_script.sh
```

- `(a)`: MALOQ only
- `(b)`: MALOQ-NTE only
- `(c)`: QHFlow3 only
- `(d)`: all three sequentially for the matched comparison

Change only `GPU=0` when selecting another GPU. Each block assigns a distinct
local master port and a timestamped output directory below `outputs/`.

## Dataset prerequisites

The target-separated production commands require both databases:

- Hamiltonian delta: `/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db`
- Density delta: `/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9StableMatrices_random.db`

The Hamiltonian DB is rebuilt from the read-only raw QH9Stable H/S source and
the source-index-aligned B3LYP5 LMDB H0/D0 values. It is staged as a separate
candidate and validated before the older absolute-only DB is archived and the
candidate takes the canonical name. The density DB now advertises D/D0 only;
H0 remains QHFlow3 conditioning and final H is not loader-visible.

Both preserve the official split membership: 104,664 train, 13,083 validation,
and 13,084 test structures. QH9Dynamic remains out of scope. Production should
only start after `both validate all` and `both smoke all` pass on scp-gpu-2.

## Environment, outputs, and status

- Project: `/dataset/seongsu/shared-home/workspace/project`
- Python: `/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python`
- GPU: six single-GPU lanes on scp-gpu-2 GPUs 0-5
- Outputs: `outputs/qh9stable-delta-<target>-six-lane-bs32-<scope>-seed44-<timestamp>/`
- Smoke status: all three production-size model paths have completed CUDA
  forward/backward and validation on the ordered 2/1/1 QH9Stable sample.
- QHFlow3 SO(3) status: grid48 backbone and MALOQ head pass the float32 `1e-4`
  general-rotation tolerance; grid FFNs use checkpoint chunks of 512.
- Density DB profile: density-only metadata applied and audited.
- Hamiltonian DB profile: full 130,831-row delta rebuild validated and promoted;
  the previous absolute-only DB is archived as
  `QH9Stable_random.absolute-only-20260722.db`.
- scp-gpu-2 smoke status: all six H/D × model lanes passed one full-size
  batch-32 train/validation epoch; smoke outputs were removed.
- Full 80-epoch training status: not launched.

## Corrected static TE lane

`qhflow3_static_te_qh9stable.yaml` replaces the native gated MALOQ head with
SC26's gate-free static tensor-expansion head. Its node and edge contraction
weights remain semantic `(path row, channel)` matrices and are scattered to
degree-first output positions with the corrected `path_offsets` mapping. The
two matrices are routed through Muon; scalar biases remain on AdamW.

Run the ordered 2/1/1 QH9Stable CUDA smoke with:

```bash
cd /dataset/seongsu/shared-home/workspace/project
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
PYTHONPATH=src "${PY}" \
  _my_script/experiment/2026-07-22/run_qhflow3_static_te_qh9stable.py \
  --smoke --gpu 0
```

Remove `--smoke` for the configured 80-epoch run. Full and smoke artifacts are
written to distinct timestamped directories below `outputs/`. Dataset:
QH9Stable official random split; environment: `proj-dft-baselines-maloq-sc26`;
GPU selection: `--gpu`; full-run status: not launched.

CUDA smoke status: passed on 2026-07-22 with two train steps and one validation
step. Artifacts are in
`outputs/qh9stable-qhflow3-static-te-smoke-seed44-20260722-062144/`; validation
matrix MAE was `0.137507`, and both versioned semantic matrices received
nonzero updates.

## NTE reference LayerScale, zero-init, and L3/L4 gate

The quasar/ml-dft audit and two directly executable QH9Stable density-delta
presets are documented in `README_nte_reference_tricks.md`. Run
`07_qh9stable_nte_reference_tricks.sh validate all` before smoke/full launch.
