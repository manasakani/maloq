# NablaDFT V2 native-head RAW lanes

This isolated suite adds the three missing `Native + RAW` cross-combinations
for the matched NablaDFT V2 comparison. It is self-contained below this
directory and does not import any experiment runner or config from
`2026-07-27`.

Status: prepared and verified. Static/typed checks and a disposable two-GPU
CUDA/DDP smoke passed for all three architectures on 2026-07-28. The durable
queue state for the job IDs below is the authoritative source for enqueue and
full-run status.

## Exact lanes

| CLI lane | Architecture | Edge blocks | Head | Optimizer | Target | W&B display name |
|---|---|---:|---|---|---|---|
| `maloq-e3` | canonical MALOQ | 3 | native `maloq` | Muon + AuxAdamW | RAW | `NablaDFT \| MALOQ-E3 \| Native \| RAW \| V2` |
| `qhflow3-e3` | QHFlow3 clean | 3 | native `maloq` | Muon + AuxAdamW | RAW | `NablaDFT \| QHFlow3-E3 \| Native \| RAW \| V2` |
| `ntev2-e3` | MALOQ-NTE-V2 | 3 | native `maloq` | Muon + AuxAdamW | RAW | `NablaDFT \| NTEV2-E3 \| Native \| RAW \| V2` |

`Native` is the existing native `maloq` Fock head. It does not mean an
AdamW-only training run. For comparability with the existing head axis, all
three lanes keep `optimizer_type: muon`, Muon LR `0.02`, and auxiliary AdamW
LR `5e-4`. The workflow routes eligible trainable matrices to Muon and the
remaining parameters, including the native head's flat contractions, to
AuxAdamW.

RAW is enforced as:

```yaml
scale_and_shift: false
scale_shift_mode: shift_only  # inactive
scale_shift_path: null
```

No SHIFT statistics artifact is consumed by these runs.

## Matched full-run contract

- workflow:
  `maloq.train_utils.training_workflow_v2.TrainingWorkflowV2Fixed`
- database:
  `/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
- ordered split: 12,081 train / 64 validation / 0 test
- two data-parallel GPUs
- micro-batch 5 per rank, accumulation 2, effective batch 20
- 20 epochs, float32, seed 44, shuffle off
- absolute Fock target, no delta learning
- three node blocks and three edge/pair blocks
- trunk/hidden channels 128 and distance basis 512
- MALOQ keeps its native spectral stack and 128-channel output
- NTEV2 and QHFlow3 keep 64 output channels and Grid48 behavior
- QHFlow3 keeps chunk size 512, radius 12, and overlap conditioning
- Muon LR `0.02`, momentum `0.95`, Nesterov, five Newton–Schulz steps
- AuxAdamW LR `5e-4`, betas `(0.9, 0.95)`, epsilon `1e-10`
- weight decay `1e-4`, gradient clipping `1.0`
- 1,000-step polynomial warmup
- W&B project exactly `kaist-korea/MALOQ-nablaDFT-v2`
- W&B group `nabladft-v2-ofat-seed44`
- W&B suite tag `suite:matched-ofat-v2`
- expected total parameters:
  MALOQ `34,489,297`, QHFlow3 `34,382,227`, NTEV2 `33,891,021`

## Files

- config:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/nabladft_native_raw_common.yaml`
- runner:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/run_nabladft_native_raw.py`
- launcher:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/01_nabladft_native_raw_2gpu.sh`
- queue manifest:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/queue_nabladft_native_raw.yaml`
- environment:
  `/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26`

## Commands

Static preparation validates the typed config, database row count/schema, and
all three derived lane contracts without training:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/01_nabladft_native_raw_2gpu.sh prepare all
```

Validate one lane:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/01_nabladft_native_raw_2gpu.sh validate maloq-e3
```

Before enqueueing, run one disposable two-GPU smoke for each architecture:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/01_nabladft_native_raw_2gpu.sh smoke maloq-e3 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/01_nabladft_native_raw_2gpu.sh smoke qhflow3-e3 2,3
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/01_nabladft_native_raw_2gpu.sh smoke ntev2-e3 4,5
```

The launcher requires exactly two distinct GPU indices and rejects an index
with an active compute PID, more than 1,024 MiB used, or more than 10%
utilization. `EXPECTED_HOST` is an optional hostname guard. Successful smoke
outputs are removed; failure evidence is retained.

Manual full-run form:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/native_raw/01_nabladft_native_raw_2gpu.sh full maloq-e3 0,1
```

Outputs use collision-resistant paths below:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-v2-ofat-<lane>-native-raw-2gpu-eb20-mb5-ga2-<scope>-<timestamp>-<pid>/
```

The manifest contains the unique jobs
`nabla-v2-ofat-maloq-e3-native-raw-20260728a`,
`nabla-v2-ofat-qhflow3-e3-native-raw-20260728a`, and
`nabla-v2-ofat-ntev2-e3-native-raw-20260728a`. Each has exactly one `{gpus}`
placeholder and allows either queue worker label `server-1` or `server-2`.
The manifest must not be enqueued until all three CUDA smokes pass and the
current dirty-source fingerprint policy has been reviewed.
