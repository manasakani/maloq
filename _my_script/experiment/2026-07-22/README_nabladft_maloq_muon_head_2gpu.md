# NablaDFT MALOQ/MALOQ-NTE with MUON head

These are the individual two-GPU versions of the MALOQ and MALOQ-NTE lanes
from the matched three-model experiment. Both use the corrected
`head_type=maloq_muon`. Muon routing has one fixed rule: all trainable
parameters with `ndim >= 2`, including semantic output and gate matrices, use
Muon. Biases, normalization vectors, and one-dimensional LayerScale parameters
use AdamW.

## Model definitions

| Model | Backbone schedule | Output channels | Default GPUs |
| --- | --- | ---: | --- |
| MALOQ | interleaved, spectral, 3 node + 3 edge | 128 | 0,1 |
| MALOQ-NTE | node-then-edge, grid, 3 node + 2 edge | 64 | 2,3 |

Both use native NablaDFT `train_2k.db`, the 12,081/64/0 split, per-rank
micro-batch 5, two data-parallel ranks, gradient accumulation 2, effective
batch 20, seed 44, and a 20-epoch full schedule.

The matching QHFlow3-clean lane uses the same optimizer, batch, accumulation,
epoch, and tracking settings in
`08_nabladft_qhflow3_clean_muon_head_2gpu_mb5_ga2.sh` (default GPUs 6,7).

## Commands

```bash
cd /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-22

./09_nabladft_maloq_muon_head_2gpu_mb5_ga2.sh validate
./09_nabladft_maloq_muon_head_2gpu_mb5_ga2.sh smoke
./09_nabladft_maloq_muon_head_2gpu_mb5_ga2.sh full

./10_nabladft_maloq_nte_muon_head_2gpu_mb5_ga2.sh validate
./10_nabladft_maloq_nte_muon_head_2gpu_mb5_ga2.sh smoke
./10_nabladft_maloq_nte_muon_head_2gpu_mb5_ga2.sh full
```

Override GPU pairs and ports with `GPUS=6,7 MASTER_PORT=29585` before a
command. The project interpreter is
`/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python`.
The `validate` command runs one process, so its preview reports effective batch
10. The `smoke` and `full` commands launch two MPI ranks and therefore use the
intended effective batch 20.

Outputs are written below
`outputs/nabladft-<model>-muon-head-2gpu-eb20-mb5-ga2-<scope>-seed44-<timestamp>/`.
Full runs log to W&B project `kaist-korea/maloq-nablaDFT`.

## Status

- Both configs and the native NablaDFT 12,081/64 split: validated on
  2026-07-22.
- Shell syntax and MUON-head parity/routing/optimizer-step tests: passed.
- Reduced single-GPU CUDA training and validation for both architecture paths:
  passed as part of the three-model MUON-head smoke comparison.
- Full-size two-GPU smoke runs: not launched.
- Full 20-epoch runs: not launched.
