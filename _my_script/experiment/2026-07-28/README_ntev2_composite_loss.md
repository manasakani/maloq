# NablaDFT NTEV2 matrix composite-loss axis

This experiment keeps the completed `NablaDFT | NTEV2-E3 | Muon | SHIFT | V2`
baseline fixed and changes only the training loss callable to
`RMSE + MSE + MAE`. The common comparison metric remains the unscaled
validation matrix MAE/MSE.

A `10 * (RMSE + MSE + MAE)` profile remains available inside the isolated
experimental package, but its full job was cancelled before start after the
comparison scope was narrowed to the scale-1 NTEV2 lane.

- Dataset: NablaDFT `train_2k.db`
- Split: ordered 12,081 train / 64 validation / 0 test
- Model: NTEV2, 3 node layers, 3 edge layers, Muon-visible head
- Parameters: 33,891,021 expected
- SHIFT: training-only l=0 mean subtraction, checksum pinned
- Optimizer: Muon + AuxAdamW, matched baseline settings
- Effective batch: 20 (5 per rank, 2 ranks, accumulation 2)
- Epochs: 20
- Seed: 44
- GPUs: exactly two on server 1; server 2 is excluded because GPUs 4-7 are reserved
- Outputs: `/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/nabladft-matrix-composite-loss-*`

Validate:

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/09_nabladft_ntev2_composite_loss_2gpu.sh validate all
```

Smoke one profile on two idle server-1 GPUs:

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/09_nabladft_ntev2_composite_loss_2gpu.sh smoke rmse-mse-mae 0,1
```

The scale-1 full run is submitted through
`queue_nabladft_ntev2_composite_loss.yaml` after its own two-rank CUDA smoke.

Status: CPU tests and the scale-1 two-rank CUDA smoke passed. The 20-epoch run
is active on server-1 GPU 6,7 as queue job
`nabla-v2-ntev2-e3-muon-shift-rmse-mse-mae-20260728a`; W&B run `xvx2sdyl`.

The active job above is direct regression, not FlowMatching. The separate
FlowMatching composition uses
`queue_nabladft_ntev2_flow_matching_rmse_mse_mae.yaml`, a distinct W&B name,
and the `nabladft-flow-matching-rmse-mse-mae-*` output prefix.
