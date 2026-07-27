# NablaDFT QHFlow3-E3 eSEN-grid SCALE

This is one explicit V2 comparison lane:

`NablaDFT | QHFlow3-E3-10x11 | Muon | SCALE | V2`

## Requested changes

- `qhflow3_grid_resolution: null` selects the eSEN `SO3_Grid` default.
  For the NablaDFT `lmax=4` representation this is `10x11`, rather than the
  fixed `48x48` grid used by the existing QHFlow3-E3 V2 runs.
- `scale_shift_mode: standardize` subtracts the train-only l=0 mean and divides
  by the train-only l=0 standard deviation. `SCALE` therefore means full
  standardization, not SHIFT-only normalization.

## Matched controls

- QHFlow3 backbone, three node layers and three edge layers
- native overlap conditioning enabled
- MatrixMuon head with Muon + auxiliary AdamW shape routing
- 20 epochs, 12,081 train rows, 64 validation rows, seed 44
- two ranks, micro-batch 5 per rank, gradient accumulation 2
  (`effective batch = 20`)
- float32 and the same scheduler/learning-rate settings as the matched V2 suite
- fixed train-only statistics artifact with SHA-256
  `375167ad551fb0b60dbe9cd049a4995276b54ce075e09906639ef3daa4f79475`

This single run changes both the grid and normalization relative to
`QHFlow3-E3-48x48 | Muon | SHIFT`; it should not be interpreted as an isolated
one-factor grid ablation.

## Commands

```bash
./_my_script/experiment/2026-07-28/06_nabladft_qhflow3_e3_esen_grid_scale_2gpu.sh validate
./_my_script/experiment/2026-07-28/06_nabladft_qhflow3_e3_esen_grid_scale_2gpu.sh smoke 6,7
```

The durable queue manifest restricts the full run to `server-1`, preserving the
hard reservation of `server-2` GPUs 4,5,6,7.
