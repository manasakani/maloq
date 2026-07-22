# NablaDFT QHFlow3-clean with MUON head

## Purpose

This is the explicit QHFlow3 lane for the NablaDFT comparison. The model uses
`src/maloq/helm/qhflow3_clean.py::QHFlow3MaloqBackbone`, not a legacy QHFlow3
inheritance path, and attaches `head_type=maloq_muon`. The QHFlow3 trunk and
corrected head use the single fixed Muon rule: every trainable parameter with
`ndim >= 2` uses Muon. Biases, normalization vectors, and other one-dimensional
parameters use AdamW. There is no routing-policy option.

## Dataset and schedule

- Dataset: native NablaDFT `train_2k.db`
- Source: `/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
- Full split: 12,081 train / 64 validation / 0 test
- QHFlow3 basis bridge: `def2-svp-nabla`
- QHFlow3 trunk: 3 node blocks, 2 pair blocks, grid resolution 48
- Per-rank micro-batch: 5
- Data-parallel ranks: 2
- Gradient accumulation: 2
- Effective batch: 20
- Full schedule: 20 epochs, seed 44

## Commands

```bash
cd /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-22

# CPU-side dataset/config validation
./08_nabladft_qhflow3_clean_muon_head_2gpu_mb5_ga2.sh validate

# Full-size model, one-epoch 2-GPU smoke
./08_nabladft_qhflow3_clean_muon_head_2gpu_mb5_ga2.sh smoke

# 20-epoch production run
./08_nabladft_qhflow3_clean_muon_head_2gpu_mb5_ga2.sh full
```

The default GPUs are `6,7`. Override them only when intentionally moving this
lane to another pair, for example
`GPUS=4,5 MASTER_PORT=29584 ./08_nabladft_qhflow3_clean_muon_head_2gpu_mb5_ga2.sh smoke`.

## Environment and outputs

- Python: `/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python`
- Config: `qhflow3_clean_muon_head_nabladft.yaml`
- Outputs: `outputs/nabladft-qhflow3-clean-muon-head-2gpu-eb20-mb5-ga2-<scope>-seed44-<timestamp>/`
- Training log: `<output>/logs/qhflow3.log`
- Full-run W&B project: `kaist-korea/maloq-nablaDFT`

## Status

- Config and dataset validation: passed on 2026-07-22. The resolved model uses
  `backbone_type=qhflow3_clean`, `head_type=maloq_muon`, and the native
  NablaDFT 12,081/64 split. Resolved Muon routing is
  `all_trainable_ndim_ge_2`.
- Atom-aligned matrix-conditioning regression, QHFlow3 grid48 equivariance,
  and MUON-head routing/parity tests: 7 passed.
- The atom-alignment regression changes overlap, initial density, and initial
  Hamiltonian blocks only for the second atom and verifies that the first
  atom's conditioning is unchanged.
- Full-size 2-GPU smoke: not launched.
- Full 20-epoch run: not launched.
