# NablaDFT Muon-compatible MALOQ head comparison

## Purpose

This experiment repeats the matched three-backbone NablaDFT comparison with
`head_type=maloq_muon`. The gated native MALOQ forward is unchanged, but its
degree-split flat e3nn output weights are reparameterized as semantic 2D
matrices. Muon receives the node and edge path-by-channel matrices; gates and
scalar biases remain on AdamW.

For the NablaDFT def2-SVP basis with node symmetry reduction, the semantic row
counts are 136 for nodes and 260 for edges. QHFlow3 and MALOQ-NTE therefore use
`(136, 64)` and `(260, 64)` matrices; the baseline MALOQ lane uses the same row
counts with its configured output channel width.

## Dataset and schedule

- Dataset: native NablaDFT `train_2k.db`
- Source: `/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
- Full split: 12,081 train / 64 validation / 0 test
- Optimizer: Muon for backbone matrices and semantic head matrices; AdamW for
  gates, biases, embeddings, and normalization parameters
- Per-rank micro-batch: 5
- Ranks per lane: 2
- Gradient accumulation: 2
- Effective batch: 20
- Seed: 44

## GPU layout

| Lane | GPUs |
| --- | --- |
| MALOQ | 0,1 |
| MALOQ-NTE | 2,3 |
| QHFlow3 | 6,7 |

## Commands

Run the full-size, one-epoch smoke before starting the long run:

```bash
cd /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-22
./06_nabladft_three_models_muon_head_3x2gpu_mb5_ga2.sh all smoke
```

Run the matched 20-epoch experiment:

```bash
./06_nabladft_three_models_muon_head_3x2gpu_mb5_ga2.sh all full
```

Replace `all` with `maloq`, `maloq-nte`, or `qhflow3` to launch one dedicated
2-GPU lane. The launcher does not reuse output directories.

`maloq_muon` changes the head parameter layout, so native `maloq` head
checkpoints are not loaded into it implicitly. These comparison runs start from
fresh initialization; an explicit checkpoint conversion is required if an old
native-head checkpoint must be resumed.

## Environment and outputs

- Project: `/dataset/seongsu/shared-home/workspace/project`
- Python: `/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python`
- MPI: the matching environment's `mpirun`
- Output root:
  `outputs/nabladft-three-model-muon-head-parallel-3x2gpu-eb20-mb5-ga2-<scope>-seed44-<timestamp>/`
- Logs: `<output-root>/logs/<lane>.log`
- Manifest and status: `<output-root>/launch_manifest.tsv`, `status.tsv`
- W&B full runs: `kaist-korea/maloq-nablaDFT`

## Status

- Semantic/native forward parity unit test: passed.
- Muon routing and optimizer-step unit test: passed.
- Reduced single-GPU NablaDFT CUDA smoke: passed for MALOQ, MALOQ-NTE, and
  QHFlow3 on 2026-07-22. Every lane completed two train samples and one
  validation sample with finite matrix metrics. Comparison artifacts:
  `outputs/nabladft-three-model-muon-head-reduced-smoke-seed44-20260722/`.
- Full-size 3x2-GPU smoke: not launched.
- Full 20-epoch run: not launched.
