# NablaDFT MALOQ Muon-head scale-shift

Compact experiment ID: `nabla-maloq-muon-ss1-v1`.
W&B display name: `NablaDFT | MALOQ | Muon | SS1 | V1`.

This experiment adds train-only label scale/shift to
`nabladft-maloq-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260722-190344-maloq`.
All model, optimizer, split, and batch settings are otherwise unchanged.

Configuration:

- dataset: NablaDFT `train_2k.db`
- ordered split: 12,081 train / 64 validation / 0 test
- model: interleaved MALOQ backbone with corrected `maloq_muon` head
- optimizer: Muon
- epochs: 20
- default GPUs: `0,1`
- micro-batch: 5 per rank
- world size: 2
- gradient accumulation: 2
- effective batch: 20
- scale/shift: element-wise standardization of l=0 node labels
- statistics provenance: ordered training indices `[0, 12081)` only
- W&B: `kaist-korea/maloq-nablaDFT`, every 10 optimizer steps

Scale/shift artifact:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/scale-shift-statistics/nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt
```

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-24/01_nabladft_maloq_muon_head_scale_shift_2gpu.sh prepare 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-24/01_nabladft_maloq_muon_head_scale_shift_2gpu.sh validate 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-24/01_nabladft_maloq_muon_head_scale_shift_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-24/01_nabladft_maloq_muon_head_scale_shift_2gpu.sh full 0,1
```

Outputs:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/nabla-maloq-muon-ss1-v1/
```

The completed V1 run keeps its original local path for provenance:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-maloq-muon-head-ss/
```

The compact experiment ID contains neither the seed nor a date/timestamp.
Seed, scope, and version are W&B tags, and the run belongs to group
`nabla-maloq-ss`.
Repeated full launches refuse to overwrite the existing directory. Successful
smoke output uses `outputs/nabla-maloq-muon-ss1-v1-smoke/` temporarily and
is removed.

Status:

- scale/shift tests passed (`4 passed`)
- configuration validation passed with `scale_and_shift=true` and the expected
  train-only statistics path
- `validate` is a single-process config preview, so it reports effective batch
  10; the two-rank `smoke` and `full` commands use effective batch 20
- full training completed for 20 epochs with exit code 0
- W&B run `jal9l7uk` is named
  `NablaDFT | MALOQ | Muon | SS1 | V1` and grouped with MALOQ SS0 under
  `nabla-maloq-ss`
