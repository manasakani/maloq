# NablaDFT E3 Muon+SHIFT node-reduction ablation

This three-lane OFAT control repeats the matched MALOQ-E3, NTEV2-E3, and
QHFlow3-E3 Muon+SHIFT experiments with only these matrix-head options changed:

```yaml
reduce_node: false
reduce_node_intra: false
```

`reduce_edge` remains `false`. Dataset, ordered 12,081/64/0 split, model
width, three edge layers, Muon/AuxAdamW routing, SHIFT-only statistics,
micro-batch 5 per rank, accumulation 2, seed 44, float32, and 20 epochs are
inherited from and validated against the 2026-07-27 OFAT suite.

W&B display names add `NoNodeReduce` before `V2`, and run/output identifiers
add `no-node-reduction`, so these runs cannot overwrite or be mistaken for
the completed baselines.

## Files

- runner:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/run_nabladft_no_node_reduction_shift.py`
- launcher:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/03_nabladft_no_node_reduction_shift_2gpu.sh`
- queue manifest:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/queue_nabladft_no_node_reduction_shift.yaml`
- inherited base runner:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/run_nabladft_v2_ofat.py`
- inherited base config:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/nabladft_v2_ofat_common.yaml`

## Commands

Validation does not initialize training:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/03_nabladft_no_node_reduction_shift_2gpu.sh validate all
```

Disposable two-GPU CUDA/DDP smokes:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/03_nabladft_no_node_reduction_shift_2gpu.sh smoke qhflow3-e3-muon-shift 4,5
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/03_nabladft_no_node_reduction_shift_2gpu.sh smoke ntev2-e3-muon-shift 4,5
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/03_nabladft_no_node_reduction_shift_2gpu.sh smoke maloq-e3-muon-shift 4,5
```

The launcher owns collision-resistant output directories below:

`/dataset/seongsu/shared-home/workspace/project/outputs/`

Full jobs are submitted through the durable queue:

- `nabla-v2-qhflow3-e3-muon-shift-nodered-off-20260728a`
- `nabla-v2-ntev2-e3-muon-shift-nodered-off-20260728a`
- `nabla-v2-maloq-e3-muon-shift-nodered-off-20260728a`

The queue manifest allows either SC26 server, requests two GPUs per job, uses
priority 25, and contains exactly one `{gpus}` placeholder in each job.

Status: prepared; validation, smoke, and durable queue state are authoritative.
