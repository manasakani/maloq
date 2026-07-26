# NablaDFT EdgeBlock-1 atomwise-output control

This final-MAE control starts from completed SplitOutNorm run `n7z3h6o0` and
changes only `model.direct_atomwise_layers: [1]`. It retrains from seed 44;
it does not warm-start the completed checkpoint.

## Exact layer-operation delta

EdgePre normalizes the final node state before EdgeBlock 1, while
`direct_edgewise_layers: [1]` already prevents the initial edge state from
being added back at the first edgewise boundary. Let:

```text
m1 = EdgeWise1(Norm1(node), E0, geometry)
a1 = AtomWise1(Norm2(m1))
```

The completed SplitOutNorm base and this candidate differ only at the
EdgeBlock-1 return:

```text
SplitOutNorm base: E1 = m1 + AtomLayerScale1(a1)
Edge1AtomDirect:   E1 = a1
```

Thus the candidate removes the first atomwise residual and its bounded
degree-wise LayerScale from the E1 return, matching QHFlow3 `xy2`'s direct
atomwise return at that boundary. EdgeBlock 2 remains recurrent and
`residual_scaled`; `direct_edgewise_layers: [1]`, EdgePre, QHF conditioning,
QHFlow3 irrep projection, separate node/edge output norms, and the MALOQ head
remain unchanged.

## Base and selection thresholds

The primary selection metric is epoch-20 validation matrix MAE. Epoch 16-20
means are stability diagnostics.

- SplitOutNorm `n7z3h6o0`
  - final matrix/node/edge:
    `5.3684045842383014e-5 / 1.9448897111227616e-4 /
    6.472260160257549e-5`
  - epoch 16-20 mean matrix/node/edge:
    `7.998210265337332e-5 / 3.9827224251590593e-4 /
    9.206929769077577e-5`
- QHFlow3 comparator `aeorq52s`
  - final matrix MAE: `5.216202903956794e-5`

SplitOutNorm improved final matrix MAE by 3.5199% relative to `tq5e9a5p` and
remains 2.9179% above QHFlow3. This candidate must finish below
`5.3684045842383014e-5` to become the best NTE lane. Matching QHFlow3 requires
`<= 5.216202903956794e-5`, an additional 2.8351% reduction from the
SplitOutNorm base.

## Fixed training contract

- dataset:
  `/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
- rows: train 12,081; validation 64; test 0
- RAW Fock-matrix target; no scale/shift
- 20 epochs; seed 44
- two data-parallel GPUs
- micro-batch 5 per rank; gradient accumulation 2; effective batch 20
- `distribute_graphs: false`
- MatrixMuon plus auxiliary AdamW
- W&B: `kaist-korea/maloq-nablaDFT`, online, every 10 optimizer steps
- display name:
  `NablaDFT | NTE-64/2 | MatrixMuon+AuxAdamW | RAW | QHFcond |
  EdgePre+Edge1+QHFProj+SplitOutNorm+Edge1AtomDirect | V1`

## Commands

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_nabladft_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct_2gpu.sh prepare
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_nabladft_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_nabladft_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_nabladft_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct_2gpu.sh full 0,1
```

Successful smoke artifacts are removed. Failed smoke evidence and full outputs
remain under:

`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-nte64-edgepre-edge1-qhfproj-splitoutnorm-edge1atom-v1-2gpu-eb20-mb5-ga2-<scope>-<timestamp>-<pid>/`

The durable queue manifest is
`queue_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct.yaml`.
It hashes the launcher and model config, requests two GPUs on one host, and
permits either SC26 server. The launcher passed `bash -n` and `prepare`;
the queue manifest schema, single-`{gpus}` contract, config-only semantic
delta, and `git diff --check` passed. Runner validation and the full-model
two-GPU smoke wait for the concurrent core option work to be finalized.
No full job has been enqueued.
