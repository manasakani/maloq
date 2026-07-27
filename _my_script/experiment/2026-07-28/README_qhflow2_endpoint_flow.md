# Full-matrix endpoint flow on NablaDFT

This dated artifact exercises the isolated
`maloq.experimental.flow_matching` path through the real canonical
`TrainingWorkflowV2Fixed` data/model/checkpoint loop. It is not a synthetic
velocity-contract runner.

Architecture version 2 flows both node/diagonal and directed-edge/off-diagonal
Hamiltonian blocks. A node and all directed edges in the same molecular graph
share one sampled time:

```text
Hnode,0, Hedge,0 ~ coupled-irrep Gaussian(0, 0.1^2 I), masked by full irreps
t ~ Uniform(0.01, 0.99), one scalar per molecular graph
Hnode,t = (1 - t) Hnode,0 + t Hnode,1
Hedge,t = (1 - t) Hedge,0 + t Hedge,1
network target = clean node and edge endpoints
loss = 10 * masked coupled Frobenius MSE
```

Sampling uses three fixed joint Euler steps and derives both velocities only at
inference: `v=(Hhat1-Ht)/(1-t)`. The same graph time is used for the node and
edge updates at every step.
`EndpointFlowTrainer.sample_batch(...)` is the feature-local inference entry
that connects this sampler to the trained QHFlow3 backbone and MALOQ head.

`e3TensorDecomp.get_H()` returns shell-pair-packed AO blocks rather than a
row-major square matrix. The feature therefore uses an explicit reversible
codec

```text
coupled irreps <-> shell-pair-packed AO <-> padded dense AO
```

instead of reshaping the packed tensor. Node states are projected with
`(Hii + Hii^T)/2`; directed edge pairs are projected with
`(Hij + Hji^T)/2`. Reverse-edge pairing is required to be an involution, and
the projection is applied as part of the full-Hamiltonian flow contract.

The edge state is mapped by an SO(3)-equivariant irrep linear and added to both
the matching QHFlow3 edge embedding and a degree-normalized incident-node
aggregate. Canonical MALOQ target irreps use even parity labels for every
degree, while the QHFlow3 spherical layout uses degree parity. The wrapper
relabels parity only for this projection; proper-rotation matrices are
unchanged. Consequently this is an SO(3) contract, not an O(3)/reflection
claim.

The matched candidate retains NablaDFT train/validation `12081/64`, seed 44,
20 epochs, QHFlow3 node depth 3, edge depth 2, 64 output channels, Muon head,
Muon+AuxAdamW optimizer, batch 5 with gradient accumulation 2, and RAW targets.
The working `full_matrix_endpoint_flow_v1` profile is direct full-H prediction
(`delta_learning=false`); QHFlow2's residual `H-Hinit` parameterization is not
ported.

## Commands

Read-only typed config and inheritance validation:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/90_qhflow2_endpoint_flow_nabladft_2gpu.sh validate
```

Optional bounded real-data smoke (2 GPUs, 20 train + 20 validation molecules,
one epoch, W&B disabled):

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/90_qhflow2_endpoint_flow_nabladft_2gpu.sh smoke
```

There is deliberately no `full` or queue mode. No CUDA, DDP, checkpoint-resume,
or scientific training run has been executed for architecture version 2.

## Files

- `qhflow2_endpoint_flow_nabladft.yaml`: strict architecture-v2 candidate config
- `run_qhflow2_endpoint_flow_nabladft.py`: validation + inherited workflow
- `90_qhflow2_endpoint_flow_nabladft_2gpu.sh`: guarded validate/smoke launcher

Failed or successful smoke outputs are retained under
`/dataset/seongsu/shared-home/workspace/project/outputs/` for inspection.
