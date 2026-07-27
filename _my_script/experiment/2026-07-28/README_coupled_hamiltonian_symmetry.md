# NablaDFT symmetry-reduced Hamiltonian axis

This matched three-lane experiment keeps the E3 Muon+SHIFT V2 structure axis
and changes only the matrix output parameterization. The public experiment
config keeps the legacy reduction flags off:

```yaml
reduce_node: false
reduce_node_intra: false
reduce_edge: false
```

The explicit experimental head has one fixed internal contract:

```text
node: upper orbital-pair triangle + even-L diagonal irreps
edge: reverse-pair alpha (even) + beta (odd) irreps
reconstruction: full coupled output with orbital exchange signs
```

This exactly follows the `reduce_node_intra=True` philosophy requested for
onsite blocks: antisymmetric diagonal odd-`L` channels are not parameterized.
For edges, reverse directions are combined before the output map, then
reconstructed with exchange parity. Dense AO transpose and post-hoc averaging
are not used in training.

## Matched contract

- lanes: MALOQ-E3, NTEV2-E3, QHFlow3-E3-10x11
- QHFlow3 grid: `qhflow3_grid_resolution: null`, giving the exact eSEN default `10x11` grid for `lmax=4` (latitude 10, longitude 11)
- head and optimizer: three Muon-visible reduced matrices plus auxiliary AdamW
- normalization: SHIFT-only
- fixed ordered split: 12,081 train / 64 validation / 0 test
- two GPUs, micro-batch 5 per rank, accumulation 2, effective batch 20
- 20 epochs, seed 44, float32
- W&B project: `kaist-korea/MALOQ-nablaDFT-v2`
- display suffix: `Node+EdgeIrrepSym | V2`

## Files

- feature: `src/maloq/experimental/coupled_hamiltonian_symmetry/`
- runner: `_my_script/experiment/2026-07-28/run_nabladft_coupled_hamiltonian_symmetry.py`
- launcher: `_my_script/experiment/2026-07-28/05_nabladft_coupled_hamiltonian_symmetry_2gpu.sh`
- queue manifest: `_my_script/experiment/2026-07-28/queue_nabladft_coupled_hamiltonian_symmetry.yaml`

## Commands

Validation:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/05_nabladft_coupled_hamiltonian_symmetry_2gpu.sh validate all
```

Disposable two-GPU smokes must use an available pair other than server 2 GPUs
4-7. Full jobs are pinned to server 1 by the queue manifest.

Queue IDs:

- `nabla-v2-qhflow3-e3-10x11-muon-shift-node-edge-irrep-sym-20260728b`
- `nabla-v2-ntev2-e3-muon-shift-node-edge-irrep-sym-20260728a`
- `nabla-v2-maloq-e3-muon-shift-node-edge-irrep-sym-20260728a`

Status: CPU verification and all three CUDA/DDP smokes complete; durable enqueue is ready.
