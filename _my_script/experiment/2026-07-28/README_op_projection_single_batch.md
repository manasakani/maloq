# Operator-projection single-batch learning smoke

This experiment checks whether the node-only MALOQ-NTE-V2 branch can learn a
real Hamiltonian through a differentiable `probe -> H_hat @ probe` callback.
The predicted path streams onsite and directed pair AO blocks in chunks; it
does not construct a molecule-scale matrix.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  _my_script/experiment/2026-07-28/run_op_projection_single_batch_overfit.py
```

Validated setting: NablaDFT `train_2k.db` row 189, 25 atoms, 291 AO, complete
directed graph with 600 pair blocks, C=8, edge chunk=128, two newly sampled
Rademacher probes per step, eight fixed held-out probes, AdamW at 1e-2, 120
head-only steps on CPU. The node backbone is frozen during overfit to isolate
the new operator head, followed by one end-to-end backward check.

## Result

- Held-out squared relative action error: `1.0037879 -> 0.7546901`
- Reduction: `24.8158%`
- Packed AO matvec oracle relative action error: `1.06e-14`
- Coupled-to-packed vs canonical `get_H`: relative error `1.36e-15`
- End-to-end backbone gradient: norm `4.1645` across 59 tensors
- Runtime: `46.54 s`, or `0.388 s/step`, CPU
- Largest transient pair chunk: 128 of 600 edges

Artifacts are in `outputs/op_projection_single_batch_row189/`: `metrics.json`
contains the full trace and `checkpoint.pt` contains both frozen-backbone and
trained-head states.

## Interpretation

This passes a learning/connectivity smoke: improvement transfers to probes that
were never optimized directly, AO block orientation matches the dense oracle,
and gradients reach the real backbone. It is not yet a quality or scaling
claim. The test uses one molecule, a random frozen backbone, and a dense target
for oracle actions plus one non-training packing diagnostic. It does not
enforce Hermiticity or density-matrix constraints.
