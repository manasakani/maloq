# MALOQ-QH9 comparison

This experiment adapts the backbone recipe from the archived `ml-dft` run
`qh9_b3lyp5_maloq0713_nte_qhflow3_parity_bounded_degree_layerscale_s64_full_seed44`
to the native SC26 MALOQ training contract.

The `maloq-nte` lane carries over node-then-edge message passing, three node
and two edge blocks, Grid atomwise FFNs, sigmoid gates, edge envelope/scalar
modulation, a 128-to-64 equivariant readout, and bounded per-branch/per-degree
LayerScale initialized at `1/64`. The baseline retains interleaved 3+3 message
passing, Spectral atomwise FFNs, tanh gates, 128 output channels, and no
LayerScale. Both lanes use the same native MALOQ coupled-irrep head, label loss,
QH9Stable order, Muon/AdamW recipe, global gradient clipping at 1.0, and seed
44, so this is a controlled native MALOQ model comparison rather than a claim
of checkpoint compatibility with the `ml-dft` dynamic expansion head.
The coupled-irrep head keeps native MALOQ's default `reduce_edge: false` in both
lanes.

The separate `qhflow3` lane ports the active `QHFlow3CleanFeatures` trunk and
removes its external expansion head. It converts the native QM7 loader's real
e3nn-ordered overlap matrices to padded per-atom QHFlow3 blocks, supplies a
zero Hamiltonian state (the converted QH9Stable database has no independent
initial Hamiltonian, and using the target would leak the answer), then feeds
the QHFlow3 node/pair latents to MALOQ's coupled-irrep head. Its config is
`qhflow3_maloq_head_qh9stable.yaml`; it has no runtime dependency on the
`ml-dft` checkout, Lightning, or `torch_scatter`.

2026-07-22 equivariance update: the QHFlow3 lane uses an explicit 48x48 SO(3)
grid. The original lmax=4 default (10x11) produced substantial pair-feature
aliasing under general rotations because its pair GridAtomwise output has no
residual branch. The 48x48 setting preserves the QHFlow3 operations and brings
production-size node and pair covariance below the float32 `1e-4` regression
tolerance. Its output directory is separated with `equivariant-grid48` because
the resulting checkpoints are a materially different configuration. The grid
FFN is checkpointed in chunks of 512 nodes/pairs to bound the larger 48x48
activation grid during batch-32 training; chunking is mathematically identical
and does not change the equivariance result.

The YAML value `dataset_name: QM7` is deliberate: converted QH9Stable records
are stored in the original MALOQ QM7 matrix convention and consumed through
that unchanged loader path. The runner separately requires QH9Stable metadata,
the def2-SVP basis, and the expected Hamiltonian/overlap storage conventions.

Smoke all three matched lanes:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  _my_script/experiment/2026-07-21/compare_maloq_qh9.py --smoke --variant all
```

Use `--variant maloq`, `--variant maloq-nte`, or `--variant qhflow3` for one
model alone. `--variant all` runs the three in that order and writes both
`comparison.json` and `comparison.csv`.

Add `--full-size-smoke` to retain the production 128-channel trunk and
candidate 64-channel readout while still using only the ordered 2/1/1 sample.
Successful smoke runs are temporary by default and delete their local output
directory after validation. Add `--keep-smoke-output` only when artifacts are
needed for debugging. A failed smoke keeps its partial output. Full runs retain
their checkpoints normally.

The canonical three-directory layout and CSV writer were verified at
`outputs/qh9stable-three-model-comparison-naming-smoke-20260722-055314/`.

For the full official split, first produce the converted database at
`/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db`, then run
the same command without `--smoke`. Full training is not launched automatically.
