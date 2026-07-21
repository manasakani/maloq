# MALOQ-QH9 comparison

This experiment adapts the backbone recipe from the archived `ml-dft` run
`qh9_b3lyp5_maloq0713_nte_qhflow3_parity_bounded_degree_layerscale_s64_full_seed44`
to the native SC26 MALOQ training contract.

The `maloq-qh9` lane carries over node-then-edge message passing, three node
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

The YAML value `dataset_name: QM7` is deliberate: converted QH9Stable records
are stored in the original MALOQ QM7 matrix convention and consumed through
that unchanged loader path. The runner separately requires QH9Stable metadata,
the def2-SVP basis, and the expected Hamiltonian/overlap storage conventions.

Smoke both lanes:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  _my_script/experiment/2026-07-21/compare_maloq_qh9.py --smoke --variant both
```

Add `--full-size-smoke` to retain the production 128-channel trunk and
candidate 64-channel readout while still using only the ordered 2/1/1 sample.
Smoke runs retain configs, losses, plots, model summaries, and the comparison
JSON, but discard `.pt` files that duplicate large optimizer states. Full runs
retain their checkpoints normally.

For the full official split, first produce the converted database at
`/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db`, then run
the same command without `--smoke`. Full training is not launched automatically.
