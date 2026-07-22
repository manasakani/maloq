# QH9Stable raw → MALOQ ASE preprocessing

`process_qh9_raw_to_maloq_ase.py` converts the original QH9Stable SQLite file into
the SchNetPack-style ASE database schema consumed by MALOQ's unchanged `QM7`
loader.

The output always contains `energy`, `forces`, `hamiltonian`, and `overlap`.
With `--initial-matrix-lmdb`, it additionally joins `initial_hamiltonian` and
`initial_density_matrix` by the immutable QH9 source index. The resulting DB is
Hamiltonian-delta-only: H0 is the residual baseline and D0 is retained only as
QHFlow3 auxiliary conditioning. Raw QH9Stable does not include forces or
energy, so both are explicit zero placeholders.

The raw Hamiltonian is PySCF real-spherical def2-SVP. It is stored in the
pre-conversion convention expected by the original QM7 loader. The overlap is
recomputed from the geometry using PySCF and stored in MALOQ/e3nn convention,
because the original loader converts only the Hamiltonian.

## Small Stable conversion

Run from the repository root:

```bash
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python

$PY _auto_script/qh9_raw_to_maloq/process_qh9_raw_to_maloq_ase.py \
  --input-db /dataset/seongsu/shared-home/data/QH9_rawdata/QH9Stable.db \
  --initial-matrix-lmdb /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/data.lmdb \
  --split-file /dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed/processed_QH9Stable_random_12.json \
  --subset-limit train=2 --subset-limit val=1 --subset-limit test=1 \
  --output-db /dataset_tmp/qh9_maloq_ase/QH9Stable_random_smoke.db
```

This output is ordered as train, validation, then test, so use
`num_train=2`, `num_val=1`, and `num_test=1` with `shuffle=False`.

## Full Stable/random conversion

Omit the limits:

```bash
$PY _auto_script/qh9_raw_to_maloq/process_qh9_raw_to_maloq_ase.py \
  --input-db /dataset/seongsu/shared-home/data/QH9_rawdata/QH9Stable.db \
  --initial-matrix-lmdb /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/data.lmdb \
  --split-file /dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed/processed_QH9Stable_random_12.json \
  --output-db /dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db
```

The current original QM7 loader materializes every selected matrix and graph in
memory. Validate the format with the 2/1/1 database first, then use a bounded
training subset before attempting the complete 104,664-molecule training split.
The converted schema can remain unchanged if the loader is later made lazy.

The converter never overwrites an existing target. It writes a `.partial.db`,
validates it with `ASEAtomsData`, marks metadata as complete, and only then
renames it to the requested output path.

QH9Dynamic is intentionally rejected by this converter.
