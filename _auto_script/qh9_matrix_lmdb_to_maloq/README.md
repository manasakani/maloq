# QH9Stable matrix LMDB → MALOQ ASE

This converter reads the existing `qh9_b3lyp5_with_density` split-key LMDB
without modifying it and creates a density-target ASE database for MALOQ's
native `QM7` loader. It preserves the official random split membership and
stores final density, initial density, initial Hamiltonian, and overlap.

The D/D0/H0 matrices are staged in the pre-`orca_to_e3nn` convention expected
by the loader. H0 is conditioning-only for QHFlow3; the database advertises
only density and density-delta targets. Overlap is stored directly in
MALOQ/e3nn convention. The loader then uses the original
`Fock_Targets`/`matrix2labels` path, so labels retain the configured graph
cutoff and edge ordering instead of being frozen in the DB.

Smoke conversion:

```bash
cd /dataset/seongsu/shared-home/workspace/project
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
mkdir -p /dataset_tmp/qh9_matrix_maloq_ase
"${PY}" _auto_script/qh9_matrix_lmdb_to_maloq/process_qh9_matrix_lmdb_to_maloq_ase.py \
  --input-lmdb /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/data.lmdb \
  --split-file /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/QH9Stable_random_split.json \
  --output-db /dataset_tmp/qh9_matrix_maloq_ase/QH9StableMatrices_random_2_1_1.db \
  --subset-limit train=2 \
  --subset-limit val=1 \
  --subset-limit test=1
```

Full conversion:

```bash
cd /dataset/seongsu/shared-home/workspace/project
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
"${PY}" _auto_script/qh9_matrix_lmdb_to_maloq/process_qh9_matrix_lmdb_to_maloq_ase.py \
  --input-lmdb /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/data.lmdb \
  --split-file /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/QH9Stable_random_split.json \
  --output-db /dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9StableMatrices_random.db \
  --matrix-dtype float64 \
  --progress-every 100
```

The converter refuses to overwrite an existing final or partial database and
only atomically finalizes the output after AO-order and source round-trip
validation succeeds.

For a legacy combined conversion that already contains H/H0/D/D0/S, change
its advertised contract without rewriting the large row blobs:

```bash
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
"${PY}" _auto_script/qh9_matrix_lmdb_to_maloq/set_qh9stable_density_profile.py \
  --db /dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9StableMatrices_random.db \
  --audit-dir outputs/qh9stable-density-profile-20260722 \
  --apply
```
