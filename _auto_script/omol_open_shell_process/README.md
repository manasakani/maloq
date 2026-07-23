# OMol25 open-shell MALOQ processing

This converter turns the restored strict metal-organic, open-shell OMol25
sources into restartable ASE database shards. It preserves the official
`train`, `val`, and `test` splits and stores both supervised matrix targets:

- `fock_matrix[alpha, beta, nao, nao]`, recovered from `orca.out`;
- `density_matrix[alpha, beta, nao, nao]`, reconstructed as
  `0.5 * (orca.scfp +/- orca.scfr)`.

The raw fp64 downloads remain unchanged. The default processed databases use
float32 for the current MALOQ CUDA label kernel. Each completed shard is
atomically renamed and has a report in `_state/`, so rerunning the same command
skips valid completed shards and retries incomplete/failed shards.

Default Quasar paths:

```text
source: /home1/irteam/data-vol1/data/omol25/open_shell_restore
output: /home1/irteam/data-vol1/data/omol25/open_shell_maloq_ase
```

Smoke and full commands:

```bash
python process_omol_open_shell_to_ase.py \
  --output-root /home1/irteam/data-vol1/data/omol25/open_shell_maloq_ase_smoke \
  --limit-per-split 1 --shard-size 1 --workers 2

python process_omol_open_shell_to_ase.py \
  --shard-size 32 --workers 4

python process_omol_open_shell_to_ase.py --status
```

For training, point `dbpath` at the processed `train/` directory and
`dbpath_val` at `val/`, use `dataset_name: omol`, `open_shell: true`, and select
either `loss_target: density_matrix` or `loss_target: fock_matrix`. Rows carry
`matrix_storage_convention=orca_real_spherical`; the SC26 loader converts this
exactly once to the MALOQ/e3nn convention.
