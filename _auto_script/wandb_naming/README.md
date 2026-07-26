# SC26 W&B naming

The scripts in this directory normalize W&B metadata without renaming or
deleting local `outputs/` directories. Those directories remain immutable
provenance paths.

## Canonical fields

- Display name: `Dataset | objective | model | head | version`
- Completed comparison groups: one group per dataset and objective
- Failed attempts: a separate `*-failed` group with a cause and attempt in the
  display name
- Tags: dataset, scope, status, objective, target, model, head, optimizer,
  batch/accumulation, seed, and version
- Notes: absolute local artifact path and any audited completion/failure detail

NablaDFT Muon runs use a compact optimizer/head policy label:

- `MatMuon+SemHead`: trainable `ndim >= 2` parameters use Muon and the
  standard semantic matrix head is used.
- `MatMuon+SGHead`: the same matrix-Muon policy with the explicitly grouped
  semantic-global head route.
- `MatMuon+SGHead+GateMuon`: the semantic-global head route plus a
  Muon-materialized scalar/gate projection; scalar and output biases remain
  on auxiliary AdamW.

The auxiliary optimizer is intentionally omitted from the display name. Its
exact routing remains queryable through `optimizer:*`, `muon-routing:*`,
`aux-optimizer:*`, and `head-routing:*` tags.

## NablaDFT normalization lanes

NablaDFT run names use normalization operations instead of the ambiguous
historical `SS0`/`SS1` labels:

- `RAW`: no target transformation
- `SHIFT`: subtract the element-specific `l=0` node mean only
- `SHIFT+STD`: subtract that mean and divide by the standard deviation

Audit the selected runs without changing W&B:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/wandb_naming/rename_nabladft_normalization.py
```

Apply the audited names, tags, and notes:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/wandb_naming/rename_nabladft_normalization.py \
  --apply
```

Local output directories are not renamed.

## NablaDFT Muon policy labels

Audit the existing runs without changing W&B:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/wandb_naming/rename_nabladft_muon_policy.py
```

Apply the compact policy labels and routing tags:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/wandb_naming/rename_nabladft_muon_policy.py \
  --apply
```

This metadata-only migration does not rename local output directories or
change checkpoints.

## QH9Stable

Run a read-only audit first:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/wandb_naming/update_qh9stable_runs.py
```

Apply the audited changes:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/wandb_naming/update_qh9stable_runs.py \
  --apply
```

The QH9Stable updater refuses to modify any run whose W&B state is `running`.
Its current inventory contains 22 inactive runs and two explicitly excluded
QHFlow3 runs. No run deletion is implemented.

The completed comparison names are:

- `QH9Stable | DΔ | MALOQ | Native | V1`
- `QH9Stable | DΔ | NTE-64/2 | Native | V1`
- `QH9Stable | HΔ | MALOQ | Native | V1`
- `QH9Stable | HΔ | NTE-64/2 | Native | V1`

They are grouped under `qh9stable-ddelta-native` and
`qh9stable-hdelta-native`. The 18 audited failed attempts are grouped under
`qh9stable-failed`. Runs `ef0jlqcw` and `o64mfwkd` remain untouched while W&B
reports them as running.
