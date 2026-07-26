# OMol25 open-shell processed-data synchronization

This helper is launched and controlled on SC26. It pulls the already completed
MALOQ-ready OMol25 open-shell ASE dataset from the Quasar Lustre mount into the
SC26 shared dataset volume. Quasar is only a read-only source for this one-time
migration; this helper does not submit a download or processing job there.

Source:

`/home1/irteam/data-vol1/data/omol25/open_shell_maloq_ase`

Destination:

`/dataset/seongsu/shared-home/datasets/omol25_open_shell_maloq_ase`

The source contains 30,815 strict metal-organic open-shell rows in 965 shards:
30,215 train rows in 945 shards, 270 validation rows in 9 shards, and 330 test
rows in 11 shards. Both alpha/beta Fock and density targets are stored as
float32 in ORCA real-spherical convention.

The synchronization is resumable and does not copy the 1.7 TB raw restore
tree. It copies the 2,005,560,193,024-byte processed database plus its manifest.
Partial files are retained in `.rsync-partial`, SSH keepalives are enabled, and
the helper retries interrupted transfers before validating exact shard counts,
database bytes, and the manifest checksum.

```bash
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_open_shell_transfer/sync_processed_open_shell.sh status
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_open_shell_transfer/sync_processed_open_shell.sh sync
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_open_shell_transfer/sync_processed_open_shell.sh verify
```
