# OMol25 electrolyte MALOQ dataset migration

This SC26-controlled helper pulls the completed electrolyte density-matrix
snapshot that already exists on Quasar. Quasar is used only as a read-only
source; orchestration, retry state, storage, and verification remain on SC26.

Destination:

`/dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb`

The completed index selects only successful shards from the historical source,
excluding incomplete directories:

- train: 43,252 samples / 10,813 shards;
- validation: 5,028 samples / 1,257 shards;
- test: 9,620 samples / 2,405 shards;
- selected LMDB and summary bytes: 3,161,180,105,435.

The matrices use def2-TZVPD, float32 storage, MALOQ/e3nn ordering, a target
density matrix, overlap, and trace-corrected SAD initial density. The transfer
copies the immutable completed index and selection manifests, seeds one shard
per split, resumes partial files after interruptions, and validates exact
split counts, bytes, manifest/index hashes, and a readable LMDB sample.

```bash
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_electrolyte_transfer/sync_completed_electrolyte.sh status
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_electrolyte_transfer/sync_completed_electrolyte.sh sync
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_electrolyte_transfer/sync_completed_electrolyte.sh verify
```
