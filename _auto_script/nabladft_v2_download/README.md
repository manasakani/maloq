# NablaDFT v2 medium-train and tiny-conformer test download

This helper downloads the two native Hamiltonian SQLite databases requested
for the next SC26 experiments:

- `train_10k.db`: official `dataset_train_medium`, 68,388,278,272 bytes.
- `test_2k_conformers.db`: official
  `dataset_test_conformations_tiny`, 3,099,738,112 bytes.

The files are written outside the repository under:

`/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases`

The downloader runs on SC26, keeps resumable `.part` files, checks the live
HTTP Content-Length and official multipart ETag before transfer, refuses
unexpected final sizes, and atomically promotes only complete files. The
verifier opens each database read-only, checks its SQLite schema and basis
metadata, and decodes the first and last matrix rows. Verification sidecars
are stored beside each database.

```bash
bash /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_v2_download/download_nabladft_v2.sh status
bash /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_v2_download/download_nabladft_v2.sh download
bash /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_v2_download/download_nabladft_v2.sh verify
```

Transfer logs and the single-writer lock are under:

`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-v2-download`

The official NablaDFT benchmark also has global `test_structures.db` and
`test_scaffolds.db` files. They are not part of this default download because
they are 1,595,255,488,512 and 1,579,001,356,288 bytes respectively. Those
multi-terabyte tests require a separate explicit transfer decision.
