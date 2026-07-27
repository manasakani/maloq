# NablaDFT 10k training and 2k-conformer test preparation

This experiment uses the official native NablaDFT v2 Hamiltonian databases:

- train: `train_10k.db` (`dataset_train_medium`)
- held-out validation: the final 64 rows of `train_10k.db`
- benchmark test artifact prepared for later checkpoint evaluation:
  `test_2k_conformers.db` (`dataset_test_conformations_tiny`)

The launcher reads the actual SQLite metadata row count and sets
`num_train = rows - 64`; it does not assume the 2k database's 12,145 rows.
All three variants keep the existing native NablaDFT loader and raw Fock
matrix objective. Full runs use two data-parallel GPUs, micro-batch 5 per
rank, gradient accumulation 2, effective batch 20, 20 epochs, Muon plus the
MALOQ Muon head, and W&B `kaist-korea/maloq-nablaDFT` every 10 optimizer
steps.

Environment:

`/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26`

Outputs:

`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-10k-<variant>-muon-head-2gpu-eb20-mb5-ga2-<scope>-<timestamp>-<pid>/`

Exact commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh prepare

/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh validate maloq 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh validate maloq-nte 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh validate qhflow3 0,1

/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh smoke maloq 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh smoke maloq-nte 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh smoke qhflow3 0,1

/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh full maloq 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh full maloq-nte 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_nabladft_10k_2gpu.sh full qhflow3 0,1
```

`validate` checks both downloaded DB schemas and model configuration without
training. `smoke` runs one epoch on 20 train and 20 validation rows and
removes successful temporary artifacts; failed evidence is retained. No full
training is launched automatically.
