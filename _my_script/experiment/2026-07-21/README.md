# 2026-07-21 experiments

Reserved for experiment launchers created on 2026-07-21 (Asia/Seoul).

For each experiment, record its purpose, exact command, QH9Stable split,
GPU selection, Python environment, output path, and current status here.

## QH9Stable native-loader preprocessing

- Raw source: `/dataset/seongsu/shared-home/data/QH9_rawdata/QH9Stable.db`
- Target format: MALOQ `ASEAtomsData`, consumed with `dataset_name="QM7"`
- Split tested: official Stable/random, ordered train → val → test
- Converted smoke data used for training: 2 train / 1 val / 1 test
- Original loader status: passed; finite node/edge labels
- Matrix → label → matrix maximum absolute error: `3.48e-7`
- One-epoch CUDA model smoke: passed on GPU 0 with finite train and validation
  losses; the launcher artifacts are in
  `outputs/qh9Stable_script_smoke_20260721/`.

## Commands

Use the project interpreter:

```bash
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
```

Run the verified smoke experiment:

```bash
$PY _my_script/experiment/2026-07-21/run_qh9stable.py --smoke --gpu 0
```

After creating the complete ordered Stable/random database, start a full run
with an explicit collision-resistant output name:

```bash
$PY _my_script/experiment/2026-07-21/run_qh9stable.py \
  --dbpath /dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db \
  --output-folder outputs/qh9Stable_random_full_20260721-01 \
  --gpu 0
```

The launcher validates the database's Stable marker, completion marker, and
ordered split counts before initializing training. It keeps `shuffle=False`
and `distribute_graphs=False` so the converted train → val → test boundaries
remain intact.

The original QM7 loader materializes the selected matrices and graphs in
memory. Before the full 130,831-row run, create and train a bounded ordered
subset, for example:

```bash
$PY _auto_script/qh9_raw_to_maloq/process_qh9_raw_to_maloq_ase.py \
  --input-db /dataset/seongsu/shared-home/data/QH9_rawdata/QH9Stable.db \
  --split-file /dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed/processed_QH9Stable_random_12.json \
  --subset-limit train=1000 --subset-limit val=100 --subset-limit test=100 \
  --output-db /dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random_1000_100_100.db

$PY _my_script/experiment/2026-07-21/run_qh9stable.py \
  --dbpath /dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random_1000_100_100.db \
  --num-train 1000 --num-val 100 --num-test 100 \
  --output-folder outputs/qh9Stable_random_1000_100_100_20260721-01 \
  --gpu 0
```

QH9Dynamic is out of scope.
