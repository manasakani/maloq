# SC26-seongsu Codex rules

These rules apply to the canonical SC26-seongsu project at
`/dataset/seongsu/shared-home/workspace/project`.

## Script ownership and placement

- Put new agent-generated helper, migration, inspection, conversion, and automation scripts under `_auto_script/`.
- Put a script under `_my_script/` only when the user explicitly identifies it as an experiment script or asks for it there.
- Organize experiment scripts as `_my_script/experiment/YYYY-MM-DD/`, using the `Asia/Seoul` calendar date. Create the dated directory when it does not exist.
- If an experiment spans several days, keep its original dated directory and record later changes in that directory's README or notes instead of moving it.
- Treat existing files under `_my_script/` as user-owned. Do not rewrite, move, or delete them unless the user explicitly asks.
- Existing production modules stay in their canonical directories (`helm/`, `fock_utils/`, `dataset_utils/`, `train_utils/`, and `tests/`). The `_auto_script/` rule applies to newly created standalone scripts, not to necessary edits of existing source files.
- Do not add new one-off scripts at the repository root or in `scripts/`.

## Experiment hygiene

- Give every dated experiment directory a short `README.md` stating the purpose, exact command, dataset/split, GPU selection, environment, output path, and status.
- Keep configs and launchers with the corresponding dated experiment. Put model checkpoints, logs, metrics, W&B files, and other run artifacts under `outputs/<run-name>/`; put datasets outside the repository and reference them by path.
- Use explicit, collision-resistant output directory names. Never reuse a checkpoint directory for a materially different configuration.
- Do not create new root-level `outputs_*` directories. Relative model-output paths are normalized below the project-root `outputs/` directory by `TrainingWorkflow`.
- Start with a tiny smoke run and verify one train and validation step before a full QH9 run.
- Do not start a long-running or multi-GPU training job unless the user explicitly asks to launch it.

## QH9 safety and correctness

- The current QH9 effort targets QH9Stable only. Do not add or run QH9Dynamic processing unless the user explicitly expands the scope.
- Preserve the official QH9 train/validation/test split files. Do not generate a new random split unless the user explicitly requests an ablation.
- Keep the QH9 raw SQLite databases read-only. Write converted ASE databases outside this repository and record their source path and split provenance.
- QH9 matrices use PySCF real-spherical def2-SVP AO ordering. When targeting the unchanged QM7 loader, store Hamiltonians in its pre-`orca_to_e3nn` input convention and overlaps in MALOQ/e3nn convention; never apply either transform twice.
- Compute any scale/shift statistics from the training split only, and save provenance with the resulting artifact.
- Keep `distribute_graphs=False` until the QH9 streaming path is explicitly implemented and tested for distributed graphs.

## Verification

- Use the project interpreter explicitly:
  `/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python`.
- For QH9 changes, verify dataset schema and split sizes, AO-order round trips, matrix-label reconstruction, and a one-epoch CUDA smoke run before recommending full training.
- Preserve the current SC26 dtype fixes and non-finite-loss checks when porting code from older sibling checkouts.
