# Project-local automatically generated scripts

Codex and other agents place MALOQ-specific helper, inspection, conversion,
migration, and automation scripts in this directory. A tool belongs here when
it imports MALOQ, implements MALOQ's formats or conventions, or operates this
checkout's configs, tests, experiments, checkpoints, or outputs.

Project-independent server, storage, download, transfer, and monitoring tools
belong under the workspace-global directory instead:

`/dataset/seongsu/shared-home/workspace/_global_auto_script/`

Their output data belongs under:

`/dataset/seongsu/shared-home/workspace/_global_auto_script_output/`

Experiment launchers do not belong here. When the user asks for an experiment
script, place it under `_my_script/experiment/YYYY-MM-DD/` instead.

Large data, checkpoints, logs, and caches must remain outside this directory.

The OMol electrolyte corrected-v2 processor, immutable source snapshot, full-view verifier, and resumable phase launcher live in `_auto_script/omol_electrolyte_process/`. Use its `run_corrected_full_v2.sh status` command for current progress.
