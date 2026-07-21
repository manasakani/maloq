# Model outputs

Store every training, evaluation, and inference run in its own directory below
this folder:

```text
outputs/<run-name>/
```

This includes checkpoints, copied configs, logs, metrics, plots, embeddings,
and W&B runtime files. `TrainingWorkflow` resolves relative output paths from
the repository root, converts legacy `outputs_<name>` values to
`outputs/<name>`, and rejects output paths outside this directory.

The pre-existing root-level `outputs_*` directories were moved here on
2026-07-21 with their `outputs_` prefix removed. Their internal files were not
rewritten, so historical configs and logs retain the original run provenance.
