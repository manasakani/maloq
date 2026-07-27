# NablaDFT NTEV2 OpProjection V4 Muon

This lane changes only the optimizer family relative to the active V4 AdamW
operator-projection experiment. The dataset split, model dimensions, operator
callback, two-rank MB5/GA2/effective-batch-20 geometry, scheduler, probe loss,
and streamed exact matrix metrics remain unchanged.

The optimizer uses the project-local `maloq.train_utils.optimizers.Muon`.
Every unique trainable tensor with `ndim >= 2` is routed to Muon (`lr=0.02`,
momentum `0.95`, Nesterov, five Newton–Schulz steps). Trainable scalars,
vectors, and packed 1-D weights use the same optimizer's auxiliary AdamW
path (`lr=5e-4`, betas `(0.9, 0.95)`, epsilon `1e-10`). Both groups use
weight decay `1e-4` and the same warmup-polynomial LR multiplier.

The run writes a complete `optimizer_groups.json` routing manifest and embeds
its SHA-256 plus group summaries in `resolved_config.json`. W&B retains the
canonical `optimizer/learning_rate` key for the primary Muon group and also
logs explicit `optimizer/muon_learning_rate` and
`optimizer/aux_adamw_learning_rate` keys.

The exact W&B display name is
`NablaDFT | NTEV2-OpProjection | AdamW | RAW | V4 MUON`; the comparison group
is `nabladft-op-projection-v4-seed44`. Smoke runs disable W&B.

Validation matrix MAE/MSE is evaluated on all 64 canonical validation rows for
the full run. It is streamed through restricted identity-column chunks and
does not materialize a complete predicted matrix internally. The loader still
reads the dense reference Hamiltonian and the backbone still consumes the
dense overlap input, matching V4.

Commands:

```bash
./_my_script/experiment/2026-07-28/07_nabladft_op_projection_muon_2gpu.sh validate
./_my_script/experiment/2026-07-28/07_nabladft_op_projection_muon_2gpu.sh smoke 0,1
```

The durable full job is restricted to `server-1` because server-2 GPUs 4-7
are hard-reserved and the queue allocator has no per-job GPU denylist.
