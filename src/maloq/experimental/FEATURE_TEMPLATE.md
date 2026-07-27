# `<semantic_feature_slug>`

- Status: `draft`
- Owner:
- Created: `YYYY-MM-DD`
- Intended promotion target:

## Hypothesis

State one falsifiable architectural or training hypothesis.

## Baseline

- Git commit:
- Config:
- Dataset and exact split:
- Seed(s):
- Effective batch and optimizer:
- Reference output/W&B run:

## Entry point

- Import:
- Experimental config namespace:
- Resolved component/profile ID:
- Optional dependencies:

The dated experiment runner must import this entry point explicitly. Canonical
MALOQ must not import it.

## Checkpoint contract

- Architecture/schema version:
- Compatible checkpoints:
- State-dict migration:
- Deterministic initialization requirements:

## Verification

- [ ] CPU import
- [ ] Config validation
- [ ] Shape and forward/backward
- [ ] Dtype/device coverage
- [ ] Equivariance, when applicable
- [ ] AO/basis convention, when applicable
- [ ] CUDA train step
- [ ] CUDA validation step
- [ ] Checkpoint save/reload
- [ ] DDP smoke, when supported
- [ ] Matched parameter/memory/throughput comparison
- [ ] Matched quality comparison

## Evidence

| Date | Config | Commit or source snapshot | Output/W&B | Result |
|---|---|---|---|---|

## Known limitations

-

## Promotion decision

Record the accepted behavior, rejected ablations, destination canonical path,
required migration notes, and approval. If abandoned, explain why and state
whether the implementation and artifacts may be removed.
