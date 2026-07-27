# MALOQ experimental features

This namespace is the incubation area for research features that have not been
accepted into canonical MALOQ. It is intentionally packaged with MALOQ but is
never imported by the normal CLI, `TrainingWorkflow`, or canonical package
initializers.

## Dependency contract

```text
maloq.experimental.<feature>  --->  canonical maloq
canonical maloq               -X->  maloq.experimental
```

An experiment runner must import exactly the feature it selects. Do not add
automatic discovery or import-time loading of every feature. If a canonical
extension seam is necessary, make it feature-neutral, keep its default behavior
unchanged, and cover that behavior with a regression test.

## Layout

```text
src/maloq/experimental/<semantic_feature_slug>/
    __init__.py
    FEATURE.md
    config.py          # feature-owned schema/defaults, when needed
    model.py           # or layer.py, head.py, dataset.py
    workflow.py        # optional explicit adapter around canonical workflow

tests/experimental/<semantic_feature_slug>/
    test_*.py

_my_script/experiment/YYYY-MM-DD/
    <config>.yaml
    <launcher>.sh
    README.md
```

Use a semantic lowercase snake-case slug, for example
`nte_edge2_atom_direct`. Names such as `v2`, `new`, and `test` do not explain
the hypothesis and are not accepted as feature directory names.

Standalone inspection, migration, and automation scripts remain under
`_auto_script/`; they are not importable training implementations.

## Starting a feature

1. Copy `FEATURE_TEMPLATE.md` to
   `src/maloq/experimental/<slug>/FEATURE.md`.
2. Put all feature-specific defaults, validation, and implementation in that
   directory. Prefer an explicit wrapper or adapter over conditionals in core.
3. Add focused tests below `tests/experimental/<slug>/`.
4. Import the feature explicitly from its dated experiment runner. A feature
   must not activate merely because MALOQ was imported.
5. Include the slug in run names, W&B tags, output directories, and checkpoint
   metadata.
6. Update `FEATURE.md` as the feature moves through
   `draft -> smoke -> validated -> promoted` or `abandoned`.

Do not extract shared experimental utilities after a single use. Keep code
feature-local until at least two validated consumers demonstrate the same
stable abstraction.

## Promotion

Promotion is a deliberate move, not a re-export from canonical code back into
this namespace. Before promotion, record:

- exact baseline and candidate commit/config/data-split provenance;
- shape, forward/backward, dtype, AO-convention, and equivariance checks that
  apply to the feature;
- CPU import and a CUDA train-plus-validation smoke;
- checkpoint save, reload, and deterministic-seed behavior;
- DDP behavior when the feature claims distributed support;
- matched parameter count, peak memory, throughput, and quality results;
- multi-seed evidence when making a performance claim;
- optional dependency and license impact.

After approval, move the selected behavior into one coherent canonical
implementation, add canonical documentation and tests, and remove ablation-only
knobs and duplicate branches. Keep a thin experimental compatibility alias only
when an existing checkpoint truly requires it, and make the alias point toward
canonical code rather than the reverse.

## Grandfathered work

Pre-policy experiments and frozen source snapshots are not moved while jobs are
running. Migrate them in a separate, reviewable change after their runs finish.
New variants must follow this contract immediately.
