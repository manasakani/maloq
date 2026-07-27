# MALOQ-NTE-V2

`backbone_type: maloq_nte_v2` selects the fixed architecture promoted from
the NablaDFT `Edge2+InitEnv+Atom2Direct` experiment. The implementation lives
in:

- `src/maloq/helm/esen_osh_v2.py`: input embedding, message-passing schedule,
  output normalization, and projection.
- `src/maloq/helm/esen_block_v2.py`: `NodeBlock`,
  `InitialEdgeBlock`, and `EdgeRefinementBlock`.

Neither V2 file subclasses the legacy `eSEN_Backbone` or `eSEN_Block`.

## Canonical and experimental policy

`backbone_type: esen` selects canonical MALOQ. It contains the original eSEN
implementation plus the feature-neutral `element_only` input seam; it does not
interpret NTE/QHFlow3 composition selectors.

The selected core rerun YAMLs use the feature-owned config and explicit
`maloq.experimental.nte_qhflow3_composition.workflow` entry point. Arbitrary
historical selector combinations and their fixed-workflow resume signatures
are intentionally unsupported.

MALOQ-NTE-V2 is a new fixed architecture boundary, not a compatibility layer.
New comparisons should start from a fresh seed with `TrainingWorkflowV2`; old
selector checkpoints and optimizer/scheduler states are intentionally outside
its support contract.
