# Edge1AtomDirect frozen-source full run

The durable full run uses job ID
`nabla-nte64-edge1atom-frozen-v1-20260727a`.

Its launcher verifies the complete SHA-256 manifest for the immutable runtime
tree reconstructed under
`outputs/experiment-source-trees/nabla-nte64-edge1atom-smoke-dd7f2ba6`.
That tree comes from commit `fce46162`, the complete tracked binary diff, and
the untracked-file archive captured at source fingerprint `dd7f2ba6`. The same
tracked source passed the two-GPU full-model smoke before the tree was frozen.

The queue also pins the launcher, tree manifest, model config, provenance JSON,
and complete source-tree archive as immutable inputs. Each full output copies
the source snapshot, source-tree manifest, archive, and queue request alongside
the training artifacts.
