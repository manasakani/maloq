# Experimental tests

Put feature-owned tests under
`tests/experimental/<semantic_feature_slug>/test_*.py`.

The repository-level import-boundary test in this directory protects the
one-way dependency rule: experimental code may depend on canonical MALOQ, but
canonical MALOQ must not import `maloq.experimental`.

Promotion moves the selected tests into the canonical test area together with
the implementation. Tests for rejected ablations should not become permanent
canonical compatibility contracts.
