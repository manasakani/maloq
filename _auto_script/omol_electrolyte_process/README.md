# OMol electrolyte corrected full-v2 processing

The production entry point is `run_corrected_full_v2.sh`:

```bash
./run_corrected_full_v2.sh status
./run_corrected_full_v2.sh start-be
./run_corrected_full_v2.sh start-missing
./run_corrected_full_v2.sh start-finalize
```

`start-be` rebuilds every already-processed Be shard with the corrected basis convention. `start-missing` rebuilds all accepted shards absent from immutable v1 after the raw-transfer marker is present. `start-finalize` publishes the exact 100,474-sample mixed v2/v1 view and runs metadata, sampled, and full-record validation. All phases are restart-safe, CPU-only, and isolated in tmux.

The project-local verification tools build a complete 100,474-sample dataset view
without copying or modifying the immutable v1 snapshot.

## Contract

- The accepted manifest is authoritative for identity, order, and split.
- A sample present in a real rebuilt-v2 shard always wins over v1.
- Every other accepted sample references its existing v1 shard record.
- The full index is atomically published only at exact manifest coverage.
- The complete view contains absolute symlinks to source shard directories and
  summary files. Its rewritten JSONL indexes contain absolute paths local to
  the view, so the existing OMol loader can open them directly.
- Neither index nor view builders replace an existing destination.
- `COMPLETE` can be written only after `--mode full` checks every LMDB record.

The production roots used on SC26 are:

```text
v1:
/dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb

real rebuilt-v2 overlay (internal provenance source):
/dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_v2_overlay

canonical unified view (user/loader root):
/dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_full_v2
```

The phase launcher uses these exact roots. The individual commands below are
provided for auditing and recovery; normal production runs should use the launcher.

## 1. Coverage dry-run

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/omol_electrolyte_process/verification/build_full_v2_index.py \
  --manifest-dir /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb/manifests/unsolvated_electrolytes_all_supported_elements_85_5_10_v1 \
  --v1-root /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb \
  --v2-root /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_v2_overlay \
  --out-index-root /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_v2_overlay/_full_index \
  --dry-run --allow-incomplete
```

`--allow-incomplete` is accepted only with `--dry-run`; it cannot publish a
partial artifact.

## 2. Publish the full index

Run the same command without `--dry-run --allow-incomplete`. It fails before
publication unless all 100,474 accepted samples resolve to v2 or v1.

## 3. Build the symlink view

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/omol_electrolyte_process/verification/build_full_v2_view.py \
  --index-root /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_v2_overlay/_full_index \
  --manifest-dir /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb/manifests/unsolvated_electrolytes_all_supported_elements_85_5_10_v1 \
  --out-view-root /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_full_v2
```

The view is built under a sibling temporary directory and renamed into place
only after all links and rewritten indexes have been created. It initially has
no `COMPLETE` marker.

## 4. Progressive verification

All-shard metadata and schema:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/omol_electrolyte_process/verification/verify_full_v2.py \
  --manifest-dir /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb/manifests/unsolvated_electrolytes_all_supported_elements_85_5_10_v1 \
  --index-root /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_full_v2/_index \
  --v2-root /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_v2_overlay \
  --view-root /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_full_v2 \
  --mode metadata
```

One or more packed-matrix records per shard:

```bash
# Use the same arguments as above, replacing the final mode:
--mode sampled --records-per-shard 1
```

Final full-record gate and completion marker:

```bash
# Use the same manifest/index/v2/view arguments as above, then:
--mode full \
--max-density-trace-error 0.05 \
--mark-complete /dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_full_v2
```

The verifier checks exact manifest/index coverage, split and identity, v2
precedence, each referenced summary, LMDB metadata and schema, packed float32
density/overlap/SAD lengths and finiteness, and both
`Tr(D S)` and `Tr(D_init S)` against the electron count.
The density-trace production limit is `0.05`: the historical non-Be v1 audit
reached about `0.0112`, so `0.01` would create a known false failure. The
initial-density trace limit remains independently strict at `0.001`; rebuilt-v2
maxima should be reported separately rather than hidden by the mixed-view gate.

## Staging tests

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  -m unittest discover \
  -s /dataset/seongsu/shared-home/workspace/project/_auto_script/omol_electrolyte_process/verification/tests \
  -v
```
