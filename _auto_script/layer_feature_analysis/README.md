# NablaDFT QHFlow3 vs NTE layer feature analysis

This helper compares the completed checkpoints named:

- `NablaDFT | QHFlow3 | Muon | RAW | V2`
- `NablaDFT | NTE-64/2 | Muon | RAW | V1`

It replays the same held-out NablaDFT molecules through both models and writes
degree-wise node/edge activation scales, channel distributions, output
projection spectra, and Muon-head semantic contraction spectra below
`outputs/nabladft-qhf-vs-nte-layer-analysis/`.

Validate paths without starting CUDA:

```bash
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
PROJ_PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
cd "$PROJECT_ROOT"
"$PROJ_PY" _auto_script/layer_feature_analysis/analyze_nabladft_qhf_vs_nte.py \
  --validate-only
```

Run the default eight-molecule comparison on one idle GPU:

```bash
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
PROJ_PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=6 "$PROJ_PY" \
  _auto_script/layer_feature_analysis/analyze_nabladft_qhf_vs_nte.py \
  --num-molecules 8 \
  --batch-size 1 \
  --master-port 29651
```

Use `--num-molecules 64` for the full fixed validation split. Do not compare
QHFlow3's two raw pair curves as if they were recurrent EdgeBlock states:
both pair blocks read the final node state independently, then their outputs
are summed and normalized. The generated report preserves that distinction.
