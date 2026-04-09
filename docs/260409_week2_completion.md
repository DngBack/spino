# Week 2 Completion (Dataset Generation + ID Split)

This document records the completed Week 2 deliverables in the current repository state.

## Delivered artifacts

- Synthetic EP2D generator:
  - `scripts/generate_synthetic_ep2d_dataset.py`
- Built dataset manifest:
  - `data/metadata/dataset_manifest.v0.2.json`
- Built split files:
  - `data/splits/split_v0.2_id.json`
  - `data/splits/split_v0.2_param_shift.json`
  - `data/splits/split_v0.2_geometry_shift.json`
  - `data/splits/split_v0.2_long_rollout.json`
- Generation config snapshot:
  - `data/metadata/generation_config.v0.2.json`
- Per-case metadata:
  - `data/metadata/case_metadata/*.json`
- Processed tensors:
  - `data/processed/tensors/*.npz`
- QC and analysis outputs:
  - `outputs/data_analysis_v0.2/summary.json`
  - `outputs/data_analysis_v0.2/cases_table.csv`
  - `outputs/data_analysis_v0.2/report.md`
  - `outputs/data_analysis_v0.2/figures/*.png`

## Current dataset statistics (`v0.2`)

- Total cases: `48`
- Scenarios:
  - `planar_wave`: `12`
  - `centrifugal_wave`: `12`
  - `stable_spiral`: `12`
  - `spiral_breakup`: `12`
- Resolution: `r64`
- Steps per trajectory: `80`
- Missing metadata: `0`
- QC hard errors: `0`

## Commands used

```bash
python3 scripts/generate_synthetic_ep2d_dataset.py \
  --dataset-version v0.2 \
  --num-parameter-sets 2 \
  --num-seeds-per-parameter-set 3 \
  --num-geometries 2 \
  --num-steps 80 \
  --resolutions "64" \
  --clean
```

```bash
python3 scripts/build_manifest.py \
  --metadata-dir data/metadata/case_metadata \
  --output-manifest data/metadata/dataset_manifest.v0.2.json \
  --dataset-name spino_ep2d \
  --dataset-version v0.2 \
  --generation-config-path data/metadata/generation_config.v0.2.json \
  --split-id-path data/splits/split_v0.2_id.json \
  --split-parameter-path data/splits/split_v0.2_param_shift.json \
  --split-geometry-path data/splits/split_v0.2_geometry_shift.json \
  --split-long-rollout-path data/splits/split_v0.2_long_rollout.json
```

```bash
python3 scripts/build_splits.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --output-dir data/splits \
  --dataset-version v0.2 \
  --seed 42 \
  --param-key excitability \
  --param-threshold 0.18 \
  --geometry-holdout-fraction 0.34
```

```bash
python3 scripts/data_qc.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --check-tensors
```

```bash
python3 scripts/visualize_and_analyze_data.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --output-dir outputs/data_analysis_v0.2
```

## Week 2 status

Week 2 is complete for a bootstrap dataset (`v0.2`) that is trainable and reproducible.

For paper-scale runs (`v1.0`), increase:
- parameter sets,
- seeds per setting,
- geometries,
- resolutions (`r64` + `r128` at minimum),
- rollout length (toward `200-400` steps).
