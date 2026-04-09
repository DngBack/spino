# Week 6 Completion (Reliability Features Pipeline)

Week 6 has been completed with feature extraction modules, cache export, and analysis artifacts.

## Implemented feature modules

- Residual features:
  - `features/residual_maps.py`
- Uncertainty features:
  - `features/uncertainty_features.py`
- Rollout drift features:
  - `features/rollout_drift.py`
- OOD features:
  - `features/ood_features.py`
- Feature package init:
  - `features/__init__.py`

## Main extraction script

- `scripts/build_reliability_features.py`

What it does:
- loads checkpoint + manifest + ID split,
- builds train embedding bank for OOD scoring,
- extracts global and local features per case:
  - residual map,
  - uncertainty map,
  - drift map,
  - OOD map,
- saves per-case cache (`npz`) and global table (`csv/json`),
- generates feature sanity report and correlation matrix.

## Executed run (FNO baseline)

Command:

```bash
python3 scripts/build_reliability_features.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --id-split data/splits/split_v0.2_id.json \
  --checkpoint outputs/week3_fno/20260409_202231/best_fno.pt \
  --model-type fno \
  --output-dir outputs/week6_features/fno_20260409 \
  --cache-dir data/processed/features_cache \
  --width 24 --depth 3 --modes 12 \
  --uncertainty-samples 4 \
  --drift-horizon 30
```

## Produced artifacts

- Per-case feature cache:
  - `data/processed/features_cache/fno_20260409_202231/*.npz`
- Global feature table:
  - `outputs/week6_features/fno_20260409/global_reliability_features.csv`
  - `outputs/week6_features/fno_20260409/global_reliability_features.json`
- Correlation outputs:
  - `outputs/week6_features/fno_20260409/feature_correlation_matrix.csv`
  - `outputs/week6_features/fno_20260409/feature_correlation_matrix.png`
- Sanity report:
  - `outputs/week6_features/fno_20260409/feature_sanity_report.md`

## Sanity highlights

- Cases processed: `48`
- Strong positive correlation with rollout error:
  - `residual_mean` vs `rollout_rmse`: `0.9350`
  - `drift_mean` vs `rollout_rmse`: `0.9443`
  - `drift_slope` vs `rollout_rmse`: `0.8592`
- Moderate/weak OOD signals on current small dataset:
  - `ood_centroid_distance` vs `rollout_rmse`: `0.1891`
  - `ood_nn_distance` vs `rollout_rmse`: `0.0546`

## Week 6 status

- [x] Feature modules implemented
- [x] Feature cache export working
- [x] Feature sanity report generated
- [x] Correlation matrix generated

This output is ready to feed Week 7 global rejector training.
