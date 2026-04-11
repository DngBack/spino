# Week 6 Feature Sanity Report

- Model type: `fno`
- Checkpoint: `outputs/week3_fno/20260410_085629/best_fno.pt`
- Cases processed: `1200`
- Feature cache directory: `data/processed/features_cache/fno_20260410_085629`

## Mean feature values
- `residual_mean`: `0.007248`
- `residual_max`: `0.064972`
- `uncertainty_mean`: `0.000083`
- `uncertainty_max`: `0.000095`
- `drift_mean`: `0.021535`
- `drift_slope`: `0.000489`
- `ood_centroid_distance`: `0.000093`
- `ood_nn_distance`: `0.000000`
- `rollout_rmse`: `0.066024`

## Correlation with rollout_rmse
- `residual_mean` vs `rollout_rmse`: `0.9578`
- `residual_max` vs `rollout_rmse`: `0.5053`
- `uncertainty_mean` vs `rollout_rmse`: `-0.8337`
- `uncertainty_max` vs `rollout_rmse`: `-0.0231`
- `drift_mean` vs `rollout_rmse`: `0.9590`
- `drift_slope` vs `rollout_rmse`: `0.8593`
- `ood_centroid_distance` vs `rollout_rmse`: `0.2929`
- `ood_nn_distance` vs `rollout_rmse`: `0.0245`