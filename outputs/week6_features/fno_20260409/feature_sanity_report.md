# Week 6 Feature Sanity Report

- Model type: `fno`
- Checkpoint: `outputs/week3_fno/20260409_202231/best_fno.pt`
- Cases processed: `48`
- Feature cache directory: `data/processed/features_cache/fno_20260409_202231`

## Mean feature values
- `residual_mean`: `0.022575`
- `residual_max`: `0.259668`
- `uncertainty_mean`: `0.000064`
- `uncertainty_max`: `0.000083`
- `drift_mean`: `0.024832`
- `drift_slope`: `0.001206`
- `ood_centroid_distance`: `0.000103`
- `ood_nn_distance`: `0.000007`
- `rollout_rmse`: `0.084394`

## Correlation with rollout_rmse
- `residual_mean` vs `rollout_rmse`: `0.9350`
- `residual_max` vs `rollout_rmse`: `0.3455`
- `uncertainty_mean` vs `rollout_rmse`: `-0.6343`
- `uncertainty_max` vs `rollout_rmse`: `0.7330`
- `drift_mean` vs `rollout_rmse`: `0.9443`
- `drift_slope` vs `rollout_rmse`: `0.8592`
- `ood_centroid_distance` vs `rollout_rmse`: `0.1891`
- `ood_nn_distance` vs `rollout_rmse`: `0.0546`