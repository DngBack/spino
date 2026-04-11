# Week 8 Global Baseline Comparison

- Target coverage: `0.8`

## Calibrated thresholds (ID validation)
- `uncertainty_only`: `0.000083`
- `residual_only`: `0.012241`
- `drift_only`: `0.045603`
- `ood_only`: `0.000138`
- `joint`: `6.440747`

## Calibration robustness (bootstrap on ID val)
- `uncertainty_only`: tau_std=0.000000, coverage_std=0.0034, selective_risk_std=0.00466
- `residual_only`: tau_std=0.000599, coverage_std=0.0062, selective_risk_std=0.00358
- `drift_only`: tau_std=0.005232, coverage_std=0.0033, selective_risk_std=0.00396
- `ood_only`: tau_std=0.000022, coverage_std=0.0062, selective_risk_std=0.00348
- `joint`: tau_std=2.744951, coverage_std=0.0042, selective_risk_std=0.00425