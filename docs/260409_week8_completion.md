# Week 8 Completion (Global Baseline Comparisons + Calibration Study)

Week 8 has been completed with baseline method comparisons and calibration robustness analysis.

## Implemented script

- `scripts/eval_week8_global_baselines.py`

This script:
- compares `joint` rejector vs `uncertainty_only`, `residual_only`, `drift_only`, `ood_only`,
- calibrates thresholds on ID validation at matched target coverage,
- evaluates all methods on ID + parameter shift + geometry shift + long rollout,
- exports risk-coverage plots and summary tables,
- runs bootstrap calibration stability analysis.

## Executed run

Command:

```bash
python3 scripts/eval_week8_global_baselines.py \
  --feature-csv outputs/week6_features/fno_20260409/global_reliability_features.csv \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --rejector-checkpoint outputs/week7_global_rejector/20260409_210702/best_global_rejector.pt \
  --target-coverage 0.8 \
  --calib-bootstrap 40 \
  --output-dir outputs/week8_global_baselines/20260409_2115
```

## Produced artifacts

- Comparison table:
  - `outputs/week8_global_baselines/20260409_2115/week8_global_baseline_comparison.csv`
- Comparison JSON:
  - `outputs/week8_global_baselines/20260409_2115/week8_global_baseline_comparison.json`
- Calibration robustness table:
  - `outputs/week8_global_baselines/20260409_2115/week8_calibration_robustness.csv`
- Calibration robustness JSON:
  - `outputs/week8_global_baselines/20260409_2115/week8_calibration_robustness.json`
- Summary card:
  - `outputs/week8_global_baselines/20260409_2115/week8_summary_card.md`
- Risk-coverage plots:
  - `risk_coverage_compare_id.png`
  - `risk_coverage_compare_parameter_shift.png`
  - `risk_coverage_compare_geometry_shift.png`
  - `risk_coverage_compare_long_rollout.png`

## Key observations

At matched calibration target coverage (0.8):
- On ID / geometry / long-rollout, `residual_only`, `drift_only`, and `joint` all reduce selective risk strongly vs `uncertainty_only`.
- `ood_only` is intermediate.
- On parameter-shift split (small/easy on current dataset), most methods converge to near-identical performance.

Calibration robustness (bootstrap on ID val):
- `joint` and `residual_only` show lower selective-risk variability than `uncertainty_only`.
- `uncertainty_only` threshold is numerically very tight in this small dataset and less discriminative.

## Week 8 status

- [x] Baselines implemented and compared
- [x] Matched-coverage calibration analysis completed
- [x] Risk-coverage visual comparisons produced
- [x] Calibration robustness report produced

This completes Week 8 deliverables and provides the ablation evidence needed before moving to Week 9 local rejector.
