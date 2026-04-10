# Week 7 Completion (Global Rejector + First Selective Results)

Week 7 has been implemented end-to-end with training, calibration, and evaluation of a global defer policy.

## Implemented components

- Global rejector model:
  - `models/heads/global_rejector.py`
- Selective losses:
  - `losses/selective_loss.py`
  - `losses/coverage_loss.py`
- Threshold calibration and risk-coverage utilities:
  - `calibration/threshold_search.py`
- Rejector trainer:
  - `trainers/rejector_trainer.py`
- Training script:
  - `scripts/train_global_rejector.py`
- Evaluation script:
  - `scripts/eval_global_rejector.py`

## Training run

- Run directory:
  - `outputs/week7_global_rejector/20260409_210702`
- Best checkpoint:
  - `outputs/week7_global_rejector/20260409_210702/best_global_rejector.pt`
- History:
  - `outputs/week7_global_rejector/20260409_210702/rejector_history.json`
- Curves:
  - `outputs/week7_global_rejector/20260409_210702/rejector_training_curves.png`

Training setup:
- Input features: residual, uncertainty, drift, OOD global features
- Unsafe label: top `25%` rollout RMSE on full table (`risk_quantile=0.75`)
- Coverage target: `0.8`

## Evaluation outputs (ID + shifts)

- Eval directory:
  - `outputs/week7_global_eval/20260409_210702`
- Main summary:
  - `outputs/week7_global_eval/20260409_210702/global_selective_summary.csv`
  - `outputs/week7_global_eval/20260409_210702/global_selective_summary.json`
- Benchmark card:
  - `outputs/week7_global_eval/20260409_210702/week7_global_card.md`
- Risk-coverage plots:
  - `risk_coverage_id.png`
  - `risk_coverage_parameter_shift.png`
  - `risk_coverage_geometry_shift.png`
  - `risk_coverage_long_rollout.png`

## Key results snapshot

- Calibrated threshold (ID val): `tau=-0.934145`

Selective vs overall risk:
- ID:
  - coverage `0.75`
  - selective risk `0.05582`
  - overall risk `0.08220`
- Geometry shift:
  - coverage `0.75`
  - selective risk `0.05583`
  - overall risk `0.08148`
- Long rollout:
  - coverage `0.75`
  - selective risk `0.05582`
  - overall risk `0.08220`
- Parameter shift:
  - coverage `1.00` (all accepted on this small split)
  - selective risk `0.02083`
  - overall risk `0.02083`

## Week 7 status

- [x] Global rejector implemented
- [x] Reliability features integrated for rejector input
- [x] Coverage-target calibration implemented
- [x] First selective risk-coverage results produced on ID + shift

This completes Week 7 deliverables and prepares Week 8 ablation/comparison against uncertainty-only and residual-only selectors.
