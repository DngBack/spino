# Week 4 Completion (PINO Baseline + Physics Loss Calibration)

Week 4 scope has been implemented and executed with actual training/evaluation runs.

## Added implementations

- PINO backbone alias on FNO architecture:
  - `models/backbones/pino_fno.py`
- Physics-informed loss terms:
  - `losses/physics_loss.py`
  - PDE residual loss
  - BC penalty
  - IC penalty
- Trainer update for PINO objective:
  - `trainers/operator_trainer.py`
- PINO training script:
  - `scripts/train_pino_baseline.py`
- FNO vs PINO comparison script:
  - `scripts/compare_fno_pino.py`

## PINO training run

- Run directory:
  - `outputs/week4_pino/20260409_203005`
- Checkpoint:
  - `outputs/week4_pino/20260409_203005/best_fno.pt`
- Curves:
  - `outputs/week4_pino/20260409_203005/training_curves.png`
- History:
  - `outputs/week4_pino/20260409_203005/train_history.json`

Loss weights used:
- `lambda_data = 1.0`
- `lambda_phys = 0.03`
- `lambda_bc = 0.005`
- `lambda_ic = 0.02`

Validation trend:
- Epoch 1 val RMSE: `0.0310`
- Epoch 10 val RMSE: `0.0156`
- Stable optimization, no divergence observed.

## PINO evaluation (ID test)

- Eval directory:
  - `outputs/week4_pino_eval/20260409_203005`
- Metrics summary:
  - `outputs/week4_pino_eval/20260409_203005/metrics_summary.json`
- Metrics table:
  - `outputs/week4_pino_eval/20260409_203005/baseline_metrics_id.csv`
- Horizon curve:
  - `outputs/week4_pino_eval/20260409_203005/rollout_rmse_horizon.png`

Metrics:
- One-step RMSE: `0.0116`
- One-step MAE: `0.0028`
- One-step relative RMSE: `0.1744`
- Rollout RMSE: `0.0811`
- Rollout MAE: `0.0300`
- Rollout relative RMSE: `1.2814`

## FNO vs PINO comparison artifacts

- Comparison dir:
  - `outputs/week4_comparison/20260409_203005`
- Table:
  - `outputs/week4_comparison/20260409_203005/fno_vs_pino_metrics.csv`
- Plots:
  - `outputs/week4_comparison/20260409_203005/fno_vs_pino_val_rmse.png`
  - `outputs/week4_comparison/20260409_203005/fno_vs_pino_bar.png`

## Main comparison snapshot (ID test)

- One-step RMSE: FNO `0.0119` -> PINO `0.0116` (better)
- Rollout RMSE: FNO `0.0822` -> PINO `0.0811` (better)
- Rollout relative RMSE: FNO `1.4183` -> PINO `1.2814` (better)

## Commands executed

```bash
python3 scripts/train_pino_baseline.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --split data/splits/split_v0.2_id.json \
  --output-dir outputs/week4_pino \
  --epochs 10 \
  --batch-size 16 \
  --lr 0.001 \
  --width 24 \
  --depth 3 \
  --modes 12 \
  --lambda-data 1.0 \
  --lambda-phys 0.03 \
  --lambda-bc 0.005 \
  --lambda-ic 0.02
```

```bash
python3 scripts/eval_baseline.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --split data/splits/split_v0.2_id.json \
  --checkpoint outputs/week4_pino/20260409_203005/best_fno.pt \
  --output-dir outputs/week4_pino_eval/20260409_203005 \
  --width 24 \
  --depth 3 \
  --modes 12
```

```bash
python3 scripts/compare_fno_pino.py \
  --fno-metrics outputs/week3_fno_eval/20260409_202231/metrics_summary.json \
  --pino-metrics outputs/week4_pino_eval/20260409_203005/metrics_summary.json \
  --fno-history outputs/week3_fno/20260409_202231/train_history.json \
  --pino-history outputs/week4_pino/20260409_203005/train_history.json \
  --output-dir outputs/week4_comparison/20260409_203005
```

## Week 4 status

- [x] Implement physics residual + BC/IC terms
- [x] Tune physics loss weights to stable setting
- [x] Train PINO baseline and get checkpoint
- [x] Compare FNO vs PINO with shared eval pipeline
- [x] Produce comparison table and figures

Exit check: pass (PINO shows measurable gain on physics-consistent synthetic benchmark).
