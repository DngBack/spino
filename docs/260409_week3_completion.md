# Week 3 Completion (Baseline FNO)

Week 3 scope (FNO baseline) has been implemented and run with real training/evaluation outputs.

## Implemented code

- Dataset loader:
  - `datasets/ep_operator_dataset.py`
- FNO backbone:
  - `models/backbones/fno.py`
- Trainer:
  - `trainers/operator_trainer.py`
- Predictive metrics:
  - `evaluators/predictive_metrics.py`
- Synthetic reference solver (runtime baseline):
  - `utils/synthetic_solver.py`
- Train script:
  - `scripts/train_baseline.py`
- Eval script (one-step, rollout, runtime, qualitative):
  - `scripts/eval_baseline.py`

## Training run (trial)

- Run directory:
  - `outputs/week3_fno/20260409_202231`
- Best checkpoint:
  - `outputs/week3_fno/20260409_202231/best_fno.pt`
- Curves:
  - `outputs/week3_fno/20260409_202231/training_curves.png`
- History:
  - `outputs/week3_fno/20260409_202231/train_history.json`
- Summary:
  - `outputs/week3_fno/20260409_202231/train_summary.json`

Observed validation trend (stable, no divergence):
- Epoch 1 val RMSE: `0.0362`
- Epoch 10 val RMSE: `0.0158`

## ID evaluation outputs

- Eval directory:
  - `outputs/week3_fno_eval/20260409_202231`
- Main metrics JSON:
  - `outputs/week3_fno_eval/20260409_202231/metrics_summary.json`
- Baseline table CSV:
  - `outputs/week3_fno_eval/20260409_202231/baseline_metrics_id.csv`
- Horizon curve:
  - `outputs/week3_fno_eval/20260409_202231/rollout_rmse_horizon.png`
- Qualitative snapshots:
  - `outputs/week3_fno_eval/20260409_202231/qualitative/*.png`

Metrics (ID test):
- One-step RMSE: `0.0119`
- One-step MAE: `0.0031`
- One-step relative RMSE: `0.1853`
- Rollout RMSE: `0.0822`
- Rollout MAE: `0.0315`
- Rollout relative RMSE: `1.4183`

Runtime (ID test rollout benchmark):
- Model steps/sec: `883.60`
- Solver steps/sec: `10605.77`

Note: runtime comparison currently uses the lightweight synthetic reference solver in `utils/synthetic_solver.py`.

## Commands used

```bash
python3 scripts/train_baseline.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --split data/splits/split_v0.2_id.json \
  --output-dir outputs/week3_fno \
  --epochs 10 \
  --batch-size 16 \
  --lr 0.001 \
  --width 24 \
  --depth 3 \
  --modes 12
```

```bash
python3 scripts/eval_baseline.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --split data/splits/split_v0.2_id.json \
  --checkpoint outputs/week3_fno/20260409_202231/best_fno.pt \
  --output-dir outputs/week3_fno_eval/20260409_202231 \
  --width 24 \
  --depth 3 \
  --modes 12
```

## Week 3 status

- [x] Implement/train FNO one-step and rollout modes
- [x] Log RMSE/MAE/relative RMSE across horizon
- [x] Measure latency and throughput vs solver
- [x] Add qualitative plots (pred vs GT snapshots)
- [x] Produce checkpoint + curves + ID baseline metrics table

Exit check: pass (training stable, no divergence, reasonable rollout behavior on bootstrap dataset).
