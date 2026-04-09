# Week 5 Completion (Shift Protocols + Evaluation Harness)

Week 5 has been completed with shift-ready splits, one-command shift evaluation, and frozen baseline outputs.

## Implemented deliverables

- Shift evaluation harness:
  - `scripts/eval_shift.py`
  - Evaluates one checkpoint across all split families in one command.
- Shift splits finalized (v0.2):
  - `data/splits/split_v0.2_param_shift.json`
  - `data/splits/split_v0.2_geometry_shift.json`
  - `data/splits/split_v0.2_long_rollout.json`
- Standardized outputs:
  - `shift_metrics_table.csv`
  - `shift_metrics_summary.json`
  - `per_split/*_summary.json`
  - `per_split/*_per_case_metrics.csv`
  - `shift_benchmark_card.md`

## Shift benchmark card

Generated benchmark card includes split definitions and case counts:
- `outputs/week5_shift_eval/fno_20260409/shift_benchmark_card.md`
- `outputs/week5_shift_eval/pino_20260409/shift_benchmark_card.md`

Current split sizes (`v0.2`):
- ID: `8`
- Parameter shift: `6`
- Geometry shift: `24`
- Long rollout: `8`

## Frozen baseline performance on all split families

FNO run:
- `outputs/week5_shift_eval/fno_20260409/shift_metrics_table.csv`

PINO run:
- `outputs/week5_shift_eval/pino_20260409/shift_metrics_table.csv`

Snapshot (rollout RMSE):
- ID: FNO `0.08220` vs PINO `0.08106`
- Parameter shift: FNO `0.02083` vs PINO `0.01679`
- Geometry shift: FNO `0.08148` vs PINO `0.08033`
- Long rollout: FNO `0.08220` vs PINO `0.08106`

## One-command evaluation (exit-check requirement)

FNO:

```bash
python3 scripts/eval_shift.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --checkpoint outputs/week3_fno/20260409_202231/best_fno.pt \
  --model-type fno \
  --output-dir outputs/week5_shift_eval/fno_20260409 \
  --width 24 --depth 3 --modes 12 \
  --short-horizon 20
```

PINO:

```bash
python3 scripts/eval_shift.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --checkpoint outputs/week4_pino/20260409_203005/best_fno.pt \
  --model-type pino \
  --output-dir outputs/week5_shift_eval/pino_20260409 \
  --width 24 --depth 3 --modes 12 \
  --short-horizon 20
```

## Week 5 status

- [x] Parameter-shift split created
- [x] Geometry-shift split created (held-out geometries)
- [x] Long-horizon shift split created and evaluated
- [x] Standardized evaluation outputs/report card
- [x] Frozen baseline performance produced across split families
- [x] Exit check satisfied (single command evaluates all split families)
