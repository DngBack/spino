# Week 11 Completion — Hybrid Fallback Routing

Week 11 turns abstention metrics into an **end-to-end hybrid simulator**: surrogate rollout plus **trusted GT (oracle) repair** where selectors defer.

## Implemented

- **Core helpers**: `utils/hybrid_inference.py`
  - `rollout_fno_only` — baseline autoregressive FNO.
  - `rollout_hybrid_global` — if global score `> τ`, replace full trajectory with GT; else FNO rollout.
  - `rollout_hybrid_local` — each step: FNO → local risk map → bilinear defer mask on tissue → blend prediction with GT.
  - `HybridStats` — FNO forward count, full-case defers, pixel repairs, wall-clock split; `compute_units(...)` for a simple **compute–risk Pareto** (oracle cost weighted by `--oracle-equiv-per-case`).
- **Evaluation script**: `scripts/eval_week11_hybrid.py`
  - Per split family: `always_predict`, `always_defer`, sweeps over `hybrid_global` (τ on test score quantiles) and `hybrid_local` (probability thresholds).
  - Writes `week11_hybrid_pareto.csv` / `.json`, `compute_norm` (relative to `always_predict`), and `pareto_compute_risk_<family>.png`.

## Example run (full test splits)

```bash
python3 scripts/eval_week11_hybrid.py \
  --feature-csv outputs/week6_features/fno_20260409/global_reliability_features.csv \
  --fno-checkpoint outputs/week3_fno/20260409_202231/best_fno.pt \
  --global-checkpoint outputs/week7_global_rejector/20260409_210702/best_global_rejector.pt \
  --local-checkpoint outputs/week9_local_rejector/20260409_211421/best_local_rejector.pt \
  --output-dir outputs/week11_hybrid \
  --device cuda
```

Optional: `--max-cases-per-family K` for quick plots; `--oracle-equiv-per-case X` to stress-test Pareto under cheaper vs more expensive oracle.

## Exit check

Confirm in CSV/PNGs that hybrid curves sit **between** always-predict and always-defer on RMSE for at least one shift family, with **moderate** compute overhead (`compute_norm` or raw `compute_units`).
