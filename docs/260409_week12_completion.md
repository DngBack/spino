# Week 12 Completion — Local vs Global Main Experiment

Week 12 produces **paper-ready contrasts** between hybrid **global** (case-level defer) and hybrid **local** (patch-level defer) under a **shared calibration target on ID validation**.

## Implemented

- **Script**: `scripts/eval_week12_main.py`
  - **Calibration (ID val)**:
    - Global: `τ_g = quantile(scores, coverage)` via `find_threshold_for_target_coverage` (same convention as Week 7: accept when `score ≤ τ`).
    - Local: bisection on probability threshold so **mean per-case defer fraction** on val ≈ `1 − coverage` (aligned with global case-defer mass only in a loose sense; see note below).
  - **Evaluation**: same `τ_g`, `τ_l` applied to each **test** split (`id`, `parameter_shift`, `geometry_shift`, `long_rollout`).
  - Outputs: `week12_main_table.csv` / `.json`, `week12_main_bar_by_family.png`, `week12_risk_compute_traces.png`, `week12_main_card.md`.

## Example run

```bash
python3 scripts/eval_week12_main.py \
  --feature-csv outputs/week6_features/fno_20260409/global_reliability_features.csv \
  --fno-checkpoint outputs/week3_fno/20260409_202231/best_fno.pt \
  --global-checkpoint outputs/week7_global_rejector/20260409_210702/best_global_rejector.pt \
  --local-checkpoint outputs/week9_local_rejector/20260409_211421/best_local_rejector.pt \
  --output-dir outputs/week12_main \
  --coverages 0.5,0.65,0.8,0.9 \
  --max-val-cases 32 \
  --device cuda
```

**Performance**: local τ search replays full val rollouts per bisection step; use `--max-val-cases` while iterating, then raise for final numbers.

## Interpretation note

Global “defer mass” is **case-wise** (0 or 1); local defer mass is **pixel-wise**. Matching `1 − coverage` on val is a practical way to tie both policies to one knob; for strict **matched compute**, use `test_compute_norm_*` from the table and subsample or re-calibrate.

## Exit check

Inspect `week12_main_table.csv`: count rows with `local_beats_global_rmse == 1` on shift families — paper claim is strongest if local wins in **multiple** regimes at comparable `test_compute_norm_*`.
