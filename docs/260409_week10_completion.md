# Week 10 Completion (Oracle Local Studies + Headroom)

Week 10 is implemented end-to-end to separate selector quality from repair quality.

## Implemented

- Oracle studies script:
  - `scripts/eval_week10_oracle_studies.py`

The script computes, per split family (`id`, `parameter_shift`, `geometry_shift`, `long_rollout`):
- **global learned** selective risk curve (from trained global rejector),
- **global oracle** selective risk curve (accept lowest-risk cases by ground-truth rollout RMSE),
- **local learned** selective risk curve (from trained local rejector scores on patches),
- **local oracle** selective risk curve (accept lowest-error patches by ground-truth patch error),
- **oracle repair gains**:
  - local hybrid learned: `mean(patch_err * accepted_mask_learned)`,
  - local hybrid oracle: `mean(patch_err * accepted_mask_oracle)`,
  where rejected patches are assumed repaired perfectly.

## Executed run

```bash
python3 scripts/eval_week10_oracle_studies.py \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --feature-csv outputs/week6_features/fno_20260409/global_reliability_features.csv \
  --global-checkpoint outputs/week7_global_rejector/20260409_210702/best_global_rejector.pt \
  --fno-checkpoint outputs/week3_fno/20260409_202231/best_fno.pt \
  --local-checkpoint outputs/week9_local_rejector/20260409_211421/best_local_rejector.pt \
  --output-dir outputs/week10_oracle \
  --patch-stride 4 \
  --target-coverage 0.75
```

Run directory:
- `outputs/week10_oracle/20260409_211934`

## Delivered artifacts

- Oracle comparison tables:
  - `week10_oracle_comparison.csv`
  - `week10_oracle_comparison.json`
- Bottleneck diagnosis:
  - `week10_bottleneck_diagnosis.csv`
  - `week10_bottleneck_diagnosis.json`
  - `week10_oracle_card.md`
- Headroom plots:
  - `headroom_id.png`
  - `headroom_parameter_shift.png`
  - `headroom_geometry_shift.png`
  - `headroom_long_rollout.png`
- Oracle repair gain plots:
  - `oracle_repair_gain_id.png`
  - `oracle_repair_gain_parameter_shift.png`
  - `oracle_repair_gain_geometry_shift.png`
  - `oracle_repair_gain_long_rollout.png`

## Key diagnosis at target coverage 0.75

From `week10_bottleneck_diagnosis.csv`:
- `id`: global_headroom=`0.000000`, local_headroom=`0.000856`, local stronger upside=`True`
- `parameter_shift`: global_headroom=`0.000009`, local_headroom=`0.000093`, local stronger upside=`True`
- `geometry_shift`: global_headroom=`0.000000`, local_headroom=`0.000861`, local stronger upside=`True`
- `long_rollout`: global_headroom=`0.000000`, local_headroom=`0.000856`, local stronger upside=`True`

Interpretation:
- On current small dataset, global selector is already near oracle ceiling.
- Local selector still has measurable headroom, consistent with Week 9 objective and Week 10 hypothesis.

## Week 10 status

- [x] Oracle global selector implemented
- [x] Oracle local selector implemented
- [x] Oracle repair metric implemented
- [x] Headroom quantified and visualized
- [x] Bottleneck diagnosis produced

