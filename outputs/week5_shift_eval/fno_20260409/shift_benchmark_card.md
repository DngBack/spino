# Shift Benchmark Card

- Dataset: `spino_ep2d`
- Version: `v0.2`
- Scenarios: `['centrifugal_wave', 'planar_wave', 'spiral_breakup', 'stable_spiral']`

## Split Definitions
- `id`: test_case_ids from ID split
- `parameter_shift`: test_shift_case_ids from parameter split (fallback test_case_ids if empty)
- `geometry_shift`: test_shift_case_ids from geometry split (fallback test_case_ids if empty)
- `long_rollout`: evaluates long horizon and reports short-vs-long degradation

## Case Counts
- `id`: `8` cases
- `parameter_shift`: `6` cases
- `geometry_shift`: `24` cases
- `long_rollout`: `8` cases
