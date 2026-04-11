# Shift Benchmark Card

- Dataset: `spino_ep2d`
- Version: `v1.0`
- Scenarios: `['centrifugal_wave', 'planar_wave', 'spiral_breakup', 'stable_spiral']`

## Split Definitions
- `id`: test_case_ids from ID split
- `parameter_shift`: test_shift_case_ids from parameter split (fallback test_case_ids if empty)
- `geometry_shift`: test_shift_case_ids from geometry split (fallback test_case_ids if empty)
- `long_rollout`: evaluates long horizon and reports short-vs-long degradation

## Case Counts
- `id`: `180` cases
- `parameter_shift`: `552` cases
- `geometry_shift`: `416` cases
- `long_rollout`: `180` cases
