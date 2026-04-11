# Week 12 — Local vs global hybrid

- Calibration: global τ from **ID val** coverage; local τ_prob via bisection to match **mean defer fraction** (1−coverage) on **ID val**.
- Evaluation: same τ settings applied to each **test** split (including shifts).
- Oracle equiv (full-case defer): `199.0` FNO-step units

- Table: `week12_main_table.csv` (16 rows).
- Count (local RMSE < global RMSE): 12/16 row-wise