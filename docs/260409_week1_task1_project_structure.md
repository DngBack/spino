# Week 1 Task 1: Project Structure (Finalized)

This document defines the baseline repository layout for SPINO implementation.

## Top-level layout

```text
spino/
  configs/
    data/
    model/
    train/
    eval/
  data/
    raw/
    processed/
    splits/
    metadata/
  simulators/
    opencarp/
    synthetic/
  datasets/
  models/
    backbones/
    heads/
    hybrid/
  losses/
  features/
  trainers/
  evaluators/
  calibration/
  scripts/
  utils/
  docs/
```

## Notes
- `configs/` is separated by concern to keep data/model/train/eval config files isolated.
- `data/` stores generated artifacts; large files should stay ignored unless required.
- `simulators/` contains PDE/EP data generation and refinement interfaces.
- `models/` is split into operator backbones, rejector heads, and hybrid wrappers.
- `trainers/` and `evaluators/` keep orchestration and metrics independent from model code.
