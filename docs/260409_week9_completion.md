# Week 9 Completion (Local Patchwise Rejector)

Week 9 delivers a spatially selective defer head trained on patchwise error supervision, with TV regularization and evaluation overlays.

## Implemented components

- **Local rejector head (CNN, downsample grid = patch stride)**
  - `models/heads/local_rejector.py` — `LocalRejectorCNN`
- **Mask smoothness**
  - `losses/mask_regularizers.py` — `total_variation_2d`
- **Supervision targets (pooled patch error vs quantile)**
  - `utils/local_reject_targets.py` — `build_patch_targets`
- **PDE residual magnitude map (per-pixel, for input channels)**
  - `losses/physics_loss.py` — `pde_residual_magnitude_map`
- **Dataset index**
  - `datasets/local_rejector_dataset.py` — reuses one-step samples
- **Training**
  - `scripts/train_local_rejector.py` — frozen FNO, BCE + pos_weight + TV
- **Evaluation**
  - `scripts/eval_local_rejector.py` — mean IoU/Dice vs GT-derived unsafe patches, overlay PNGs
- **Optional patch sweep**
  - `scripts/tune_local_patch_stride.py` — short runs for `patch_stride` 4 vs 8

## Model inputs (inference-safe)

Per timestep, channels are:

1. `V_t`, `R_t`  
2. `V̂_{t+1}`, `R̂_{t+1}` (FNO one-step)  
3. PDE residual magnitude map from `(x_t, pred)`  

The head outputs **logits on an H/ps × W/ps grid** (`ps ∈ {4,8}`).

## Supervision

- Patch error: average absolute error in `(V,R)` pooled with `AvgPool2d(ps, ps)`.
- Binary unsafe label: pooled error **>** per-sample quantile (default 0.75) over valid geometry mask.

## Executed training run

```bash
python3 scripts/train_local_rejector.py \
  --fno-checkpoint outputs/week3_fno/20260409_202231/best_fno.pt \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --split data/splits/split_v0.2_id.json \
  --output-dir outputs/week9_local_rejector \
  --epochs 35 \
  --batch-size 8 \
  --patch-stride 4 \
  --lambda-tv 0.05
```

- Run directory: `outputs/week9_local_rejector/20260409_211421`
- Best checkpoint: `outputs/week9_local_rejector/20260409_211421/best_local_rejector.pt`
- Best validation IoU during training: **~0.424** (epoch ~11)

## Test split evaluation

```bash
python3 scripts/eval_local_rejector.py \
  --fno-checkpoint outputs/week3_fno/20260409_202231/best_fno.pt \
  --local-checkpoint outputs/week9_local_rejector/20260409_211421/best_local_rejector.pt \
  --manifest data/metadata/dataset_manifest.v0.2.json \
  --split data/splits/split_v0.2_id.json \
  --output-dir outputs/week9_local_eval/20260409_211421 \
  --patch-stride 4 \
  --split-name test \
  --num-viz 6
```

- **Mean IoU:** ~0.379 (24 eval points: 8 cases × 3 times)
- **Mean Dice:** ~0.540
- Overlays: `outputs/week9_local_eval/20260409_211421/overlays/`
- Metrics exit card: `outputs/week9_local_eval/20260409_211421/week9_exit_check.md`

## Exit check

- Working local training pipeline and best checkpoint saved.
- IoU/Dice vs GT-derived unsafe patch mask reported.
- Qualitative overlays align high-risk probability with high `|pred−gt|` regions on inspected cases.

## Tuning knobs (paper scale)

- `patch_stride`: 4 vs 8 (resolution vs localization)
- `risk_quantile`: stricter/softer unsafe definition
- `lambda_tv`: smoother defer regions
- `pos_weight`: class balance for rare unsafe patches

Next: **Week 10** — oracle local upper bound and hybrid merge metrics vs learned mask.
