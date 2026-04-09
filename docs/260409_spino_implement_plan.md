# SPINO Implementation Plan (Rewritten)

## 1) Scope and Target Claim

### Project scope (first paper)
- Domain: 2D cardiac electrophysiology (EP) surrogate simulation.
- Backbone: `FNO` and `PINO` operator models.
- Selection: both global defer (case-level) and local defer (patch-level).
- Fallback: trusted solver outputs (oracle repair first), then practical local refinement.

### Working title
**Selective Physics-Informed Neural Operators for Reliable Cardiac Electrophysiology Simulation**

### Core claim to validate
Physics-aware selective prediction (uncertainty + residual + dynamics signals) achieves a better **risk-coverage-compute** trade-off than:
- unconditional surrogate prediction, and
- uncertainty-only abstention.

## 2) Problem Statement (Operational)

Given input `a` (initial condition, stimulation, geometry, parameters), predict trajectory `u_hat` and a reliability output:
- global score `s(a)` for whole-case defer, or
- local trust mask `M(x, t)` for regional defer.

System behavior:
- accept reliable predictions from the fast surrogate,
- defer unsafe regions/cases to a trusted fallback.

This is a learning-to-defer setup in operator-learning form.

## 3) Execution Strategy by Phase

## Phase A - Reproducible surrogate baseline
Build a strong, stable foundation.

Deliverables:
- `FNO` baseline and `PINO` baseline.
- Point-to-point and rollout evaluation.
- Initial ID + simple shift protocols.

Success criteria:
- Reproduce expected trend: `PINO` outperforms `FNO` on long rollout / harder dynamics.
- Surrogate is significantly faster than full solver.

## Phase B - Global selective prediction
Add defer-at-case-level first.

Deliverables:
- Scalar rejector head.
- Global reliability features (uncertainty, residual, drift, OOD).
- Coverage-controlled thresholding and global selective metrics.

Success criteria:
- Global selective policy beats unconditional prediction at matched compute or matched risk.

## Phase C - Local selective prediction (core novelty)
Move to patchwise local defer before any pixelwise approach.

Deliverables:
- Local risk head outputting patchwise `M`.
- Mask regularization and region-level metrics.
- Oracle local repair for clean upper-bound analysis.

Success criteria:
- Local defer beats global defer at comparable fallback budget.
- Rejected patches concentrate most high-error regions.

## Phase D - Hybrid realistic fallback
Convert from "abstention demo" to deployable hybrid simulator.

Implement at least one:
- patch replacement from trusted solver,
- coarse-to-fine local refinement, or
- slower secondary surrogate for deferred regions.

Success criteria:
- Hybrid reduces catastrophic error while preserving a strong fraction of surrogate speedup.

## 4) Repository Structure

Use modular separation for simulation, modeling, defer logic, and evaluation:

```text
spino/
  configs/{data,model,train,eval}/
  data/{raw,processed,splits,metadata}/
  simulators/{opencarp,synthetic}/
  datasets/
  models/{backbones,heads,hybrid}/
  losses/
  features/
  trainers/
  evaluators/
  calibration/
  scripts/
  utils/
```

## 5) File-by-File Implementation Order

1. **Data + simulator plumbing**
   - `simulators/opencarp/generate_cases.py`
   - `simulators/opencarp/export_grids.py`
   - `datasets/ep_operator_dataset.py`

2. **Operator baseline**
   - `models/backbones/fno.py`
   - `models/backbones/pino_fno.py`
   - `losses/data_loss.py`
   - `losses/physics_loss.py`
   - `trainers/operator_trainer.py`

3. **Evaluation harness**
   - `evaluators/predictive_metrics.py`
   - `evaluators/event_metrics.py`
   - `scripts/eval_id.py`
   - `scripts/eval_shift.py`

4. **Reliability features**
   - `features/residual_maps.py`
   - `features/uncertainty_features.py`
   - `features/rollout_drift.py`
   - `features/ood_features.py`

5. **Global rejector**
   - `models/heads/global_rejector.py`
   - `losses/selective_loss.py`
   - `losses/coverage_loss.py`
   - `calibration/threshold_search.py`
   - `trainers/rejector_trainer.py`

6. **Local rejector**
   - `models/heads/local_rejector.py`
   - `losses/mask_regularizers.py`
   - `models/hybrid/patch_merger.py`

7. **Hybrid fallback**
   - `models/hybrid/fallback_router.py`
   - `simulators/opencarp/patch_refine.py`
   - `models/hybrid/selective_wrapper.py`

## 6) Data Plan

### Minimal dataset for paper v1
- 2D EP scenarios: planar, centrifugal, stable spiral, spiral breakup.
- Parameter sweeps: conductivity/diffusion, excitability, restitution, stimulation site.
- Structural variation: mesh resolution + simple boundary/geometry perturbation.

### Split families
- ID split: held-out seeds.
- Parameter shift split.
- Geometry shift split.
- Rollout shift split (train short horizon, test longer horizon).

### Practical generation target
- 4 scenarios.
- 8-12 parameter settings per scenario.
- 20-40 seeds per setting.
- 200-400 time steps per trajectory.
- 2-3 resolutions.

## 7) Model and Feature Design

### Backbone defaults
- 4 FNO blocks.
- Hidden width 32 or 48.
- Inputs: voltage, recovery, positional encoding, optional parameter channels.
- Output: next frame or short window.

### Rejector inputs
Global features:
- mean/max PDE residual,
- BC violation norm,
- rollout drift score,
- ensemble variance summary,
- latent distance / parameter OOD / geometry OOD.

Local features (patchwise):
- local residual map,
- local uncertainty map,
- local temporal inconsistency,
- local gradient/wavefront complexity,
- optional activation-time mismatch proxy.

### Rejector outputs
- Global: scalar risk score `s in R` (lower = safer).
- Local: patch score map `S in R^(T' x H' x W')`, thresholded to `M`.

## 8) Training Plan

## Stage 1 - Operator pretraining
Train `FNO/PINO` without defer heads.

Loss:
`L_op = lambda_data * L_data + lambda_phys * L_phys + lambda_bc * L_bc (+ lambda_ic * L_ic if needed)`

Outputs to freeze:
- strongest checkpoint,
- baseline plots (ID + rollout),
- residual and drift artifacts.

## Stage 2 - Rejector target construction
Build supervision from realized error.

Global targets:
- rollout RMSE, event error, max local error.
- use continuous/ranking target if possible (preferred over hard bins).

Local targets:
- patchwise local L2 error,
- unsafe patch map via threshold,
- optional dilation for class-balance stability.

## Stage 3 - Rejector training
Freeze backbone first.

Loss options:
- BCE / focal for safe-vs-unsafe,
- ranking loss for ordering quality,
- coverage penalty for target acceptance rate.

Local regularization:
`L = L_risk + lambda_cov * L_cov + lambda_tv * L_smooth`

## Stage 4 - Joint fine-tuning
Unfreeze only later backbone layers.

Rationale:
- full end-to-end updates can destabilize backbone and make rejector compensate for poor simulation quality.

## 9) Calibration Plan

Always calibrate defer thresholds on validation splits.

Report three operating modes:
- fixed target coverage,
- fixed accepted risk,
- fixed fallback budget.

## 10) Metrics

### Predictive
- RMSE, relative RMSE, MAE.
- rollout RMSE vs horizon.
- event-level metrics (activation time MAE, CV error, spiral-tip trajectory error).

### Selective
Global:
- coverage,
- selective risk,
- AURC-style area,
- accepted RMSE at fixed coverage,
- coverage at fixed accepted RMSE.

Local:
- localized coverage,
- localized selective risk,
- unsafe-region IoU / Dice,
- accepted-region RMSE,
- error concentration ratio.

### Physics
- PDE residual norm,
- BC violation,
- residual drift over rollout.

### Efficiency
- surrogate / fallback / hybrid latency,
- deferred fraction,
- speedup vs full solver,
- risk per unit compute.

## 11) Experiment Matrix

## Block A - Backbone study
- Models: `FNO`, `PINO`.
- Regimes: one-step, rollout, ID, parameter shift, geometry shift, long horizon.
- Goal: establish strong baseline simulator.

## Block B - Global selective study
- Methods: confidence-only, uncertainty-only, residual-only, joint score, oracle global selector.
- Outputs: risk-coverage and compute-risk curves.
- Goal: prove selective utility before local routing.

## Block C - Local selective study (main)
- Methods: global rejector, local uncertainty-only, local residual-only, local joint, oracle local.
- Fallbacks: oracle repair first, optional numerical patch refinement.
- Outputs: local coverage-risk, accepted/rejected error split, hybrid Pareto.

## 12) Paper Assets Checklist

### Priority tables
1. Backbone performance.
2. Global selective at matched coverage.
3. Global selective at matched accepted risk.
4. Local mask quality.
5. Hybrid routing benefit.
6. Shift robustness.
7. Ablations.

### Priority figures
1. System overview diagram.
2. Qualitative EP case comparisons.
3. Risk-coverage curves (global + local).
4. Compute-risk Pareto frontier.
5. Error concentration plots.
6. Rollout stability under selection.
7. Shift robustness panels.
8. Ablation heatmap.

## 13) Theory-to-Experiment Mapping

Keep theory testable:
- If theorem uses accepted risk -> report accepted RMSE/event error + coverage.
- If theorem uses residual-aware scoring -> include residual-only and joint baselines + residual-error correlation.
- If theorem is local -> report local coverage/risk and rejected-region error concentration.

## 14) Oracle Studies (Required)

- Oracle global selector: reject highest true-error cases.
- Oracle local selector: reject highest true-error patches.
- Oracle repair: replace rejected regions with trusted solver outputs.

Purpose: separate selector quality from repair quality and quantify headroom.

## 15) Failure Analysis (Required Section)

Add a dedicated subsection: **Where does the rejector fail?**

Analyze:
- false accepts with low residual but high error,
- false rejects in complex yet safe regions,
- shift regimes dominated by OOD proxies,
- horizons where residual signals rise too late.

## 16) 10-12 Week Schedule

- **Weeks 1-2:** data export pipeline, ID dataset, `FNO/PINO` baseline.
- **Weeks 3-4:** shift splits, rollout + event evaluation, baseline tables frozen.
- **Weeks 5-6:** feature extraction, global rejector, calibration, global selective results.
- **Weeks 7-8:** local patchwise rejector, local targets, oracle local studies.
- **Weeks 9-10:** hybrid fallback routing, compute-risk Pareto, major ablations.
- **Weeks 11-12:** finalize figures/tables, write results/discussion, lock appendix/theory links.

## 17) Go/No-Go Criteria

Project is strong if at least two hold:
1. Joint physics+uncertainty rejector beats uncertainty-only.
2. Local defer beats global defer at equal fallback budget.
3. Hybrid retains substantial speedup while reducing catastrophic errors.

## 18) Recommended First Submission Configuration

- Domain: 2D cardiac EP.
- Base models: `FNO + PINO`.
- Selective methods: global + local (patchwise).
- Features: uncertainty + residual + rollout drift.
- Fallback: oracle local repair first; practical patch refinement optional.
- Experiments: ID, parameter shift, geometry shift, long rollout.
- Theory: one accepted-risk proposition + one local variant.
- Framing: deployment-oriented risk-coverage-compute trade-off.