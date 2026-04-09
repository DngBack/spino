# SPINO Detailed Weekly Execution Plan

## Goal
Implement and deliver a strong paper-ready SPINO pipeline:
- selective physics-informed neural operators,
- global and local defer mechanisms,
- hybrid fallback simulation,
- evaluation under in-distribution and shift regimes,
- publication-ready figures, tables, and claims.

This plan assumes a **16-week execution window** with clear outputs each week.

---

## Success Criteria (Paper-Level)
- Joint score (uncertainty + residual + drift + shift features) beats uncertainty-only rejector.
- Local defer beats global defer at similar fallback budget.
- Hybrid method reduces catastrophic errors while preserving meaningful speedup vs full solver.
- Results are reproducible (fixed seeds/configs/scripts) and support a coherent paper narrative.

---

## Week-by-Week Plan

## Week 1 - Environment + Reproducibility Foundation
**Objectives**
- Lock project structure and reproducible training/eval workflow.
- Confirm data generation path and metadata schema.

**Tasks**
- Finalize folder layout for configs, datasets, models, trainers, evaluators.
- Add baseline config files (`data`, `model`, `train`, `eval`).
- Implement run tracking (seed, git hash, config snapshot, output dirs).
- Define case metadata format (scenario, parameters, geometry ID, seed).

**Deliverables**
- `scripts/train_baseline.py` + `scripts/eval_baseline.py` scaffolds.
- Reproducibility checklist and experiment naming convention.
- One dry-run training job that completes end-to-end.

**Exit check**
- Same config + same seed reproduces similar metrics.

---

## Week 2 - Dataset Generation (ID Regime)
**Objectives**
- Build first usable 2D EP benchmark for in-distribution training/testing.

**Tasks**
- Generate scenarios: planar, centrifugal, stable spiral, breakup.
- Sweep key parameters (excitability/diffusion/restitution/stimulation).
- Export trajectories and aligned tensors for operator learning.
- Create train/val/test ID splits with no leakage.

**Deliverables**
- Versioned dataset manifest.
- Dataset loader returning `(a, u)` pairs with consistent normalization.
- Data quality report (missingness, outlier stats, trajectory lengths).

**Exit check**
- Can train on full ID set without loader/runtime failures.

---

## Week 3 - Baseline FNO
**Objectives**
- Establish unconditional surrogate baseline.

**Tasks**
- Implement/train FNO one-step and rollout modes.
- Log RMSE/MAE/relative RMSE across horizon.
- Measure latency and throughput vs solver.
- Add first qualitative plots (pred vs GT snapshots).

**Deliverables**
- Best FNO checkpoint + training curves.
- Baseline metrics table (ID only).
- Initial figure set for qualitative behavior.

**Exit check**
- Stable training, no divergence, reasonable rollout behavior.

---

## Week 4 - PINO Baseline + Physics Loss Calibration
**Objectives**
- Add physics-informed backbone and compare to FNO.

**Tasks**
- Implement physics residual, BC/IC penalty terms.
- Tune `lambda_data`, `lambda_phys`, `lambda_bc` for stable optimization.
- Compare FNO vs PINO on rollout and hard dynamic regions.

**Deliverables**
- Best PINO checkpoint.
- FNO vs PINO comparison table (ID + rollout horizon slices).
- Residual-vs-error correlation plot.

**Exit check**
- PINO gives measurable gain in physics consistency and/or rollout robustness.

---

## Week 5 - Shift Protocols and Evaluation Harness
**Objectives**
- Build shift-ready benchmark and robust evaluation scripts.

**Tasks**
- Create parameter-shift split.
- Create geometry-shift split (held-out shapes/meshes).
- Create long-horizon shift split (train short, test long).
- Standardize evaluation scripts and report templates.

**Deliverables**
- `scripts/eval_shift.py` producing comparable outputs for all methods.
- Shift benchmark card (what changes in each split).
- Frozen baseline performance on all splits.

**Exit check**
- One command can evaluate a checkpoint across all split families.

---

## Week 6 - Reliability Feature Pipeline (Global + Local)
**Objectives**
- Implement feature extraction used by rejectors.

**Tasks**
- Add uncertainty signals (ensemble/dropout variance).
- Add PDE residual maps and summary stats.
- Add rollout drift and temporal inconsistency features.
- Add OOD/embedding distance features.

**Deliverables**
- `features/` modules with cached feature export.
- Feature sanity report (distribution by safe/unsafe error bins).
- Correlation matrix: feature vs true error metrics.

**Exit check**
- Features are stable and informative enough for rejector learning.

---

## Week 7 - Global Rejector (First Selective Results)
**Objectives**
- Achieve working case-level defer policy.

**Tasks**
- Implement global rejector head and training loop.
- Define global risk targets (continuous preferred; binarized fallback).
- Train with coverage control objective.
- Calibrate thresholds for target coverage and target accepted risk.

**Deliverables**
- Global rejector checkpoint(s).
- Global risk-coverage curves.
- Accepted-error vs coverage plots on ID and shifts.

**Exit check**
- Global selective policy improves accepted risk over always-predict baseline.

---

## Week 8 - Global Baseline Comparisons + Calibration Study
**Objectives**
- Validate that physics-aware score is better than simple confidence-only.

**Tasks**
- Implement baselines: confidence-only, uncertainty-only, residual-only, OOD-only.
- Run matched-coverage and matched-risk comparisons.
- Perform threshold calibration stability analysis across splits.

**Deliverables**
- Main global selective comparison table.
- Ablation of feature groups for global policy.
- Calibration robustness figure.

**Exit check**
- Joint features clearly outperform at least 2 single-signal baselines.

---

## Week 9 - Local Rejector Design (Patchwise)
**Objectives**
- Implement core novelty: spatial-temporal selective defer.

**Tasks**
- Build local rejector head outputting patch risk map.
- Create local supervision targets (patch error map + safe/unsafe masks).
- Add mask smoothness regularization (TV/graph).
- Tune patch size and stride trade-off.

**Deliverables**
- Working local rejector training pipeline.
- Local mask quality metrics (IoU/Dice for unsafe regions).
- Visualization overlays (risk mask on voltage fields).

**Exit check**
- Rejected regions visually and quantitatively align with high-error zones.

---

## Week 10 - Oracle Local Studies (Headroom Quantification)
**Objectives**
- Separate selection quality from fallback quality.

**Tasks**
- Implement oracle global selector and oracle local selector.
- Implement oracle repair (replace rejected regions with trusted target).
- Compute headroom between learned and oracle selectors.

**Deliverables**
- Oracle comparison table (global/local).
- Error concentration and upper-bound gain plots.
- Clear diagnosis of selector bottlenecks.

**Exit check**
- Demonstrable headroom and evidence that local selection has strongest upside.

---

## Week 11 - Hybrid Fallback Routing (Practical)
**Objectives**
- Move from abstention metrics to actual hybrid simulator.

**Tasks**
- Implement fallback router and selective merger.
- Add practical fallback option (patch refinement or secondary surrogate).
- Benchmark full pipeline latency and deferred fraction.

**Deliverables**
- End-to-end hybrid inference path.
- Compute-risk Pareto data for always-predict / always-defer / hybrid.
- Stability check on long rollouts.

**Exit check**
- Hybrid curve shows useful risk reduction with moderate compute overhead.

---

## Week 12 - Local vs Global Defer Main Experiment
**Objectives**
- Produce core claim evidence for the paper.

**Tasks**
- Run matched fallback budget comparisons.
- Run matched accepted-risk comparisons.
- Evaluate on ID + parameter shift + geometry shift + long rollout shift.

**Deliverables**
- Main table: local defer vs global defer.
- Main figure: local/global risk-coverage-compute comparison.
- Case studies highlighting where local defer wins.

**Exit check**
- Local defer beats global defer in at least two key regimes.

---

## Week 13 - Full Ablation Block
**Objectives**
- Stress-test method and defend novelty.

**Tasks**
- Remove physics loss from backbone.
- Remove each feature group from rejector.
- Compare global-only vs local-only vs combined.
- Compare two-stage training vs joint fine-tuning.

**Deliverables**
- Comprehensive ablation table.
- Feature contribution bar/heatmap figure.
- Clear textual interpretation for discussion section.

**Exit check**
- Ablations support that performance depends on joint physics-aware selective design.

---

## Week 14 - Theory Alignment + Claim Validation
**Objectives**
- Ensure experiments directly support theoretical statements.

**Tasks**
- Map each proposition/theorem statement to a measurable metric.
- Add accepted-risk vs threshold monotonicity plots.
- Add residual-aware bound motivation evidence (empirical proxy).
- Prepare theorem assumptions/limitations text linked to observed behavior.

**Deliverables**
- Theory-to-experiment mapping section draft.
- Supporting plots for accepted-risk control story.
- Finalized claim checklist (what is proven empirically vs theoretically).

**Exit check**
- Every main claim is backed by at least one robust experiment.

---

## Week 15 - Paper Artifact Assembly
**Objectives**
- Freeze assets and produce publication-ready package.

**Tasks**
- Freeze final checkpoints and random seeds.
- Generate final figures/tables in camera-ready style.
- Build reproducibility appendix (commands, configs, hardware, runtime).
- Write methods/results/discussion drafts around finalized assets.

**Deliverables**
- Final figure directory and table exports.
- Results narrative draft.
- Reproducibility document.

**Exit check**
- Another person can run core results using provided commands/configs.

---

## Week 16 - Buffer + Risk Mitigation + Submission Lock
**Objectives**
- Resolve weak spots and finalize complete submission package.

**Tasks**
- Re-run unstable experiments or missing seeds.
- Improve failure analysis examples (false accept / false reject).
- Tighten abstract/introduction around strongest empirical story.
- Final audit of claims, limitations, and ethical/deployment framing.

**Deliverables**
- Final paper-ready experiment bundle.
- Failure analysis subsection complete.
- Submission checklist complete.

**Exit check**
- Project is internally consistent, reproducible, and defensible.

---

## Weekly Operating Cadence (Recommended)
- **Monday:** plan runs and lock hypotheses for the week.
- **Tuesday-Thursday:** execute experiments and monitor training quality.
- **Friday:** aggregate metrics, produce plots, document findings.
- **Weekend/overflow:** reruns and cleanup only if blocked.

---

## Risk Register and Mitigation
- **Risk:** Data generation bottleneck.
  - **Mitigation:** parallelize case generation; cache processed tensors.
- **Risk:** PINO unstable training.
  - **Mitigation:** warm-start from FNO; tune physics loss weights gradually.
- **Risk:** Rejector learns trivial reject-all behavior.
  - **Mitigation:** enforce target coverage penalty + calibration constraints.
- **Risk:** Local masks noisy/speckled.
  - **Mitigation:** patchwise supervision + smoothness regularizer + post-processing.
- **Risk:** Hybrid fallback too expensive.
  - **Mitigation:** strict defer budget and patch-priority routing.

---

## Minimal Paper-Ready Configuration (If Time Constrained)
- Use 2D EP only.
- Keep FNO + PINO backbones.
- Implement global + local rejectors with uncertainty/residual/drift features.
- Use oracle local repair first (practical fallback optional).
- Evaluate on ID, parameter shift, geometry shift, long rollout.
- Deliver 3 core figures: risk-coverage, compute-risk Pareto, qualitative local masks.

This minimum path is sufficient for a strong first submission narrative.
