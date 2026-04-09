# SPINO Implementation & Experiment Plan (Full Detailed Docs)

This document converts the previous plan into a **fully structured, implementation-ready documentation**. It is designed to be directly translated into a working repo and a NeurIPS-quality experiment pipeline.

---

# 1. Project Scope (Strict Definition)

## 1.1 Objective

Build a system that:

- Learns a **Physics-Informed Neural Operator (PINO)** for cardiac electrophysiology
- Learns a **rejector (global + local)**
- Performs **selective prediction (predict or defer)**
- Supports **hybrid simulation (surrogate + solver)**

## 1.2 Core Claim

> Physics-aware selective prediction improves the **risk–coverage–compute trade-off** compared to:

- unconditional surrogate
- uncertainty-only rejection

---

# 2. System Architecture (End-to-End)

## 2.1 High-level pipeline

```
Input a
   ↓
PINO Operator → prediction u_hat
   ↓
Feature Extractor → (uncertainty, residual, drift, OOD)
   ↓
Rejector
   ↓
Decision:
   - Accept → keep u_hat
   - Reject → send to fallback H
   ↓
Merge → final u_tilde
```

---

# 3. Repository Specification (Final Form)

```
spino/
  configs/
  data/
  simulators/
  datasets/
  models/
  losses/
  features/
  trainers/
  evaluators/
  calibration/
  scripts/
  utils/
```

---

# 4. Module-Level Design

## 4.1 Dataset

### Input tensor

```
[B, C_in, T_in, H, W]
```

Where:
- C_in = voltage + recovery + positional + parameters

### Output tensor

```
[B, C_out, T_out, H, W]
```

---

## 4.2 Operator Backbone (PINO/FNO)

### Architecture

- Lift layer
- 4 FNO blocks
- Projection layer

### Output

```
u_hat ∈ R[B, 2, T, H, W]
```

---

## 4.3 Feature Extraction

### Global features

- mean residual
- max residual
- rollout drift
- ensemble variance
- latent distance
- OOD score

### Local features

```
[B, C_feat, T, H, W]
```

Where C_feat includes:
- residual map
- variance map
- gradient map
- drift map

---

## 4.4 Rejector

### Global Rejector

MLP:

```
[B, F] → [B, 1]
```

### Local Rejector

U-Net style:

```
[B, C_feat, T, H, W] → [B, 1, T', H', W']
```

---

## 4.5 Fallback Module

Initial version:

- Oracle replacement

Advanced:

- Patch PDE solver
- Secondary model

---

# 5. Training Pipeline

## 5.1 Stage 1 — Operator

Loss:

- Data loss
- Physics loss

Output:

- Stable surrogate

---

## 5.2 Stage 2 — Target Construction

### Global

```
risk = rollout RMSE
```

### Local

```
risk_map = local error
```

---

## 5.3 Stage 3 — Rejector Training

Loss:

```
L = L_risk + λ_cov + λ_smooth
```

---

## 5.4 Stage 4 — Joint Fine-tuning

- Unfreeze last layers
- Optimize combined objective

---

# 6. Calibration

Modes:

- Fixed coverage
- Fixed risk
- Fixed compute budget

---

# 7. Dataset Plan

## 7.1 Scenarios

- planar
- centrifugal
- spiral
- spiral breakup

## 7.2 Variations

- parameters
- geometry
- stimulation
- resolution

## 7.3 Splits

- ID
- parameter shift
- geometry shift
- rollout shift

---

# 8. Metrics

## 8.1 Predictive

- RMSE
- MAE
- rollout error

## 8.2 Selective

- coverage
- selective risk
- AURC

## 8.3 Local

- localized coverage
- localized risk
- IoU / Dice

## 8.4 Efficiency

- latency
- defer ratio

---

# 9. Experiments

## 9.1 Block A — Backbone

Compare:
- FNO
- PINO

---

## 9.2 Block B — Global Selective

Compare:
- uncertainty
- residual
- joint

---

## 9.3 Block C — Local Selective

Compare:
- global vs local
- oracle vs learned

---

# 10. Tables (Final)

## Table 1 — Backbone

| Method | ID RMSE | Rollout RMSE | Shift RMSE | Latency |

## Table 2 — Global Selective

| Method | Coverage | Accepted RMSE | AUC |

## Table 3 — Matched Risk

| Method | Coverage | Latency |

## Table 4 — Local

| Method | Local Coverage | Accepted RMSE | IoU |

## Table 5 — Hybrid

| Method | RMSE | Compute | Speedup |

## Table 6 — Shift

| Method | Param | Geometry | Rollout |

## Table 7 — Ablation

| Setting | Accepted RMSE | IoU |

---

# 11. Figures

## Fig 1 — Architecture

## Fig 2 — Qualitative Maps

## Fig 3 — Risk-Coverage

## Fig 4 — Compute-Risk

## Fig 5 — Error Concentration

## Fig 6 — Rollout Stability

## Fig 7 — Shift

## Fig 8 — Ablation Heatmap

---

# 12. Oracle Studies

- Oracle global
- Oracle local
- Oracle repair

---

# 13. Timeline (12 Weeks)

## Week 1–2
Dataset + baseline

## Week 3–4
Evaluation + shifts

## Week 5–6
Global rejector

## Week 7–8
Local rejector

## Week 9–10
Hybrid routing

## Week 11–12
Finalize paper

---

# 14. Success Criteria

Paper is strong if:

1. Joint rejector > uncertainty
2. Local > global
3. Hybrid improves compute-risk

---

# 15. Minimal First Submission

- 2D EP
- PINO backbone
- global + local rejector
- oracle fallback
- shift experiments

---

# 16. End Goal

A NeurIPS-level paper showing:

> Reliable, physics-aware selective simulation is necessary for deployable biomedical digital twins.

