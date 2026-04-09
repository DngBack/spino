# SPINO Data Blueprint (Paper-Ready)

## 1) Muc tieu phan du lieu

Muc tieu la tao bo du lieu mo phong cardiac EP 2D du manh de:
- train backbone (`FNO`, `PINO`),
- train rejector global/local cho selective defer,
- danh gia duoi cac loai shift,
- viet paper co bang chung day du (tables + figures + ablations).

Phien ban dau uu tien du lieu mo phong co kiem soat, de reproducible va de viet phan methods.

---

## 2) Don vi du lieu va dinh dang chuan

Moi mau du lieu la mot **case trajectory**.

- Input operator `a`:
  - `geometry` (mask hoac mesh id + geometry fields),
  - `parameter map/scalars` (conductivity, excitability, restitution...),
  - `stimulation spec` (site, start, duration, amplitude),
  - `initial state` (`V0`, co the kem bien recovery ban dau).
- Output operator `u`:
  - trajectory `V(t, x, y)` va bien phu (neu co),
  - tensor de xuat: `[T, H, W, C]`.

De training thuan loi:
- Luu file tensor theo `.npz`/`.npy` (hoac `.pt` neu dung PyTorch native).
- Metadata tung case luu JSON rieng.
- Manifest tong hop luu toan bo case va split.

---

## 3) Cau truc thu muc du lieu de xai ngay

```text
data/
  raw/
    opencarp_runs/
    synthetic_runs/
  processed/
    tensors/
      <case_id>.npz
    features_cache/
  splits/
    split_v1_id.json
    split_v1_param_shift.json
    split_v1_geometry_shift.json
    split_v1_long_rollout.json
  metadata/
    case_metadata/
      <case_id>.json
    dataset_manifest.v1.json
    generation_config.v1.json
```

Quy uoc:
- `raw/` = output goc tu solver/generator.
- `processed/tensors/` = dau vao train/eval model.
- `splits/` = file split co version.
- `metadata/case_metadata/` = schema thong nhat tung case.

---

## 4) Scenario bat buoc cho paper v1

Can toi thieu 4 scenario:
1. `planar_wave`
2. `centrifugal_wave`
3. `stable_spiral`
4. `spiral_breakup`

Ly do:
- 1-2 la easy/regular regimes,
- 3-4 la hard/unstable regimes,
- rejector moi hoc duoc "khi nao defer".

---

## 5) Parameter space (de khai bao trong generation config)

Nhom tham so toi thieu:
- `diffusion` / `conductivity`
- `excitability`
- `restitution-related parameter`
- `stimulation` (location, amplitude, duration, start_time)

Khuyen nghi dat grid:
- 8-12 settings / scenario,
- 20-40 seeds / setting,
- 2-3 resolutions (`64x64`, `128x128`, tuy tai nguyen them `256x256`).

---

## 6) Quy mo de co "nguyen lieu day du"

Moc khuyen nghi:
- So case tong: `~800` den `~2500` (tuy compute),
- Timesteps: `200-400` / case,
- Co du split shift de lam main claims.

Moc toi thieu de paper van on:
- `~500` case da dang + shift protocols ro rang.

---

## 7) Split protocol cho paper

Can co 4 split chinh:

1) **ID split**
- Train/val/test cung distribution, khac seed.

2) **Parameter shift**
- Test tren vung tham so khong xuat hien trong train.

3) **Geometry shift**
- Hold-out geometry IDs, khong trung voi train.

4) **Long-rollout shift**
- Train horizon ngan (vi du 40-80 steps), test horizon dai (200-400).

Rule quan trong:
- Khong leakage theo `(geometry_id, parameter_bin, seed)`.
- Split phai deterministic va versioned.

---

## 8) Metadata bat buoc moi case

Moi case can co:
- `case_id`
- `generator` (`opencarp` hoac `synthetic`)
- `scenario_type`
- `geometry_id`
- `mesh_or_grid`
- `resolution`
- `dt`, `dx`, `num_steps`
- `seed`
- `parameters` (dict)
- `stimulus` (dict)
- `file_paths` (`raw`, `processed_tensor`)
- `split_tags` (list)
- `quality_flags` (dict)

Schema chi tiet nam trong `data/metadata/schema_case_metadata.md`.

---

## 9) Data quality gates (truoc khi train)

Case bi loai neu:
- NaN/Inf trong tensor,
- voltage vuot nguong phi vat ly (theo nguong ban xac dinh),
- trajectory qua ngan,
- metadata thieu truong bat buoc,
- mismatch shape giua metadata va tensor.

Case can canh bao:
- residual qua cao toan horizon,
- stimulation khong tao response mong doi,
- duplicate case hash.

---

## 10) Metrics-level support tu phan du lieu

Du lieu phai ho tro tinh cac metric:
- Predictive: RMSE/MAE/rollout error.
- Selective: coverage, selective risk, local selective risk.
- Physics: residual norm, drift.
- Efficiency: deferred fraction, latency.

Vi vay metadata can luu:
- thong tin horizon,
- scenario labels,
- thong so de tao shift slices.

---

## 11) Dataset versions de viet paper

Dat version ro:
- `v0.1`: smoke-test (nho)
- `v0.2`: baseline trainable
- `v1.0`: paper main
- `v1.1`: camera-ready rerun/fix

Moi version phai kem:
- manifest JSON,
- split JSON,
- generation config JSON,
- changelog ngan.

---

## 12) Reproducibility package cho appendix

Can chot:
- seed list (data generation + training),
- exact generation config,
- hash/size thong ke dataset,
- command tao du lieu va command train/eval,
- mapping tu bang/figure -> dataset version + split + run id.

---

## 13) Ke hoach tao du lieu theo 3 dot

Dot 1 (1-2 ngay):
- Tao `v0.1` de chay dry-run end-to-end.

Dot 2 (3-5 ngay):
- Mo rong thanh `v0.2` cho baseline FNO/PINO + shift scripts.

Dot 3 (1-2 tuan):
- Chot `v1.0` day du cho rejector global/local + hybrid experiments.

Neu can cat pham vi:
- Giu 4 scenario, giam so seeds truoc, khong bo split shift.

