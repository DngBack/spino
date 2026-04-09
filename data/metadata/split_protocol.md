# Split Protocol (SPINO EP2D)

## Muc tieu
Tao split cho paper ma khong leakage va support day du ID + shift experiments.

---

## 1) ID split

- Don vi tach: `case_id`
- Rule:
  - 70% train, 15% val, 15% test (co the dieu chinh),
  - cung scenario distribution tuong doi can bang,
  - khong trung seed trong nhom case trung cau hinh.

---

## 2) Parameter shift split

- Y tuong:
  - Train tren parameter bins A,
  - Test tren bins B khong overlap.
- Rule:
  - Define bins ro rang trong generation config.
  - Cam overlap theo `parameter_set_id` hoac parameter interval.

Vi du:
- train `excitability in [0.10, 0.18]`
- test  `excitability in (0.18, 0.24]`

---

## 3) Geometry shift split

- Hold-out mot tap `geometry_id` hoan toan cho test.
- Rule:
  - geometry test khong xuat hien o train/val,
  - giu scenario mix o train/test co the so sanh.

---

## 4) Long-rollout shift split

- Train dung short windows (vi du 40-80 steps),
- Test rollout dai (200-400 steps) tren case test.

Rule:
- horizon train phai duoc luu ro trong train config,
- horizon eval dai phai co thong tin trong metadata/time.

---

## 5) Split file format de dung chung

Moi split file JSON:

```json
{
  "split_name": "split_v1_id",
  "dataset_version": "v1.0",
  "created_at": "2026-04-09T10:00:00Z",
  "train_case_ids": [],
  "val_case_ids": [],
  "test_case_ids": [],
  "notes": []
}
```

Voi shift splits, them:
- `test_shift_case_ids`
- `shift_definition`

---

## 6) Leakage checks bat buoc

Truoc khi train:
- Kiem tra giao train/test theo `case_id` = rong.
- Kiem tra giao train/test theo `(geometry_id, parameter_set_id, seed)` = rong.
- Kiem tra geo holdout dung cho geometry shift.
- Kiem tra parameter interval overlap = khong.

Neu fail bat ky check nao: split bi invalid.

