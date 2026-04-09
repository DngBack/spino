# Data QC Checklist (Before Training)

Checklist nay dung truoc khi chot dataset version (`v0.2`, `v1.0`...).

## A) File Integrity
- [ ] Tat ca `processed_tensor_path` trong manifest ton tai.
- [ ] Tat ca metadata JSON parse duoc.
- [ ] Khong co file rong bat thuong.
- [ ] Hash tensor duoc tinh va luu (`sha256`).

## B) Tensor Validity
- [ ] Khong co NaN.
- [ ] Khong co Inf.
- [ ] Shape dung voi metadata (`num_steps`, `resolution`, channels).
- [ ] Gia tri nam trong khoang duoc chap nhan (theo domain).

## C) Metadata Completeness
- [ ] Day du required fields theo `schema_case_metadata.md`.
- [ ] `case_id` duy nhat.
- [ ] `scenario_type`, `geometry_id`, `seed` hop le.
- [ ] `split_tags` hop voi split files.

## D) Split Validity
- [ ] Train/val/test khong overlap `case_id`.
- [ ] Khong leakage theo `(geometry_id, parameter_set_id, seed)`.
- [ ] Geometry shift holdout dung protocol.
- [ ] Parameter shift dung interval/bin protocol.

## E) Distribution Coverage
- [ ] Co du 4 scenarios toi thieu.
- [ ] Moi scenario co du so case toi thieu dat ra.
- [ ] Co it nhat 2 resolutions neu compute cho phep.
- [ ] Co du hard regimes (`stable_spiral`, `spiral_breakup`).

## F) Paper Readiness
- [ ] Manifest co version + created_at + notes.
- [ ] Generation config duoc luu va versioned.
- [ ] Mapping table figure/table -> split -> dataset version san sang.
- [ ] Da co changelog dataset ngan.

## Gate de cho phep train main experiments
- Dataset chi duoc "go" khi tat ca muc A-D pass.
- Muc E-F co the pass theo tung phase, nhung bat buoc pass truoc khi chot paper main.

