# Case Metadata Schema (SPINO)

## Muc dich
Schema thong nhat cho moi case trajectory de:
- reproducible generation,
- split dung cach,
- training/evaluation khong leakage,
- truy vet nhanh khi viet paper.

---

## Required fields

```json
{
  "case_id": "string",
  "dataset_version": "string",
  "generator": "opencarp|synthetic",
  "scenario_type": "planar_wave|centrifugal_wave|stable_spiral|spiral_breakup|...",
  "geometry_id": "string",
  "mesh_or_grid": {
    "type": "grid|mesh",
    "shape": [128, 128],
    "resolution_tag": "r128",
    "dx": 0.25
  },
  "time": {
    "dt": 0.1,
    "num_steps": 300,
    "t_start": 0.0
  },
  "seed": 12345,
  "parameters": {
    "diffusion": 0.001,
    "conductivity": 1.0,
    "excitability": 0.15,
    "restitution": 0.2
  },
  "stimulus": {
    "type": "point|line|region",
    "location": [0.25, 0.75],
    "start_step": 5,
    "duration_steps": 10,
    "amplitude": 1.0
  },
  "state_channels": {
    "input_channels": ["V0", "R0"],
    "output_channels": ["V", "R"]
  },
  "file_paths": {
    "raw_case_path": "data/raw/.../case_x",
    "processed_tensor_path": "data/processed/tensors/case_x.npz"
  },
  "split_tags": ["train_id"],
  "quality_flags": {
    "has_nan": false,
    "has_inf": false,
    "is_duplicate_hash": false,
    "physics_warning": false
  },
  "hashes": {
    "processed_tensor_sha256": "string"
  },
  "created_at": "2026-04-09T10:00:00Z"
}
```

---

## Field constraints

- `case_id`: duy nhat toan dataset version.
- `dataset_version`: format khuyen nghi `vX.Y`.
- `seed`: integer >= 0.
- `shape`: bat buoc hop voi tensor that su.
- `split_tags`: cho phep nhieu tag, nhung test tag khong duoc overlap voi train.
- `processed_tensor_path`: phai ton tai truoc khi add vao manifest.

---

## Optional fields (khuyen nghi)

- `parameter_bin`: nhan bin de lam split parameter shift.
- `geometry_family`: nhom hinh hoc.
- `solver_stats`: runtime, convergence, residual summary.
- `event_annotations`: activation time map, spiral tip track, breakup intervals.

---

## Naming convention

`case_id` khuyen nghi:

`spino_<scenario>_<geometry>_<paramset>_s<seed>_r<res>_t<num_steps>`

Vi du:

`spino_stable_spiral_geo07_p03_s1024_r128_t300`

