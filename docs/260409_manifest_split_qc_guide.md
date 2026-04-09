# Manifest, Splits, and QC Guide

## 1) Build manifest

Tao manifest tu cac file trong `data/metadata/case_metadata/*.json`.

```bash
python3 scripts/build_manifest.py \
  --metadata-dir data/metadata/case_metadata \
  --output-manifest data/metadata/dataset_manifest.v1.0.json \
  --dataset-name spino_ep2d \
  --dataset-version v1.0 \
  --generation-config-path data/metadata/generation_config.v1.json
```

## 2) Build split files

Sinh 4 split files: ID, parameter shift, geometry shift, long rollout.

```bash
python3 scripts/build_splits.py \
  --manifest data/metadata/dataset_manifest.v1.0.json \
  --output-dir data/splits \
  --dataset-version v1.0 \
  --seed 42 \
  --param-key excitability \
  --param-threshold 0.18 \
  --geometry-holdout-fraction 0.2
```

## 3) Run data QC

Chay QC metadata + split leakage:

```bash
python3 scripts/data_qc.py \
  --manifest data/metadata/dataset_manifest.v1.0.json
```

Chay them kiem tra tensor content (`NaN/Inf/shape`):

```bash
python3 scripts/data_qc.py \
  --manifest data/metadata/dataset_manifest.v1.0.json \
  --check-tensors
```

## 4) Pipeline khuyen nghi

1. generate case metadata  
2. `build_manifest.py`  
3. `build_splits.py`  
4. `data_qc.py`  
5. `visualize_and_analyze_data.py`

Neu buoc 4 fail, khong nen train main experiments.

