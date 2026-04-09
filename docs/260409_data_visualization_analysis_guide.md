# Data Visualization and Analysis Guide

## Muc tieu
Script nay giup ban:
- tong hop thong ke dataset tu manifest + metadata,
- tao bang CSV de phan tich nhanh,
- tao cac hinh visualization de dua vao notes/paper.

Script:
- `scripts/visualize_and_analyze_data.py`

---

## 1) Chay nhanh voi template hien tai

```bash
python scripts/visualize_and_analyze_data.py \
  --manifest data/metadata/dataset_manifest.template.json \
  --output-dir outputs/data_analysis
```

Output:
- `outputs/data_analysis/cases_table.csv`
- `outputs/data_analysis/summary.json`
- `outputs/data_analysis/report.md`
- `outputs/data_analysis/figures/*.png`

---

## 2) Chay voi manifest that su (khi da generate data)

```bash
python scripts/visualize_and_analyze_data.py \
  --manifest data/metadata/dataset_manifest.v1.0.json \
  --output-dir outputs/data_analysis_v1
```

---

## 3) Cac hinh se duoc tao

- `scenario_counts.png`
- `split_counts.png`
- `resolution_counts.png`
- `scenario_split_heatmap.png`
- `parameter_histograms.png`

---

## 4) Cach doc ket qua nhanh

- `summary.json`: thong ke tong quan va canh bao metadata thieu.
- `cases_table.csv`: bang full de filter/pivot.
- `report.md`: tom tat de copy vao lab notes/paper draft.

---

## 5) Luu y

- Neu manifest co `metadata_path` tro den file chua ton tai, script van chay va bao `metadata_exists=false`.
- So hinh histogram tham so co the gioi han bang:

```bash
python scripts/visualize_and_analyze_data.py \
  --manifest data/metadata/dataset_manifest.v1.0.json \
  --output-dir outputs/data_analysis_v1 \
  --max-parameter-plots 12
```

