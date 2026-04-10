#!/usr/bin/env bash
# Full SPINO pipeline:.train + eval từ manifest/split v1.0 → tables + figures.
# Chạy từ thư mục gốc repo:  bash scripts/run_full_paper_pipeline_v1.sh
#
# Biến môi trường (tùy chọn):
#   DEVICE=cuda|cpu     (mặc định: cuda nếu có PyTorch CUDA, ngược lại cpu)
#   SKIP_QC=1           bỏ qua data_qc + visualize
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAN="${MAN:-data/metadata/dataset_manifest.v1.0.json}"
SPLIT_ID="${SPLIT_ID:-data/splits/split_v1.0_id.json}"

if [[ ! -f "$MAN" ]]; then
  echo "Thiếu manifest: $MAN — chạy build_manifest + build_splits sau khi generate data."
  exit 1
fi
if [[ ! -f "$SPLIT_ID" ]]; then
  echo "Thiếu split: $SPLIT_ID"
  exit 1
fi

# Kiến trúc FNO/PINO thống nhất với eval_week10–12 (hardcode 24/3/12 trong các script đó).
W="24"
D="3"
M="12"
ARCH_FNO=(--width "$W" --depth "$D" --modes "$M")
ARCH_PINO=(--width "$W" --depth "$D" --modes "$M")

if [[ -z "${DEVICE:-}" ]]; then
  if python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="outputs/paper_pipeline_${STAMP}"
mkdir -p "$OUT"

echo "[INFO] ROOT=$ROOT DEVICE=$DEVICE OUT=$OUT" | tee "$OUT/run_log.txt"

# --- Tuỳ chọn: QC + plots dữ liệu ---
if [[ "${SKIP_QC:-0}" != "1" ]]; then
  mkdir -p "$OUT/week2_data_qc"
  python3 scripts/data_qc.py \
    --manifest "$MAN" \
    --repo-root "$ROOT" \
    --check-tensors 2>&1 | tee "$OUT/week2_data_qc/qc_report.txt" | tee -a "$OUT/run_log.txt"

  python3 scripts/visualize_and_analyze_data.py \
    --manifest "$MAN" \
    --output-dir "$OUT/week2_data_analysis" 2>&1 | tee -a "$OUT/run_log.txt"
fi

# --- Week 3: FNO ---
python3 scripts/train_baseline.py \
  --manifest "$MAN" \
  --split "$SPLIT_ID" \
  --output-dir outputs/week3_fno \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-3 \
  "${ARCH_FNO[@]}" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

FNO_DIR="$(ls -td outputs/week3_fno/*/ 2>/dev/null | head -n1)"
FNO_CKPT="${FNO_DIR}best_fno.pt"
FNO_HIST="${FNO_DIR}train_history.json"
[[ -f "$FNO_CKPT" ]] || { echo "Không thấy $FNO_CKPT"; exit 1; }

python3 scripts/eval_baseline.py \
  --manifest "$MAN" \
  --split "$SPLIT_ID" \
  --checkpoint "$FNO_CKPT" \
  --output-dir "$OUT/week3_fno_eval" \
  "${ARCH_FNO[@]}" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"
FNO_METRICS="$OUT/week3_fno_eval/metrics_summary.json"

# --- Week 4: PINO + so sánh FNO vs PINO (metrics ID lấy từ nhánh id trong shift eval) ---
python3 scripts/train_pino_baseline.py \
  --manifest "$MAN" \
  --split "$SPLIT_ID" \
  --output-dir outputs/week4_pino \
  --epochs 25 \
  --batch-size 16 \
  --lr 1e-3 \
  "${ARCH_PINO[@]}" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

PINO_DIR="$(ls -td outputs/week4_pino/*/ 2>/dev/null | head -n1)"
PINO_CKPT="${PINO_DIR}best_fno.pt"
PINO_HIST="${PINO_DIR}train_history.json"
[[ -f "$PINO_CKPT" ]] || { echo "Không thấy $PINO_CKPT"; exit 1; }

python3 scripts/eval_shift.py \
  --manifest "$MAN" \
  --checkpoint "$PINO_CKPT" \
  --model-type pino \
  --output-dir "$OUT/week4_pino_shift_for_compare" \
  "${ARCH_PINO[@]}" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"
PINO_METRICS="$OUT/week4_pino_shift_for_compare/per_split/id_summary.json"

python3 scripts/compare_fno_pino.py \
  --fno-metrics "$FNO_METRICS" \
  --pino-metrics "$PINO_METRICS" \
  --fno-history "$FNO_HIST" \
  --pino-history "$PINO_HIST" \
  --output-dir "$OUT/week4_comparison" 2>&1 | tee -a "$OUT/run_log.txt"

# --- Week 5: shift benchmark (FNO) ---
python3 scripts/eval_shift.py \
  --manifest "$MAN" \
  --checkpoint "$FNO_CKPT" \
  --model-type fno \
  --output-dir "$OUT/week5_shift_eval" \
  "${ARCH_FNO[@]}" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

# --- Week 6: reliability features ---
python3 scripts/build_reliability_features.py \
  --manifest "$MAN" \
  --id-split "$SPLIT_ID" \
  --checkpoint "$FNO_CKPT" \
  --model-type fno \
  --output-dir "$OUT/week6_features" \
  "${ARCH_FNO[@]}" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"
FEATURE_CSV="$OUT/week6_features/global_reliability_features.csv"
[[ -f "$FEATURE_CSV" ]] || { echo "Không thấy $FEATURE_CSV"; exit 1; }

# --- Week 7: global rejector ---
python3 scripts/train_global_rejector.py \
  --feature-csv "$FEATURE_CSV" \
  --id-split "$SPLIT_ID" \
  --output-dir outputs/week7_global_rejector \
  --epochs 200 \
  --batch-size 16 \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

GDIR="$(ls -td outputs/week7_global_rejector/*/ 2>/dev/null | head -n1)"
GLOBAL_CKPT="${GDIR}best_global_rejector.pt"
[[ -f "$GLOBAL_CKPT" ]] || { echo "Không thấy $GLOBAL_CKPT"; exit 1; }

python3 scripts/eval_global_rejector.py \
  --feature-csv "$FEATURE_CSV" \
  --manifest "$MAN" \
  --rejector-checkpoint "$GLOBAL_CKPT" \
  --target-coverage 0.8 \
  --output-dir "$OUT/week7_global_eval" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

# --- Week 8: global baselines ---
python3 scripts/eval_week8_global_baselines.py \
  --feature-csv "$FEATURE_CSV" \
  --manifest "$MAN" \
  --rejector-checkpoint "$GLOBAL_CKPT" \
  --output-dir "$OUT/week8_global_baselines" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

# --- Week 9: local rejector ---
python3 scripts/train_local_rejector.py \
  --manifest "$MAN" \
  --split "$SPLIT_ID" \
  --fno-checkpoint "$FNO_CKPT" \
  --output-dir outputs/week9_local_rejector \
  --patch-stride 4 \
  --epochs 50 \
  --batch-size 8 \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

LDIR="$(ls -td outputs/week9_local_rejector/*/ 2>/dev/null | head -n1)"
LOCAL_CKPT="${LDIR}best_local_rejector.pt"
[[ -f "$LOCAL_CKPT" ]] || { echo "Không thấy $LOCAL_CKPT"; exit 1; }

python3 scripts/eval_local_rejector.py \
  --manifest "$MAN" \
  --split "$SPLIT_ID" \
  --fno-checkpoint "$FNO_CKPT" \
  --local-checkpoint "$LOCAL_CKPT" \
  --patch-stride 4 \
  --output-dir "$OUT/week9_local_eval" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

# --- Week 10–12 ---
python3 scripts/eval_week10_oracle_studies.py \
  --manifest "$MAN" \
  --feature-csv "$FEATURE_CSV" \
  --global-checkpoint "$GLOBAL_CKPT" \
  --fno-checkpoint "$FNO_CKPT" \
  --local-checkpoint "$LOCAL_CKPT" \
  --patch-stride 4 \
  --target-coverage 0.75 \
  --output-dir "$OUT/week10_oracle" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

python3 scripts/eval_week11_hybrid.py \
  --manifest "$MAN" \
  --feature-csv "$FEATURE_CSV" \
  --fno-checkpoint "$FNO_CKPT" \
  --global-checkpoint "$GLOBAL_CKPT" \
  --local-checkpoint "$LOCAL_CKPT" \
  --patch-stride 4 \
  --output-dir "$OUT/week11_hybrid" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

python3 scripts/eval_week12_main.py \
  --manifest "$MAN" \
  --feature-csv "$FEATURE_CSV" \
  --fno-checkpoint "$FNO_CKPT" \
  --global-checkpoint "$GLOBAL_CKPT" \
  --local-checkpoint "$LOCAL_CKPT" \
  --patch-stride 4 \
  --coverages 0.5,0.65,0.8,0.9 \
  --max-val-cases 32 \
  --output-dir "$OUT/week12_main" \
  --device "$DEVICE" 2>&1 | tee -a "$OUT/run_log.txt"

# --- Tóm tắt đường dẫn ---
cat > "$OUT/RESULT_PATHS.txt" << EOF
Run bundle: $OUT
Manifest:   $MAN
Split ID:   $SPLIT_ID

FNO checkpoint:   $FNO_CKPT
PINO checkpoint:  $PINO_CKPT
Global rejector:  $GLOBAL_CKPT
Local rejector:   $LOCAL_CKPT
Feature CSV:      $FEATURE_CSV

Week 3 ID eval:     $OUT/week3_fno_eval/
Week 4 compare:     $OUT/week4_comparison/
Week 5 shift (FNO): $OUT/week5_shift_eval/
Week 6 features:    $OUT/week6_features/
Week 7 global eval: $OUT/week7_global_eval/
Week 8 baselines:   $OUT/week8_global_baselines/
Week 9 local eval:  $OUT/week9_local_eval/
Week 10 oracle:     $OUT/week10_oracle/
Week 11 hybrid:     $OUT/week11_hybrid/
Week 12 main:       $OUT/week12_main/
EOF

echo ""
echo "[OK] Hoàn tất. Đọc $OUT/RESULT_PATHS.txt"
cat "$OUT/RESULT_PATHS.txt"
