#!/usr/bin/env bash
# reproduce_public.sh — reproduce all results derived from public benchmarks.
#
# This script:
#   1. downloads OULAD and EdNet-KT1 (skips if already present),
#   2. runs the early-window preprocessing described in §2.6,
#   3. trains the 13-feature generic LSTM-Attention backbone,
#   4. reports the external-validation metrics of Table 9, and
#   5. runs the synthetic-cohort scale-out stress test of Table 11.
#
# The institutional dataset is privacy-restricted and is NOT downloaded.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

DATA_DIR="${ROOT}/data"
RESULTS_DIR="${ROOT}/results"
mkdir -p "${DATA_DIR}" "${RESULTS_DIR}"

echo "==> [1/5] downloading public benchmarks"
python -m src.evaluation.external_oulad   --download_dir "${DATA_DIR}/oulad"
python -m src.evaluation.external_ednet   --download_dir "${DATA_DIR}/ednet"

echo "==> [2/5] OULAD external validation (early-window, 13 generic features)"
python scripts/run_oulad_validation.py \
    --data_dir "${DATA_DIR}/oulad" \
    --output   "${RESULTS_DIR}/oulad_metrics.json"

echo "==> [3/5] EdNet-KT1 external validation (early-window, 13 generic features)"
python scripts/run_ednet_validation.py \
    --data_dir "${DATA_DIR}/ednet" \
    --output   "${RESULTS_DIR}/ednet_metrics.json"

echo "==> [4/5] synthetic scale-out stress test"
python synthetic_data/generate_synthetic_institutional_data.py \
    --n_students 100000 --n_semesters 4 \
    --output "${DATA_DIR}/synthetic_100k.csv"

python scripts/run_synthetic_scaleout.py \
    --input    "${DATA_DIR}/synthetic_100k.csv" \
    --workers  1 2 4 8 \
    --output   "${RESULTS_DIR}/scaleout_table.csv"

echo "==> [5/5] figure regeneration"
python scripts/generate_figures.py --results_dir "${RESULTS_DIR}"

echo ""
echo "Done. Results written to ${RESULTS_DIR}/"
ls -lh "${RESULTS_DIR}"
