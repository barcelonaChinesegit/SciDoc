#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# Defaults
# ============================================================

GPU="${GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1


mkdir -p \
  output/raw_result \
  output/parsed_result \
  output/logs \
  output/manifests


# ============================================================
# Usage
# ============================================================

usage() {
    cat <<EOF

Usage:

  bash run.sh <dataset|all> [max_samples]

Datasets:

  ordinary
  unanswerable
  reasoning
  cross_pdf
  all

Examples:

  # Run 1 ordinary sample on default GPU 2
  bash run.sh ordinary 1

  # Run 1 reasoning sample
  bash run.sh reasoning 1

  # Run full ordinary dataset
  bash run.sh ordinary

  # Run all four datasets
  bash run.sh all

  # Explicit GPU
  GPU=2 bash run.sh cross_pdf 1

EOF
}


if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi


TARGET="$1"
MAX_SAMPLES="${2:-0}"


case "$TARGET" in

    all)
        DATASETS=(
            ordinary
            unanswerable
            reasoning
            cross_pdf
        )
        ;;

    ordinary|unanswerable|reasoning|cross_pdf)
        DATASETS=("$TARGET")
        ;;

    *)
        echo "[ERROR] Unknown dataset: $TARGET"
        usage
        exit 1
        ;;
esac


echo "============================================================"
echo "Qwen3-VL-8B PDF-QA"
echo "GPU         : $GPU"
echo "Target      : $TARGET"
echo "Max samples : $MAX_SAMPLES"
echo "Started     : $(date -Is)"
echo "============================================================"


for dataset_id in "${DATASETS[@]}"; do

    LOG_FILE="output/logs/${dataset_id}__gpu${GPU}.log"

    echo
    echo "============================================================"
    echo "[Dataset] $dataset_id"
    echo "============================================================"

    # Preflight
    "$PYTHON_BIN" -u run_pdfqa.py \
        --dataset-id "$dataset_id" \
        --dry-run

    # Inference
    CMD=(
        "$PYTHON_BIN"
        -u
        run_pdfqa.py
        --dataset-id
        "$dataset_id"
    )

    if [[ "$MAX_SAMPLES" -gt 0 ]]; then
        CMD+=(
            --max-samples
            "$MAX_SAMPLES"
        )
    fi

    echo "[Run] ${CMD[*]}"

    "${CMD[@]}" \
        2>&1 | tee -a "$LOG_FILE"

    # Parse
    "$PYTHON_BIN" -u parse_pdfqa_results.py \
        --dataset-id "$dataset_id" \
        2>&1 | tee -a "$LOG_FILE"

done


echo
echo "============================================================"
echo "Finished: $(date -Is)"
echo "============================================================"
