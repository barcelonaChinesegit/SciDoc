#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/settings/local_env.sh" ]]; then
    source "$REPO_ROOT/settings/local_env.sh"
fi

GPU="${GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-${PY_MINICPM45:-python}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PDFQA_GPU="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  output/raw_result \
  output/parsed_result \
  output/logs \
  output/manifests

usage() {
    cat <<EOF2
Usage:
  GPU=2 bash run.sh <dataset|all> [max_samples]

Datasets:
  ordinary
  unanswerable
  reasoning
  cross_pdf
  all

Examples:
  GPU=2 bash run.sh ordinary 1
  GPU=2 bash run.sh reasoning 1
  GPU=2 bash run.sh cross_pdf 1
  GPU=2 bash run.sh all
EOF2
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
echo "MiniCPM-V-4.5 PDF-QA"
echo "GPU         : $GPU"
echo "Python      : $PYTHON_BIN"
echo "Target      : $TARGET"
echo "Max samples : $MAX_SAMPLES"
echo "Protocol    : DPI144 / contextfit-v4.1 / slice 9->1"
echo "Prompt      : shared pdfqa-prompt-v2-strict-min"
echo "Thinking    : OFF"
echo "Sampling    : OFF"
echo "============================================================"

for dataset_id in "${DATASETS[@]}"; do

    LOG_FILE="output/logs/${dataset_id}__minicpm_v45__gpu${GPU}.log"

    echo
    echo "============================================================"
    echo "[Dataset] $dataset_id"
    echo "============================================================"

    "$PYTHON_BIN" -u run_pdfqa.py \
        --dataset-id "$dataset_id" \
        --dry-run

    CMD=(
        "$PYTHON_BIN"
        -u
        run_pdfqa.py
        --dataset-id
        "$dataset_id"
    )

    if [[ "$MAX_SAMPLES" -gt 0 ]]; then
        CMD+=(--max-samples "$MAX_SAMPLES")
    fi

    echo "[Run] ${CMD[*]}"

    "${CMD[@]}" \
        2>&1 | tee -a "$LOG_FILE"

    echo "[Parse] $dataset_id"

    "$PYTHON_BIN" -u parse_pdfqa_results.py \
        --dataset-id "$dataset_id" \
        2>&1 | tee -a "$LOG_FILE"
done
