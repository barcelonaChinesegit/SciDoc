#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/settings/local_env.sh" ]]; then
    source "$REPO_ROOT/settings/local_env.sh"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${PY_GEMMA3:-python}}"
MODEL_DIR="$WORKSPACE_ROOT/models/Gemma-3-27B-IT"

TARGET="${1:-all}"
MAX_SAMPLES="${2:--1}"

GPU="${GPU:-2}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p \
    output/raw_result \
    output/parsed_result \
    output/manifests \
    output/logs \
    output/render_cache

case "$TARGET" in
    ordinary)
        DATASETS=(ordinary)
        ;;
    unanswerable)
        DATASETS=(unanswerable)
        ;;
    reasoning)
        DATASETS=(reasoning)
        ;;
    cross_pdf)
        DATASETS=(cross_pdf)
        ;;
    all)
        DATASETS=(
            ordinary
            unanswerable
            reasoning
            cross_pdf
        )
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo
        echo "Usage:"
        echo "  GPU=2 bash run.sh ordinary [max_samples]"
        echo "  GPU=2 bash run.sh unanswerable [max_samples]"
        echo "  GPU=2 bash run.sh reasoning [max_samples]"
        echo "  GPU=2 bash run.sh cross_pdf [max_samples]"
        echo "  GPU=2 bash run.sh all"
        exit 2
        ;;
esac

echo "============================================================"
echo "Gemma-3-27B-IT PDF-QA"
echo "GPU         : $GPU"
echo "Python      : $PYTHON_BIN"
echo "Model       : $MODEL_DIR"
echo "Target      : $TARGET"
echo "Max samples : $MAX_SAMPLES"
echo "Protocol    : DPI144 / JPEG95 / whole-PDF / no repair"
echo "Prompt      : shared pdfqa-prompt-v2-strict-min"
echo "============================================================"

for dataset_id in "${DATASETS[@]}"
do
    echo
    echo "============================================================"
    echo "[Dataset] $dataset_id"
    echo "============================================================"

    LOG_FILE="output/logs/${dataset_id}__gemma3_27b__dpi144.log"

    "$PYTHON_BIN" -u run_pdfqa.py \
        --dataset-id "$dataset_id" \
        --model-path "$MODEL_DIR" \
        --dry-run \
        2>&1 | tee -a "$LOG_FILE"

    CMD=(
        "$PYTHON_BIN"
        -u
        run_pdfqa.py
        --dataset-id
        "$dataset_id"
        --model-path
        "$MODEL_DIR"
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

    echo "[Parse] $dataset_id"

    "$PYTHON_BIN" -u parse_pdfqa_results.py \
        --dataset-id "$dataset_id" \
        2>&1 | tee -a "$LOG_FILE"
done

echo
echo "============================================================"
echo "Finished: $(date -Is)"
echo "============================================================"
