#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/settings/local_env.sh" ]]; then
    source "$REPO_ROOT/settings/local_env.sh"
fi


# ============================================================
# InternVL3.5-8B defaults
# ============================================================

GPU="${GPU:-2}"

PYTHON_BIN="${PYTHON_BIN:-${PY_INTERNVL35:-python}}"



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PDFQA_GPU="$GPU"

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
    cat <<EOF2

Usage:

  bash run.sh <dataset|all> [max_samples]

Datasets:

  ordinary
  unanswerable
  reasoning
  cross_pdf
  all

Examples:

  # One sample on physical GPU 2
  bash run.sh ordinary 1

  # One reasoning sample
  bash run.sh reasoning 1

  # Full ordinary dataset
  bash run.sh ordinary

  # All four datasets
  bash run.sh all

Environment overrides:

  GPU=2
  PYTHON_BIN=$PYTHON_BIN

Example:

  GPU=2 bash run.sh cross_pdf 1

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
echo "InternVL3.5-8B PDF-QA"
echo "GPU         : $GPU"
echo "Python      : $PYTHON_BIN"
echo "Target      : $TARGET"
echo "Max samples : $MAX_SAMPLES"
echo "Protocol    : dpi144 / input448 / docbudget96-a40safe-v2"
echo "Thumbnail   : OFF"
echo "Started     : $(date -Is)"
echo "============================================================"


# ============================================================
# Run datasets
# ============================================================

for dataset_id in "${DATASETS[@]}"; do

    LOG_FILE="output/logs/${dataset_id}__internvl35_8b__gpu${GPU}.log"

    echo
    echo "============================================================"
    echo "[Dataset] $dataset_id"
    echo "============================================================"


    # --------------------------------------------------------
    # Preflight
    # --------------------------------------------------------

    "$PYTHON_BIN" -u run_pdfqa.py \
        --dataset-id "$dataset_id" \
        --dry-run


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    echo "[Parse] $dataset_id"

    "$PYTHON_BIN" -u parse_pdfqa_results.py \
        --dataset-id "$dataset_id" \
        2>&1 | tee -a "$LOG_FILE"

done


echo
echo "============================================================"
echo "Finished: $(date -Is)"
echo "============================================================"
