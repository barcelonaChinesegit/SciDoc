#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/settings/local_env.sh" ]]; then
    source "$REPO_ROOT/settings/local_env.sh"
fi

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"

cd "$SCRIPT_DIR"

PY="${PY:-${PY_MINICPM26:-python}}"
GPU="${GPU:-2}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PDFQA_GPU="$GPU"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  output/raw_result \
  output/parsed_result \
  output/manifests \
  output/logs

DATASET="${1:-}"
MAX_SAMPLES="${2:-0}"

usage() {
  echo "Usage:"
  echo "  GPU=2 bash run.sh ordinary [max_samples]"
  echo "  GPU=2 bash run.sh unanswerable [max_samples]"
  echo "  GPU=2 bash run.sh reasoning [max_samples]"
  echo "  GPU=2 bash run.sh cross_pdf [max_samples]"
  echo "  GPU=2 bash run.sh all [max_samples]"
}

run_one() {
  local ds="$1"
  local log="output/logs/${ds}__minicpm_v26__gpu${GPU}.log"

  {
    echo "============================================================"
    echo "MiniCPM-V-2.6 | ${ds}"
    echo "GPU=${GPU}"
    echo "Python=${PY}"
    echo "Started=$(date -Is)"
    echo "============================================================"

    "$PY" -u run_pdfqa.py \
      --dataset-id "$ds" \
      --dry-run

    ARGS=(
      --dataset-id "$ds"
    )

    if [[ "$MAX_SAMPLES" -gt 0 ]]; then
      ARGS+=(
        --max-samples "$MAX_SAMPLES"
      )
    fi

    "$PY" -u run_pdfqa.py \
      "${ARGS[@]}"

    "$PY" -u parse_pdfqa_results.py \
      --dataset-id "$ds"

    echo "============================================================"
    echo "Finished=$(date -Is)"
    echo "============================================================"
  } 2>&1 | tee "$log"
}

case "$DATASET" in
  ordinary|unanswerable|reasoning|cross_pdf)
    run_one "$DATASET"
    ;;

  all)
    for ds in \
      ordinary \
      unanswerable \
      reasoning \
      cross_pdf
    do
      run_one "$ds"
    done
    ;;

  *)
    usage
    exit 2
    ;;
esac
