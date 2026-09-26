#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/settings/local_env.sh" ]]; then
    source "$REPO_ROOT/settings/local_env.sh"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-${PY_CLAUDE:-python}}"
TARGET="${1:-all}"
MAX_NEW="${2:--1}"

mkdir -p output/raw_result output/parsed_result output/manifests output/logs output/render_cache

case "$TARGET" in
  ordinary) DATASETS=(ordinary) ;;
  unanswerable) DATASETS=(unanswerable) ;;
  reasoning) DATASETS=(reasoning) ;;
  cross_pdf) DATASETS=(cross_pdf) ;;
  all) DATASETS=(ordinary unanswerable reasoning cross_pdf) ;;
  *)
    echo "[ERROR] target must be ordinary|unanswerable|reasoning|cross_pdf|all"
    exit 2
    ;;
esac

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "[ERROR] ANTHROPIC_API_KEY is not set"
  exit 3
fi

echo "============================================================"
echo "Claude-Sonnet-5 PDF-QA"
echo "Python      : $PY"
echo "Model       : ${CLAUDE_MODEL:-claude-sonnet-5}"
echo "Endpoint    : ${CLAUDE_BASE_URL:-https://api.aicodemirror.ai/api/claudecode}"
echo "Target      : $TARGET"
echo "Max new QA  : $MAX_NEW"
echo "DPI         : 144"
echo "Whole PDF   : YES"
echo "Repair      : OFF"
echo "Thinking    : provider/model default"
echo "Prompt      : pdfqa-prompt-v2-strict-min"
echo "============================================================"

for DS in "${DATASETS[@]}"; do
  LOG="output/logs/${DS}__claude-sonnet-5.log"
  CMD=(
    "$PY" -u run_pdfqa.py
    --datasets "$DS"
    --pdf-dpi 144
    --request-timeout 1800
  )

  if [[ "$MAX_NEW" -gt 0 ]]; then
    CMD+=(--max-new-qa "$MAX_NEW")
  fi

  if [[ "${CLAUDE_DISABLE_THINKING:-0}" == "1" ]]; then
    CMD+=(--disable-thinking)
  fi

  if [[ "${CLAUDE_USE_ENV_PROXY:-0}" == "1" ]]; then
    CMD+=(--use-env-proxy)
  fi

  echo
  echo "[Run] ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee -a "$LOG"

  echo "[Parse] $DS"
  "$PY" -u parse_pdfqa_results.py --dataset-id "$DS" 2>&1 | tee -a "$LOG"
done
