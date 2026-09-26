#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$REPO_ROOT/settings/local_env.sh" ]]; then
    source "$REPO_ROOT/settings/local_env.sh"
fi

cd "$(dirname "$0")"

PY="${PY:-${PY_DEEPSEEK:-python}}"
TARGET="${1:-all}"
MAX_NEW="${2:--1}"

MODEL="${DEEPSEEK_V_MODEL:-deepseek-v4-flash-vision-exp}"
BASE_URL="${DEEPSEEK_V_BASE_URL:-https://api.zhizengzeng.com/v1}"

if [[ -z "${ZHIZENGZENG_API_KEY:-}" ]]; then
  echo "[ERROR] ZHIZENGZENG_API_KEY is not set"
  exit 2
fi

mkdir -p \
  output/raw_result \
  output/parsed_result \
  output/manifests \
  output/logs \
  output/render_cache

case "$TARGET" in
  ordinary|unanswerable|reasoning|cross_pdf)
    DATASETS=("$TARGET")
    ;;
  all)
    DATASETS=(ordinary unanswerable reasoning cross_pdf)
    ;;
  *)
    echo "Usage: bash run.sh {ordinary|unanswerable|reasoning|cross_pdf|all} [max_new_qa]"
    exit 2
    ;;
esac

echo "============================================================"
echo "DeepSeek-V4-Flash-Vision-Exp PDF-QA"
echo "Python      : $PY"
echo "Model       : $MODEL"
echo "Endpoint    : $BASE_URL"
echo "Target      : $TARGET"
echo "Max new QA  : $MAX_NEW"
echo "DPI         : 144"
echo "JPEG Q      : 82"
echo "Per-page b64: 9000000 bytes"
echo "Whole PDF   : YES"
echo "Repair      : OFF"
echo "Thinking    : disabled"
echo "Prompt      : pdfqa-prompt-v2-strict-min"
echo "============================================================"

env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  DEEPSEEK_V_MODEL="$MODEL" \
  DEEPSEEK_V_BASE_URL="$BASE_URL" \
  "$PY" -u run_pdfqa.py \
    --datasets "${DATASETS[@]}" \
    --pdf-dpi 144 \
    --jpeg-quality 82 \
    --max-base64-bytes-per-page 9000000 \
    --max-tokens 4096 \
    --request-timeout 1800 \
    --max-api-attempts 5 \
    --max-new-qa "$MAX_NEW"

for ds in "${DATASETS[@]}"; do
  echo "[Parse] $ds"
  "$PY" parse_pdfqa_results.py --dataset-id "$ds" --dpi 144
done
