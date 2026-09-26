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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-${PY_MISTRAL31:-python}}"
MODEL="$WORKSPACE_ROOT/models/Mistral-Small-3.1-24B-Instruct-2503"

TARGET="${1:-all}"
MAX_SAMPLES="${2:--1}"

PAIR="${GPU:-4,5}"
PORT="${MISTRAL31_PORT:-8102}"

GPU_UTIL="${MISTRAL31_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_IMAGES="${MISTRAL31_MAX_IMAGES:-512}"
MIN_FREE="${MISTRAL31_MIN_FREE_MIB:-70000}"
READY_TIMEOUT="${MISTRAL31_READY_TIMEOUT:-720}"

CACHE="$HERE/output/render_cache"
RAW="$HERE/output/raw_result"

export MISTRAL31_GPU_PAIR="$PAIR"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
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
    echo "[ERROR] unknown target: $TARGET"
    echo "Usage:"
    echo "  GPU=4,5 bash run.sh ordinary [max_samples]"
    echo "  GPU=4,5 bash run.sh unanswerable [max_samples]"
    echo "  GPU=4,5 bash run.sh reasoning [max_samples]"
    echo "  GPU=4,5 bash run.sh cross_pdf [max_samples]"
    echo "  GPU=4,5 bash run.sh all"
    exit 2
    ;;
esac

IFS=',' read -r G0 G1 <<< "$PAIR"

if [[ ! "$G0" =~ ^[0-9]+$ ]] \
   || [[ ! "$G1" =~ ^[0-9]+$ ]] \
   || [[ "$G0" == "$G1" ]]
then
  echo "[ERROR] GPU must be two different indices, e.g. GPU=4,5"
  exit 2
fi

# ------------------------------------------------------------
# GPU preflight
# ------------------------------------------------------------

for G in "$G0" "$G1"
do
  FREE="$(
    nvidia-smi \
      -i "$G" \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits \
    | head -n 1 \
    | tr -d ' '
  )"

  echo "[GPU] $G free=${FREE} MiB"

  if (( FREE < MIN_FREE ))
  then
    echo "[ERROR] GPU $G free=${FREE} MiB < ${MIN_FREE} MiB"
    exit 3
  fi
done

# ------------------------------------------------------------
# Dataset count preflight
# ------------------------------------------------------------

DATASET_CSV="$(IFS=,; echo "${DATASETS[*]}")"
export DATASET_CSV

"$PY" - <<'PY'
import json
import os
import run_pdfqa as r

for ds in os.environ["DATASET_CSV"].split(","):

    spec = r.DATASETS[ds]
    path = spec["data"]

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    n = 0

    for unit in data.values():

        if not isinstance(unit, dict):
            continue

        qa = unit.get("QA", {})

        if isinstance(qa, dict):
            n += len(qa)

    expected = int(
        spec["expected"]
    )

    if n != expected:
        raise RuntimeError(
            f"{ds}: QA={n}, expected={expected}"
        )

    print(
        f"[Preflight] {ds} "
        f"QA={n} "
        f"prompt={spec['prompt_id']}"
    )
PY

# ------------------------------------------------------------
# vLLM server lifecycle
# ------------------------------------------------------------

SERVER_LOG="output/logs/vllm_server_gpu${G0}${G1}.log"
SERVER_PGID=""

cleanup() {
  trap - EXIT INT TERM

  if [[ -n "${SERVER_PGID:-}" ]]
  then
    echo "[STOP] vLLM process group $SERVER_PGID"

    kill -TERM -- "-$SERVER_PGID" \
      2>/dev/null || true

    sleep 3

    kill -KILL -- "-$SERVER_PGID" \
      2>/dev/null || true
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ss -ltn 2>/dev/null | grep -q ":${PORT} "
then
  echo "[ERROR] port $PORT is already in use"
  exit 4
fi

echo
echo "============================================================"
echo "Mistral-Small-3.1-24B-Instruct-2503 PDF-QA"
echo "GPU pair    : $PAIR"
echo "Python      : $PY"
echo "Model       : $MODEL"
echo "Target      : $TARGET"
echo "Max samples : $MAX_SAMPLES"
echo "Backend     : vLLM TP2"
echo "Context     : 128000"
echo "DPI         : 144"
echo "JPEG        : quality=95 subsampling=0"
echo "Context fit : target=118000 min_scale=0.35 attempts=6"
echo "Repair      : OFF"
echo "Prompt      : pdfqa-prompt-v2-strict-min"
echo "============================================================"

setsid env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$PAIR" \
  TOKENIZERS_PARALLELISM=false \
  VLLM_USE_V1=1 \
  "$PY" \
  -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name mistral31 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 128000 \
    --max-num-seqs 1 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --limit-mm-per-prompt "{\"image\":${MAX_IMAGES}}" \
    --allowed-local-media-path "$CACHE" \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --disable-custom-all-reduce \
    --enforce-eager \
    --skip-mm-profiling \
    --mm-processor-cache-gb 0 \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

sleep 1

SERVER_PGID="$(
  ps -o pgid= -p "$SERVER_PID" \
  2>/dev/null \
  | tr -d ' '
)"

if [[ -z "$SERVER_PGID" ]]
then
  echo "[ERROR] server exited before PGID capture"
  tail -n 150 "$SERVER_LOG" || true
  exit 5
fi

echo "[WAIT] vLLM pid=$SERVER_PID pgid=$SERVER_PGID"

READY=0

for ((SEC=0; SEC<READY_TIMEOUT; SEC+=2))
do
  if curl -fsS \
    "http://127.0.0.1:${PORT}/health" \
    >/dev/null 2>&1
  then
    READY=1
    break
  fi

  if ! kill -0 "$SERVER_PID" 2>/dev/null
  then
    echo "[ERROR] vLLM server died during startup"
    tail -n 200 "$SERVER_LOG" || true
    exit 6
  fi

  sleep 2
done

if [[ "$READY" != "1" ]]
then
  echo "[ERROR] vLLM readiness timeout"
  tail -n 200 "$SERVER_LOG" || true
  exit 7
fi

echo "[READY] vLLM server"

# ------------------------------------------------------------
# Sequential inference
# ------------------------------------------------------------

for DATASET_ID in "${DATASETS[@]}"
do
  echo
  echo "============================================================"
  echo "[Dataset] $DATASET_ID"
  echo "============================================================"

  CLIENT_LOG="output/logs/${DATASET_ID}__mistral31_tp2.log"

  CMD=(
    "$PY"
    -u
    run_pdfqa.py

    --datasets
    "$DATASET_ID"

    --model-path
    "$MODEL"

    --output-root
    "$RAW"

    --cache-root
    "$CACHE"

    --dpi
    144

    --jpeg-quality
    95

    --max-new-tokens
    512

    --base-url
    "http://127.0.0.1:${PORT}/v1"

    --served-model
    mistral31

    --request-timeout
    1800

    --context-fit-target-prompt-tokens
    118000

    --context-fit-min-scale
    0.35

    --context-fit-max-attempts
    6

    --no-format-repair
  )

  if [[ "$MAX_SAMPLES" -gt 0 ]]
  then
    CMD+=(
      --max-new-qa
      "$MAX_SAMPLES"
    )
  fi

  echo "[Run] ${CMD[*]}"

  "${CMD[@]}" \
    2>&1 \
    | tee -a "$CLIENT_LOG"

  echo "[Parse] $DATASET_ID"

  "$PY" \
    -u \
    parse_pdfqa_results.py \
    --dataset-id "$DATASET_ID" \
    2>&1 \
    | tee -a "$CLIENT_LOG"
done

echo
echo "============================================================"
echo "Finished: $(date -Is)"
echo "============================================================"
