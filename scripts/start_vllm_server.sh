#!/usr/bin/env bash
set -euo pipefail

# Start an OpenAI-compatible vLLM server for the trained CnakeCharmer model.
#
# Defaults target the Hugging Face model repo directly:
#   CnakeCharmer/CnakeAgent-sft-v0.1
#
# Usage:
#   scripts/start_vllm_server.sh
#   scripts/start_vllm_server.sh --port 8003 --host 0.0.0.0
#   scripts/start_vllm_server.sh --model /abs/path/to/model --served-model-name my-model
#
# Optional env:
#   VLLM_API_KEY=...          # Require API key auth on the server
#   VLLM_EXTRA_ARGS="..."     # Extra args appended to vLLM launch command

MODEL_PATH="${MODEL_PATH:-CnakeCharmer/CnakeAgent-sft-v0.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-20b-cython}"
HOST="0.0.0.0"
PORT="8003"
GPU_MEM_UTIL="0.90"
MAX_MODEL_LEN="16384"
DTYPE="bfloat16"

print_help() {
  cat <<EOF
Start vLLM OpenAI API server.

Options:
  --model PATH_OR_REPO        Model directory or HF repo id (default: ${MODEL_PATH})
  --served-model-name NAME    API model name (default: ${SERVED_MODEL_NAME})
  --host HOST                 Bind host (default: ${HOST})
  --port PORT                 Bind port (default: ${PORT})
  --gpu-mem-util FLOAT        GPU memory utilization (default: ${GPU_MEM_UTIL})
  --max-model-len N           Max model length (default: ${MAX_MODEL_LEN})
  --dtype DTYPE               dtype (default: ${DTYPE})
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --gpu-mem-util)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_help
      exit 2
      ;;
  esac
done

if [[ "${MODEL_PATH}" == */* && ! -d "${MODEL_PATH}" && "${MODEL_PATH}" != /* ]]; then
  echo "Using HF model repo id: ${MODEL_PATH}"
elif [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found in PATH." >&2
  exit 1
fi

echo "Starting vLLM server..."
echo "  model: ${MODEL_PATH}"
echo "  served-model-name: ${SERVED_MODEL_NAME}"
echo "  endpoint: http://${HOST}:${PORT}/v1"

CMD=(
  uv run --no-sync python -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --trust-remote-code
)

if [[ -n "${VLLM_API_KEY:-}" ]]; then
  CMD+=(--api-key "${VLLM_API_KEY}")
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${VLLM_EXTRA_ARGS} )
  CMD+=("${EXTRA[@]}")
fi

echo
echo "Command:"
printf '  %q' "${CMD[@]}"
echo
echo

exec "${CMD[@]}"
