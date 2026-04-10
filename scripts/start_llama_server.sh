#!/usr/bin/env bash
set -euo pipefail

llama_server_bin="llama-server"
model_path="${HOME}/models/Qwen3.5-9B-UD-Q8_K_XL.gguf"
mmproj_path="${HOME}/models/qwen35-mmproj-F16.gguf"
host="127.0.0.1"
port="8080"
ctx_size="8192"
gpu_layers="-1"
threads="8"
parallel="1"
reasoning_budget="0"
reasoning_budget_message=""
extra_args=()

usage() {
  cat <<'EOF'
Usage: ./scripts/start_llama_server.sh [options] [-- extra llama-server args]

Options:
  --model PATH              Model GGUF path (default: ~/models/Qwen3.5-9B-UD-Q8_K_XL.gguf)
  --mmproj PATH             Multimodal projector path (default: ~/models/qwen35-mmproj-F16.gguf)
  --host HOST               Server host (default: 127.0.0.1)
  --port PORT               Server port (default: 8080)
  --ctx-size TOKENS         Context size (default: 8192)
  --gpu-layers N            Number of GPU layers (default: -1)
  --threads N               CPU thread count (default: 8)
  --parallel N              Number of parallel requests (default: 1)
  --reasoning-budget N      Thinking token budget (default: 0, disabled)
  --reasoning-budget-message MSG
                            Message injected when reasoning budget is exhausted
  --bin PATH                llama-server binary path (default: llama-server)
  -h, --help                Show this help message

Notes:
  Sampling parameters such as temperature, top_p, top_k, and seed are sent
  by the captioning client on each request for reproducibility.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model_path="$2"
      shift 2
      ;;
    --mmproj)
      mmproj_path="$2"
      shift 2
      ;;
    --host)
      host="$2"
      shift 2
      ;;
    --port)
      port="$2"
      shift 2
      ;;
    --ctx-size)
      ctx_size="$2"
      shift 2
      ;;
    --gpu-layers)
      gpu_layers="$2"
      shift 2
      ;;
    --threads)
      threads="$2"
      shift 2
      ;;
    --parallel)
      parallel="$2"
      shift 2
      ;;
    --reasoning-budget)
      reasoning_budget="$2"
      shift 2
      ;;
    --reasoning-budget-message)
      reasoning_budget_message="$2"
      shift 2
      ;;
    --bin)
      llama_server_bin="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$model_path" ]]; then
  printf 'Model file not found: %s\n' "$model_path" >&2
  exit 1
fi

if [[ -n "$mmproj_path" && ! -f "$mmproj_path" ]]; then
  printf 'MM projector file not found: %s\n' "$mmproj_path" >&2
  exit 1
fi

args=(
  --model "$model_path"
  --host "$host"
  --port "$port"
  --ctx-size "$ctx_size"
  --n-gpu-layers "$gpu_layers"
  --threads "$threads"
  --parallel "$parallel"
  --reasoning-budget "$reasoning_budget"
)

if [[ -n "$mmproj_path" ]]; then
  args+=(--mmproj "$mmproj_path")
fi

if [[ -n "$reasoning_budget_message" ]]; then
  args+=(--reasoning-budget-message "$reasoning_budget_message")
fi

exec "$llama_server_bin" "${args[@]}" "${extra_args[@]}"
