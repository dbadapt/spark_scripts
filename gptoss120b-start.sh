#!/bin/sh
#
# Start a vLLM OpenAI-compatible server for OpenAI's GPT-OSS 120B.
#
# Based on OpenAI's official vLLM recipe for gpt-oss:
#   https://cookbook.openai.com/articles/gpt-oss/run-vllm
#     vllm serve openai/gpt-oss-120b
#
# Why vLLM: OpenAI ships gpt-oss with vLLM as the reference serving path. The
# weights are natively MXFP4-quantized (no separate --quantization flag needed;
# vLLM detects it from the checkpoint) and the model uses the Harmony response
# format. vLLM has built-in Harmony parsing plus the "openai" tool-call parser.
#
# Hardware note: this host is an NVIDIA GB10 (Grace Blackwell / DGX Spark),
# aarch64 with 121 GB UNIFIED CPU/GPU memory. The MXFP4 120B weights are ~63 GB,
# which fits comfortably on a single GB10. The vllm/vllm-openai image ships a
# linux/arm64 variant; if it lacks Blackwell (sm_121 / CUDA 13) kernels, switch
# VLLM_IMAGE to NVIDIA's NGC build (nvcr.io/nvidia/vllm).

# Get a token from https://huggingface.co/settings/tokens
# Load environment variables from .env in script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

# GPT-OSS 120B model handle (native MXFP4 checkpoint)
export MODEL_HANDLE="openai/gpt-oss-120b"

# vLLM OpenAI server image (has a linux/arm64 variant)
export VLLM_IMAGE="vllm/vllm-openai:latest"

# Serve on port 8000 (vLLM default, OpenAI-compatible /v1 API)
export PORT=8000

# gpt-oss supports the full 128k context window
export MAX_MODEL_LEN=131072

echo "Starting vLLM server for $MODEL_HANDLE..."

# ---------------------------------------------------------------------------
# Harmony vocab pre-seed (fixes: openai_harmony.HarmonyError: error downloading
# or loading vocab file).
#
# gpt-oss uses the Harmony response format. At the FIRST /v1/chat/completions
# request, vLLM lazily calls openai_harmony -> tiktoken-rs, which tries to
# download the o200k_base tiktoken vocab from openaipublic.blob.core.windows.net.
# Model *weights* are cached in ~/.cache/huggingface, but this vocab is fetched
# from a DIFFERENT source and is NOT part of that cache. If the container has no
# network egress at request time (or the blob endpoint is blocked/flaky), every
# chat request fails with a 500 even though the server "started" fine.
#
# Fix: pre-download the vocab on the host (which does have connectivity) into a
# persistent tiktoken-rs cache dir, then mount it into the container and point
# openai_harmony at it via TIKTOKEN_RS_CACHE_DIR. tiktoken-rs names its cache
# files sha1(url); we reproduce that so the lib finds the file offline.
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken-rs"
VOCAB_URL="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
VOCAB_SHA256="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
VOCAB_NAME=$(printf '%s' "$VOCAB_URL" | sha1sum | cut -d' ' -f1)
VOCAB_PATH="$TIKTOKEN_CACHE_DIR/$VOCAB_NAME"

mkdir -p "$TIKTOKEN_CACHE_DIR"
if [ ! -f "$VOCAB_PATH" ] || [ "$(sha256sum "$VOCAB_PATH" | cut -d' ' -f1)" != "$VOCAB_SHA256" ]; then
  echo "Pre-downloading Harmony vocab (o200k_base.tiktoken)..."
  if ! curl -fSL -o "$VOCAB_PATH" "$VOCAB_URL"; then
    echo "ERROR: failed to download Harmony vocab from $VOCAB_URL" >&2
    echo "The server will start but chat completions will 500 until this is fixed." >&2
  elif [ "$(sha256sum "$VOCAB_PATH" | cut -d' ' -f1)" != "$VOCAB_SHA256" ]; then
    echo "ERROR: Harmony vocab sha256 mismatch after download" >&2
    rm -f "$VOCAB_PATH"
  else
    echo "Harmony vocab cached at $VOCAB_PATH"
  fi
fi

# Remove any previous container so logs/name don't collide
docker rm -f vllm-gptoss120b 2>/dev/null

# Launch the container (no --rm so logs survive a crash for debugging).
# --ipc=host is recommended by vLLM for shared-memory tensor transport.
# --gpu-memory-utilization 0.90: on this 121 GB unified pool the ~63 GB MXFP4
#   weights leave plenty of room; 0.90 gives a large KV cache for 128k context.
# --async-scheduling: OpenAI's recommended flag for gpt-oss throughput on vLLM.
# --enable-auto-tool-choice + --tool-call-parser openai: required for clients
#   (like opencode) that send tool_choice="auto". gpt-oss emits tool calls in
#   the Harmony format, which the "openai" parser understands.
docker run -d \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  -p ${PORT}:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TIKTOKEN_RS_CACHE_DIR=/root/.cache/tiktoken-rs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$TIKTOKEN_CACHE_DIR":/root/.cache/tiktoken-rs \
  --name vllm-gptoss120b \
  "$VLLM_IMAGE" \
    --model "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser openai

echo "Container started. View logs with: docker logs -f vllm-gptoss120b"
