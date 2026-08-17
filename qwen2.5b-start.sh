#!/bin/sh

# Load environment variables from .env in script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

# Model handle for Qwen2.5 (instruct-tuned chat LLM).
# NOTE: Qwen2.5 has no "2.5B" checkpoint (valid small sizes are 0.5B/1.5B/3B/7B);
# 3B is the closest real size and is what this script serves.
export MODEL_HANDLE="Qwen/Qwen2.5-3B-Instruct"

# Context length. Qwen2.5-3B's native max is 32768; going higher requires
# YaRN scaling and SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN, so keep it here.
export MAX_MODEL_LEN=32768

# Use the CUDA 13.0 image required for the Spark's Blackwell GPU
export SGLANG_IMAGE="lmsysorg/sglang:latest-cu130"

echo "Starting SGLang server for $MODEL_HANDLE..."

# Launch the container
docker run -d \
  --rm \
  --gpus all \
  --shm-size 32g \
  -p 30001:30000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name sglang-qwen2.5b \
  "$SGLANG_IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.65 \
    --context-length "$MAX_MODEL_LEN" \
    --quantization fp8

echo "Container started. View logs with: docker logs -f sglang-qwen2.5b"