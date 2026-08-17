#!/bin/sh

# Load environment variables from .env in script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

# Qwen3-14B dense chat LLM (bf16). ~28 GB of weights, fits with huge headroom
# on the DGX Spark's ~128 GB unified memory.
export MODEL_HANDLE="Qwen/Qwen3-14B"

# Context length. Qwen3-14B's native max is 32768; going higher requires YaRN
# scaling and SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN, so keep it here.
export MAX_MODEL_LEN=32768

# Use the CUDA 13.0 image required for the Spark's Blackwell GPU
export SGLANG_IMAGE="lmsysorg/sglang:latest-cu130"

echo "Starting SGLang server for $MODEL_HANDLE..."

# Launch the container. Standalone: its own name and port (30003) so it does not
# collide with the DeepSeek harness (sglang-deepseek on 30000). bf16 native, so
# no --quantization (which would otherwise conflict with the compute dtype).
docker run -d \
  --rm \
  --gpus all \
  --shm-size 32g \
  -p 30003:30000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name sglang-qwen3-14b \
  "$SGLANG_IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.65 \
    --context-length "$MAX_MODEL_LEN"

echo "Container started. View logs with: docker logs -f sglang-qwen3-14b"
