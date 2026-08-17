#!/bin/sh

# Get a token from https://huggingface.co/settings/tokens
# Load environment variables from .env in script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

# Set to the DeepSeek Distill model
export MODEL_HANDLE="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Set context length (8192 is a safe starting point, you can increase this later for patents)
export MAX_MODEL_LEN=65536

# Use the CUDA 13.0 image required for the Spark's Blackwell GPU
export SGLANG_IMAGE="lmsysorg/sglang:latest-cu130"

echo "Starting SGLang server for $MODEL_HANDLE..."

# Launch the container
docker run -d \
  --rm \
  --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name sglang-deepseek \
  "$SGLANG_IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.65 \
    --context-length "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --quantization fp8

echo "Container started. View logs with: docker logs -f sglang-deepseek"
