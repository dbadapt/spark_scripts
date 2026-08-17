#!/bin/sh
#
# Start a vLLM OpenAI-compatible server for DiffusionGemma.
#
# Based on Google's official vLLM recipe from the model card:
#   https://huggingface.co/google/diffusiongemma-26B-A4B-it
#     vllm serve "google/diffusiongemma-26B-A4B-it"
#
# Why vLLM (and not SGLang): the model's architecture is
# DiffusionGemmaForBlockDiffusion (block-diffusion encoder-decoder MoE). As of
# the lmsysorg/sglang:latest-cu130 image, SGLang has NO model class registered
# for this architecture (a repo-wide search for "DiffusionGemma" returns
# nothing), so it falls back to a generic multimodal-MoE causal adapter that
# crashes the scheduler on the warmup forward pass. Google's card lists vLLM as
# a first-class serving path, so we serve with vLLM instead.
#
# Hardware note: this host is an NVIDIA GB10 (Grace Blackwell / DGX Spark),
# aarch64 with 121 GB UNIFIED CPU/GPU memory. The official image ships an
# linux/arm64 variant; if it lacks Blackwell (sm_121 / CUDA 13) kernels we will
# need NVIDIA's NGC vLLM build (nvcr.io/nvidia/vllm) instead.

# Get a token from https://huggingface.co/settings/tokens
# Load environment variables from .env in script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

# DiffusionGemma model handle
export MODEL_HANDLE="google/diffusiongemma-26B-A4B-it"

# vLLM OpenAI server image (has a linux/arm64 variant)
export VLLM_IMAGE="vllm/vllm-openai:latest"

# Serve on port 8000 (vLLM default, matches the model card curl example)
export PORT=8000

echo "Starting vLLM server for $MODEL_HANDLE..."

# Remove any previous container so logs/name don't collide
docker rm -f vllm-diffusiongemma 2>/dev/null

# Launch the container (no --rm so logs survive a crash for debugging).
# --ipc=host is recommended by vLLM for shared-memory tensor transport.
# --gpu-memory-utilization 0.85: on this 121 GB unified pool ~110 GB is free at
# startup; vLLM's default 0.92 wants ~112 GB and aborts. 0.85 (~103 GB) fits.
# --enable-auto-tool-choice + --tool-call-parser gemma4: required for clients
# (like opencode) that send tool_choice="auto". Without these, requests fail
# with: '"auto" tool choice requires --enable-auto-tool-choice and
# --tool-call-parser to be set'. DiffusionGemma is Gemma 4 family, so the
# "gemma4" parser handles its native <|tool_call> syntax.
docker run -d \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  -p ${PORT}:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm-diffusiongemma \
  "$VLLM_IMAGE" \
    --model "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4

echo "Container started. View logs with: docker logs -f vllm-diffusiongemma"
