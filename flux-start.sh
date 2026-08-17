#!/bin/sh
#
# Start ComfyUI serving Black Forest Labs' FLUX.2 [dev] -- the current
# state-of-the-art OPEN image generation/editing model (32B rectified-flow
# transformer). This is the "best FLUX that runs locally" pick: FLUX.2 [dev]
# in FULL bf16 is ~64 GB of DiT weights plus a large Mistral-3 text encoder,
# which is far too big for a 24 GB consumer GPU (those setups need 4-bit + a
# REMOTE text encoder). This host, however, is a good fit -- see hardware note.
#
# Why ComfyUI (and not vLLM / SGLang): FLUX is a diffusion image model, not an
# autoregressive LLM. vLLM and SGLang do not serve it at all. FLUX.2 is
# supported natively by ComfyUI, which exposes both a web UI and an HTTP API
# (POST /prompt, GET /history, GET /view) on the same port.
#   ComfyUI FLUX.2 example: https://comfyanonymous.github.io/ComfyUI_examples/flux2/
#   FLUX.2 model card:      https://huggingface.co/black-forest-labs/FLUX.2-dev
#
# Hardware note: this host is an NVIDIA GB10 (Grace Blackwell / DGX Spark),
# aarch64 with 121 GB UNIFIED CPU/GPU memory. That unified pool is exactly what
# makes full bf16 FLUX.2 [dev] viable locally: the ~64 GB DiT + ~14 GB fp8 text
# encoder + VAE all fit with headroom, with NO quantization of the diffusion
# transformer (best quality). The ComfyUI image must have Blackwell (sm_121 /
# CUDA 13) kernels; we use the CUDA 13 tag below.

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Get a token from https://huggingface.co/settings/tokens
# NOTE: FLUX.2-dev is GATED. You must visit the model card once and click
# "Agree" to the FLUX.2 Non-Commercial License before this token can download
# the full bf16 diffusion weights:
#   https://huggingface.co/black-forest-labs/FLUX.2-dev
# Load environment variables from .env in script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

# Serve the ComfyUI web UI + HTTP API on port 8188 (ComfyUI default).
export PORT=8188

# Base image: NVIDIA's NGC PyTorch build. This is the ONLY reliable path on the
# DGX Spark because it publishes a linux/arm64 (aarch64) variant WITH Blackwell
# (sm_121 / CUDA 13) kernels. Community ComfyUI images (e.g. yanwk/comfyui-boot)
# are linux/amd64-only and will not run on this Grace/aarch64 host. We install
# ComfyUI itself into a persisted host dir on first boot (see below).
export COMFY_IMAGE="nvcr.io/nvidia/pytorch:25.10-py3"

# Host dir for the ComfyUI source + venv (persists so we don't reinstall each run).
export COMFY_HOME="$HOME/comfyui"

# Host directory that holds all ComfyUI models (persists across runs so we
# never re-download the ~78 GB of weights). Laid out as ComfyUI expects.
export COMFY_MODELS="$COMFY_HOME/models"

# ---------------------------------------------------------------------------
# Model files (downloaded to the host once, then mounted into the container)
#
#   diffusion_models/flux2-dev.safetensors        ~64 GB bf16 DiT (GATED, official repo)
#   text_encoders/mistral_3_small_flux2_fp8.safetensors  ~14 GB fp8 text encoder
#   vae/flux2-vae.safetensors                     VAE
#
# The DiT is pulled in FULL bf16 from the official gated repo (max quality).
# The text encoder is fp8 (Comfy-Org split) -- text encoders are far less
# quality-sensitive than the DiT, and the fp8 encoder is the one ComfyUI's
# FLUX.2 reference workflow expects.
# ---------------------------------------------------------------------------

DIT_DIR="$COMFY_MODELS/diffusion_models"
TE_DIR="$COMFY_MODELS/text_encoders"
VAE_DIR="$COMFY_MODELS/vae"
mkdir -p "$DIT_DIR" "$TE_DIR" "$VAE_DIR"

DIT_PATH="$DIT_DIR/flux2-dev.safetensors"
TE_PATH="$TE_DIR/mistral_3_small_flux2_fp8.safetensors"
VAE_PATH="$VAE_DIR/flux2-vae.safetensors"

DIT_URL="https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/flux2-dev.safetensors?download=true"
TE_URL="https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors?download=true"
VAE_URL="https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors?download=true"

# download <url> <dest> <label>  -- resumable, authenticated HF download.
download() {
  _url="$1"; _dest="$2"; _label="$3"
  if [ -f "$_dest" ]; then
    echo "  [skip] $_label already present ($_dest)"
    return 0
  fi
  echo "  [get ] $_label -> $_dest"
  # -C - resumes partial downloads; large files (up to ~64 GB) benefit from this.
  if ! curl -fL -C - \
        -H "Authorization: Bearer $HF_TOKEN" \
        -o "$_dest" "$_url"; then
    echo "ERROR: failed to download $_label" >&2
    echo "If this is flux2-dev.safetensors, make sure you accepted the license at" >&2
    echo "  https://huggingface.co/black-forest-labs/FLUX.2-dev" >&2
    exit 1
  fi
}

echo "Ensuring FLUX.2 [dev] model files are present (first run downloads ~78 GB)..."
download "$DIT_URL" "$DIT_PATH" "FLUX.2 dev DiT (bf16, ~64 GB)"
download "$TE_URL"  "$TE_PATH"  "Mistral-3 text encoder (fp8, ~14 GB)"
download "$VAE_URL" "$VAE_PATH" "FLUX.2 VAE"

echo "Starting ComfyUI server for FLUX.2 [dev]..."

# Remove any previous container so logs/name don't collide
docker rm -f comfyui-flux2 2>/dev/null

# Launch the container (no --rm so logs survive a crash for debugging).
# --ipc=host / --shm-size: diffusion pipelines use large shared-memory tensors.
#
# On first boot we clone ComfyUI into the persisted $COMFY_HOME/app and install
# its requirements against the NGC image's Blackwell-ready PyTorch (we do NOT
# reinstall torch -- the NGC torch already has sm_121/CUDA 13 kernels). The
# models dir is symlinked to the persisted host models tree. Subsequent runs
# skip the clone/install and start straight away.
docker run -d \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  -p ${PORT}:8188 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$COMFY_HOME":/workspace/comfyui \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name comfyui-flux2 \
  "$COMFY_IMAGE" \
    bash -c '
      set -e
      APP=/workspace/comfyui/app
      # ComfyUI Python deps are installed into a PERSISTED dir on the mounted
      # volume (not the ephemeral image layer), so they survive container
      # recreation. Without this, deps installed at first run vanish when the
      # container is replaced, and ComfyUI crashes on missing modules such as
      # sqlalchemy. We prepend this dir to PYTHONPATH so it is importable.
      PYDEPS=/workspace/comfyui/pydeps
      export PYTHONPATH="$PYDEPS:$PYTHONPATH"
      mkdir -p "$PYDEPS"

      if [ ! -d "$APP/.git" ]; then
        echo "First run: cloning ComfyUI..."
        git clone https://github.com/comfyanonymous/ComfyUI "$APP"
      fi

      # Install/refresh ComfyUI deps into the persisted PYDEPS dir. This is
      # idempotent -- already-satisfied packages are skipped quickly. We keep
      # the NGC Blackwell torch (do NOT let ComfyUI pull generic torch/vision/
      # audio wheels over it -- those are built against a different PyTorch ABI
      # and fail to load, e.g. torchaudio raises "undefined symbol:
      # torch_library_impl"), so strip torch* from the requirements.
      echo "Ensuring ComfyUI Python deps are installed (persisted)..."
      grep -viE "^[[:space:]]*torch(vision|audio)?([<>=!~[:space:]].*)?$" \
        "$APP/requirements.txt" > "$APP/requirements.notorch.txt"
      pip install --no-cache-dir --target "$PYDEPS" \
        -r "$APP/requirements.notorch.txt"

      # Stripping torch* from requirements only drops the DIRECT pins; pip still
      # resolves torch/torchvision/triton + nvidia-*-cu13 as TRANSITIVE deps of
      # packages like kornia/spandrel/torchsde and drops generic wheels into
      # PYDEPS. Because PYDEPS is first on PYTHONPATH, those would SHADOW the NGC
      # Blackwell torch and re-break the ABI. Remove them so the NGC builds in
      # the image (sm_121/CUDA 13) always win.
      echo "Removing shadowing generic torch/cuda wheels from PYDEPS..."
      ( cd "$PYDEPS" && rm -rf \
          torch torch-*.dist-info torchgen functorch \
          torchvision torchvision-*.dist-info torchvision.libs \
          torchaudio torchaudio-*.dist-info \
          triton triton-*.dist-info \
          nvidia nvidia_*.dist-info \
          cuda cuda_*.dist-info 2>/dev/null ) || true

      # ComfyUI imports torchaudio unconditionally (comfy/ldm/lightricks/vae/
      # audio_vae.py), so it must be present AND ABI-compatible with the NGC
      # torch. The NGC image ships no torchaudio, and the pip default is too new
      # (2.11) which crashes with an undefined-symbol error. Pin the matching
      # 2.9.0 build and install with --no-deps so it cannot pull a different
      # torch over the NGC one. This one goes into the image dist-packages
      # (needs to sit alongside the NGC torch); it is cheap to reinstall.
      if ! python3 -c "import torchaudio" >/dev/null 2>&1; then
        echo "Installing ABI-matching torchaudio 2.9.0 for NGC torch..."
        pip uninstall -y torchaudio >/dev/null 2>&1 || true
        pip install --no-cache-dir --no-deps "torchaudio==2.9.0"
      fi
      # Point ComfyUI at the persisted host models tree.
      rm -rf "$APP/models"
      ln -s /workspace/comfyui/models "$APP/models"
      cd "$APP"
      exec python3 main.py --listen 0.0.0.0 --port 8188
    '

echo "Container started."
echo "  Web UI : http://localhost:${PORT}"
echo "  API    : POST http://localhost:${PORT}/prompt  (see ComfyUI API docs)"
echo "  Logs   : docker logs -f comfyui-flux2"
echo
echo "In the UI, load the FLUX.2 example workflow and select:"
echo "  UNET/diffusion model : flux2-dev.safetensors"
echo "  text encoder         : mistral_3_small_flux2_fp8.safetensors"
echo "  VAE                  : flux2-vae.safetensors"
