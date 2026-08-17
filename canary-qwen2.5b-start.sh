#!/bin/sh
#
# Start a Gradio web UI for NVIDIA Canary-Qwen-2.5B, an English speech
# recognition (ASR) model from NVIDIA NeMo.
#
#   Model card: https://huggingface.co/nvidia/canary-qwen-2.5b
#
# IMPORTANT: Canary-Qwen-2.5B is a SPEECH-TO-TEXT (ASR) model, NOT a chat LLM.
# You give it English audio (.wav/.flac, 16 kHz mono is ideal) and it returns a
# transcript with punctuation and capitalization. It has a secondary "LLM mode"
# that can post-process a transcript, but in that mode it no longer understands
# raw audio -- only the transcript text. There is therefore NO OpenAI
# /v1/chat/completions endpoint; it runs through the NeMo Python toolkit.
#
# This script serves a simple Gradio page where you upload or record audio and
# get the transcript back in the browser.
#
# Why NGC PyTorch + pip install NeMo (and not an nvcr.io/nvidia/nemo tag):
# Canary-Qwen requires the trunk build of NeMo (nemo.collections.speechlm2), and
# the NGC PyTorch image is the known-good Blackwell (sm_121 / CUDA 13) base on
# this DGX Spark host -- the same image the flux script uses.
#
# Hardware note: this host is an NVIDIA GB10 (Grace Blackwell / DGX Spark),
# aarch64 with 121 GB unified CPU/GPU memory. The 2.5B bf16 model is tiny
# (~3 GB) and fits with enormous headroom.

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Get a token from https://huggingface.co/settings/tokens
# Load environment variables from .env in script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

# Canary-Qwen-2.5B model handle (NeMo / Hugging Face)
export MODEL_HANDLE="nvidia/canary-qwen-2.5b"

# Base image: NVIDIA NGC PyTorch (linux/arm64 with Blackwell sm_121 / CUDA 13
# kernels). Same known-good base as the flux script. NeMo is pip-installed into
# a persisted dir on first boot (see below) so it is not re-downloaded each run.
export NEMO_IMAGE="nvcr.io/nvidia/pytorch:25.10-py3"

# Host dir that persists the pip-installed NeMo toolkit + deps across runs, so
# the (large) install only happens on the first boot.
export NEMO_PYDEPS="$HOME/canary-qwen/pydeps"

# Serve the Gradio web UI on this host port.
export PORT=7860

mkdir -p "$NEMO_PYDEPS"

echo "Starting Canary-Qwen-2.5B transcription web UI..."

# Remove any previous container so logs/name don't collide
docker rm -f canary-qwen 2>/dev/null

# Launch the container (no --rm so logs survive a crash for debugging).
# --ipc=host / --shm-size: audio tensors + dataloader workers use shared memory.
#
# On first boot we pip-install the NeMo trunk (which provides the speechlm2 SALM
# model class Canary-Qwen needs) plus gradio into the PERSISTED $NEMO_PYDEPS dir
# and prepend it to PYTHONPATH, so subsequent runs start fast. The HF cache is
# mounted so the ~3 GB checkpoint is downloaded only once.
docker run -d \
  --gpus all \
  --shm-size 16g \
  --ipc=host \
  -p ${PORT}:7860 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v "$NEMO_PYDEPS":/workspace/pydeps \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name canary-qwen \
  "$NEMO_IMAGE" \
    bash -c '
      set -e
      PYDEPS=/workspace/pydeps
      export PYTHONPATH="$PYDEPS:$PYTHONPATH"

      # Install NeMo (trunk) + gradio into the persisted dir on first run only.
      # nemo.collections.speechlm2 (the SALM class Canary-Qwen uses) is only in
      # the trunk build, per the model card, hence the git install.
      #
      # peft is required by nemo.collections.speechlm2 (the underlying LLM uses
      # LoRA adapters) but is NOT pulled in by the [asr] extra, so add it here.
      # We PIN peft: recent peft (>=~0.17) gates LoRA dispatch behind a strict
      # "torchao > 0.16.0" check, but the NGC image ships torchao 0.14.0+git and
      # simply *having* torchao present makes peft raise:
      #   ImportError: Found an incompatible version of torchao ... only
      #   versions above 0.16.0 are supported
      # peft 0.15.2 predates that strict gate and does standard LoRA on the
      # Qwen nn.Linear layers Canary-Qwen needs, so we pin it.
      if [ ! -f "$PYDEPS/.installed" ]; then
        echo "First run: installing NeMo toolkit (trunk) + gradio (persisted)..."
        pip install --no-cache-dir --target "$PYDEPS" \
          "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git" \
          "peft==0.15.2" gradio soundfile librosa
        touch "$PYDEPS/.installed"
      fi

      # Ensure the pinned peft is present even if the persisted dir predates this
      # pin (avoids a full reinstall on already-populated PYDEPS dirs). We force
      # the exact version so a previously-installed newer peft is replaced.
      if [ ! -f "$PYDEPS/.peft-0.15.2" ]; then
        echo "Pinning peft==0.15.2 (compatible with the image torchao)..."
        pip install --no-cache-dir --target "$PYDEPS" --upgrade \
          "peft==0.15.2"
        touch "$PYDEPS/.peft-0.15.2"
      fi

      # CRITICAL: pip resolves a generic torch + a full generic CUDA stack as
      # deps of NeMo and drops them into PYDEPS. Because PYDEPS is first on
      # PYTHONPATH, that generic torch SHADOWS the NGC image Blackwell torch
      # (2.9.0a0, sm_121/CUDA 13). The image torchvision was built against the
      # NGC torch, so the mismatch breaks with:
      #   RuntimeError: operator torchvision::nms does not exist
      # NeMo only needs torch>=2.6, which the NGC torch already satisfies, so
      # remove the shadowing generic torch/cuda/triton wheels and let the NGC
      # builds in the image win. This is idempotent and cheap (just rm -rf of a
      # few dirs), so we run it on EVERY launch -- crucially AFTER any pip
      # install above (e.g. peft) which can itself re-drop a generic torch.
      echo "Removing shadowing generic torch/cuda wheels from PYDEPS..."
      ( cd "$PYDEPS" && rm -rf \
          torch torch-*.dist-info torchgen functorch \
          torchvision torchvision-*.dist-info torchvision.libs \
          torchaudio torchaudio-*.dist-info \
          triton triton-*.dist-info pytorch_triton* \
          nvidia nvidia_*.dist-info \
          cuda cuda_*.dist-info cuda_bindings* cuda_pathfinder* 2>/dev/null ) || true

      cat > /workspace/app.py <<"PYEOF"
import os
import gradio as gr
import librosa
import soundfile as sf
import tempfile
from nemo.collections.speechlm2.models import SALM

MODEL_HANDLE = os.environ.get("MODEL_HANDLE", "nvidia/canary-qwen-2.5b")
print(f"Loading {MODEL_HANDLE} ...")
model = SALM.from_pretrained(MODEL_HANDLE)
print("Model loaded.")


def _prep(audio_path):
    # Canary expects 16 kHz mono audio.
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav, 16000)
    return tmp.name


def transcribe(audio_path):
    if not audio_path:
        return "Please provide an audio file (upload or record)."
    wav_path = _prep(audio_path)
    answer_ids = model.generate(
        prompts=[[{
            "role": "user",
            "content": f"Transcribe the following: {model.audio_locator_tag}",
            "audio": [wav_path],
        }]],
        max_new_tokens=256,
    )
    return model.tokenizer.ids_to_text(answer_ids[0].cpu())


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="English audio (upload or record)"),
    outputs=gr.Textbox(label="Transcript", lines=8),
    title="NVIDIA Canary-Qwen-2.5B - Speech to Text",
    description=("English ASR with punctuation & capitalization. "
                 "Best with clips under ~40s at 16 kHz."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
PYEOF

      exec python3 /workspace/app.py
    '

echo "Container started."
echo "  Web UI : http://localhost:${PORT}"
echo "  Logs   : docker logs -f canary-qwen"
echo
echo "First run installs NeMo + downloads the model (~3 GB); watch the logs"
echo "until you see 'Model loaded.' before opening the UI."
