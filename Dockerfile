# Qwen-Image + Lightning LoRA worker for Runpod Serverless.
# Apache 2.0 throughout: base model (Qwen/Qwen-Image), LoRA (lightx2v),
# handler (this repo).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install torch first so the heavy CUDA wheel is its own layer (cache-friendly)
RUN pip install --upgrade pip && \
    pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Bake weights into the image ──
# Downloaded once at build time. First cold start drops from ~6 min
# (runtime download) to ~10s (just load from image layer to VRAM).
# Eliminates the "network volume fills up with partial downloads" failure
# mode and removes the Network Volume as a runtime dependency.
# Image grows to ~35-40 GB which is fine on Runpod.
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HUB_DOWNLOAD_TIMEOUT=600

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen-Image', allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model']); \
snapshot_download('lightx2v/Qwen-Image-Lightning', allow_patterns=['Qwen-Image-Lightning-8steps-V1.0.safetensors', '*.json']); \
print('weights cached to /root/.cache/huggingface') \
"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
