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

COPY handler.py .

# Hugging Face cache lives on the Network Volume mounted by Runpod.
# /runpod-volume is the conventional mount path.
ENV HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface

# Optional: pre-warm by downloading the model at build time. Disabled by
# default — keeps the image small and lets the Network Volume cache do
# its job on first cold start.
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-Image')"

CMD ["python", "-u", "handler.py"]
