"""Runpod Serverless handler for Qwen-Image with Lightning 8-step LoRA.

Loads the official Apache-2.0 Qwen-Image diffusers pipeline once at worker
boot, fuses the Apache-2.0 lightx2v Lightning LoRA, and serves text-to-image
requests. All weights cached to /runpod-volume so the network volume catches
re-uses across workers.

Request shape (from homebrew/images/qwen_client.py):
    {"input": {
        "prompt": str,
        "width": int (default 1024),
        "height": int (default 1024),
        "num_inference_steps": int (default 8 — Lightning),
        "true_cfg_scale": float (default 1.0 — Lightning prefers low cfg),
        "negative_prompt": str (optional),
        "seed": int (optional, deterministic)
    }}

Response shape:
    {"image_base64": str (PNG bytes, base64-encoded), "model": str, "seed": int}

Why this design:
- Apache 2.0 base + Apache 2.0 LoRA + our handler under our own license
- 8-step Lightning fits in 16-20GB VRAM on L4/A10G (~$0.30/hr serverless)
- ~3-4s per image after cold start, ~$0.0004/image
- Network volume eliminates the disk-full failure mode that killed pqhaz3925
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Optional

import runpod
import torch
from diffusers import DiffusionPipeline

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("qwen-image-worker")

# ── Model + LoRA identifiers ──
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen-Image")
LIGHTNING_REPO = os.getenv("LIGHTNING_REPO", "lightx2v/Qwen-Image-Lightning")
LIGHTNING_WEIGHT = os.getenv(
    "LIGHTNING_WEIGHT",
    "Qwen-Image-Lightning-8steps-V1.0.safetensors",
)

# Cache lives on the Runpod Network Volume mounted at /runpod-volume.
# Falls back to ~/.cache/huggingface if no volume is attached (works but
# every worker re-downloads ~20GB of weights, much slower cold start).
HF_HOME = os.getenv("HF_HOME", "/runpod-volume/huggingface")
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HOME

# ── Load pipeline once at worker boot ──
logger.info("loading base model: %s", BASE_MODEL)
logger.info("cache: %s", HF_HOME)

_load_start = time.monotonic()

# bfloat16 for memory efficiency on L4/A10G; matches the official model card.
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

pipeline = DiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    cache_dir=HF_HOME,
)

logger.info("loading Lightning LoRA: %s / %s", LIGHTNING_REPO, LIGHTNING_WEIGHT)
pipeline.load_lora_weights(
    LIGHTNING_REPO,
    weight_name=LIGHTNING_WEIGHT,
    cache_dir=HF_HOME,
)
pipeline.fuse_lora()
pipeline.unload_lora_weights()  # weights are fused; drop the LoRA scaffolding

if torch.cuda.is_available():
    pipeline.to("cuda")
    # Attention slicing keeps us under 16GB VRAM on L4 at 1328x1328
    pipeline.enable_attention_slicing()
    logger.info("pipeline on cuda, attention slicing on")
else:
    logger.warning("no CUDA detected; running on CPU (testing only)")

_load_elapsed = time.monotonic() - _load_start
logger.info("ready in %.1fs", _load_elapsed)


# ── Helpers ──

DEFAULT_NEGATIVE = (
    "text, watermark, logo, caption, ui element, screenshot, illustration, "
    "cartoon, rendered 3d, cgi, plastic skin, low resolution, jpeg artifact"
)

# Lightning is calibrated for these dimensions. Other sizes work but may
# show artifacts around the edges of the diffusion grid.
SUPPORTED_SIZES = {
    (1024, 1024), (1280, 720), (720, 1280),
    (1024, 1280), (1280, 1024),
    (1024, 1536), (1536, 1024),
    (1328, 1328),
}


def _coerce_size(width: int, height: int) -> tuple[int, int]:
    """Force dimensions to multiples of 16 for VAE compatibility."""
    w = max(512, min(2048, int(width)))
    h = max(512, min(2048, int(height)))
    w = (w // 16) * 16
    h = (h // 16) * 16
    return w, h


def handler(job: dict) -> dict:
    """Runpod handler entry point."""
    try:
        job_input = job.get("input") or {}
        prompt = (job_input.get("prompt") or "").strip()
        if not prompt:
            return {"error": "missing required field: prompt"}

        width = int(job_input.get("width", 1024))
        height = int(job_input.get("height", 1024))
        width, height = _coerce_size(width, height)

        # Lightning 8-step defaults
        num_steps = int(job_input.get("num_inference_steps", 8))
        # Lightning prefers true_cfg_scale ≈ 1.0 (we treat values >4 as
        # accidentally targeting the standard model and clip)
        cfg_scale = float(job_input.get("true_cfg_scale", 1.0))
        if cfg_scale > 4.0:
            cfg_scale = 1.0

        negative = job_input.get("negative_prompt") or DEFAULT_NEGATIVE
        seed = job_input.get("seed")

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(int(seed) & 0x7FFFFFFF)

        t0 = time.monotonic()
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                true_cfg_scale=cfg_scale,
                generator=generator,
            )
        elapsed = time.monotonic() - t0

        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        logger.info(
            "generated %dx%d in %.1fs (%d steps, cfg=%.2f, seed=%s)",
            width, height, elapsed, num_steps, cfg_scale, seed,
        )

        return {
            "image_base64": b64,
            "model": f"{BASE_MODEL}+lightning-{num_steps}",
            "width": width,
            "height": height,
            "seed": seed,
            "elapsed_s": round(elapsed, 2),
        }
    except torch.cuda.OutOfMemoryError as e:
        logger.exception("OOM")
        return {"error": f"out of memory: {e}", "error_type": "oom"}
    except Exception as e:
        logger.exception("handler failed")
        return {"error": str(e), "error_type": e.__class__.__name__}


# ── Boot the Runpod serverless loop ──
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
