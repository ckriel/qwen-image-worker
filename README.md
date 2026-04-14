# qwen-image-worker

Runpod Serverless worker for [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) with the [Lightning 8-step LoRA](https://huggingface.co/lightx2v/Qwen-Image-Lightning).

Built and maintained by [Screenburn Group](https://screenburn.app) for the Metrotone Media imaging pipeline. Apache 2.0 throughout.

## What this is

A minimal `handler.py` plus `Dockerfile` that:

- Loads the Qwen-Image 20B MMDiT pipeline once at worker boot
- Fuses the Lightning 8-step LoRA so inference takes 3-4 seconds on L4 / A10G / L40S
- Serves text-to-image requests via Runpod's standard async serverless contract
- Caches model weights on the attached Network Volume so cold starts after the first one are fast

## Why we built this

Existing Hub templates either had license issues (AGPL) or worker-disk problems (40GB Qwen-Image weights don't fit on default ephemeral disk). Rolling our own gave us:

- Apache 2.0 base + Apache 2.0 LoRA + our own handler under permissive license (no copyleft inheritance)
- Network Volume properly configured for model weight cache
- ~$0.0004 per image, ~$1-2/month at 60 images/day

## API contract

POST `https://api.runpod.ai/v2/{ENDPOINT_ID}/run`

```json
{
  "input": {
    "prompt": "Editorial photograph of a quiet Copenhagen coffee bar at golden hour",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 8,
    "true_cfg_scale": 1.0,
    "negative_prompt": "text, watermark, logo, ui element",
    "seed": 42
  }
}
```

Response (after polling `/status/{job_id}` until `COMPLETED`):

```json
{
  "output": {
    "image_base64": "iVBORw0KGgo...",
    "model": "Qwen/Qwen-Image+lightning-8",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "elapsed_s": 3.4
  }
}
```

## Deploy

1. Connect this GitHub repo to your Runpod account (Console → Settings → Connections → GitHub).
2. Console → Serverless → New Endpoint → "GitHub repo" tab.
3. Pick this repo, branch `main`.
4. GPU types: L4, A10G, or L40S (16GB+ VRAM is enough).
5. Container Disk: 20 GB (handler is small, weights live on Network Volume).
6. **Attach a Network Volume of at least 30 GB** mounted at `/runpod-volume`. First cold start downloads ~22 GB of weights into it. Subsequent workers mount the cache and start in ~30 seconds.
7. Idle Timeout: 5-30 seconds (we don't need long idle periods; scale-to-zero saves money).
8. Max Workers: 1-3 depending on expected concurrency.
9. Active Workers: 0 (cold-start tolerated; saves money).

## Local sanity check

```bash
docker build -t qwen-image-worker .
docker run --rm --gpus all -e RUNPOD_TEST=1 qwen-image-worker
```

## License

Apache License 2.0. Same as Qwen-Image and the Lightning LoRA.
