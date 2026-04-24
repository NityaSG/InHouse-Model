"""Kreeda ICH-Next model API — gender + fashion category classification."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from classifiers import CategoryInference, GenderInference
from classifiers.image_loader import ImageLoadError, load_image

load_dotenv()

ROOT = Path(__file__).parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "ml_models"))
GENDER_CKPT = MODELS_DIR / os.getenv("GENDER_CHECKPOINT", "gender_classifier_best.pt")
CATEGORY_CKPT = MODELS_DIR / os.getenv("CATEGORY_CHECKPOINT", "category_classifier_best.pt")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
FETCH_WORKERS = int(os.getenv("FETCH_WORKERS", "8"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["gender"] = GenderInference(GENDER_CKPT, device=DEVICE)
    state["category"] = CategoryInference(CATEGORY_CKPT, device=DEVICE)
    state["fetch_pool"] = ThreadPoolExecutor(max_workers=FETCH_WORKERS)
    try:
        yield
    finally:
        state["fetch_pool"].shutdown(wait=False)


app = FastAPI(
    title="Kreeda ICH-Next Model API",
    description="Gender and fashion-category classification for images from s3:// or http(s) URLs.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    url: str = Field(..., description="s3://bucket/key or https://... image URL")
    tta: bool = Field(False, description="Enable test-time augmentation (5×, slower, more accurate)")


class BatchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, description="List of image URLs (s3:// or http(s))")
    tta: bool = Field(False, description="Enable test-time augmentation")


class FetchFailure(BaseModel):
    url: str
    error: str


# ── Helpers ──────────────────────────────────────────────────────────────────
def _fetch_many(urls: list[str]) -> tuple[list[tuple[int, Image.Image]], list[FetchFailure]]:
    """Download URLs in parallel. Returns (successes[(orig_idx, img)], failures)."""
    pool: ThreadPoolExecutor = state["fetch_pool"]
    futures = {pool.submit(load_image, url): (i, url) for i, url in enumerate(urls)}
    successes: list[tuple[int, Image.Image]] = []
    failures: list[FetchFailure] = []
    for fut, (idx, url) in futures.items():
        try:
            successes.append((idx, fut.result()))
        except ImageLoadError as e:
            failures.append(FetchFailure(url=url, error=str(e)))
        except Exception as e:  # noqa: BLE001
            failures.append(FetchFailure(url=url, error=f"unexpected: {e}"))
    successes.sort(key=lambda x: x[0])
    return successes, failures


def _run_single(model_key: Literal["gender", "category"], url: str, tta: bool) -> dict:
    try:
        image = load_image(url)
    except ImageLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start = time.perf_counter()
    result = state[model_key].predict(image, tta=tta)
    result["url"] = url
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result


def _run_batch(model_keys: list[Literal["gender", "category"]], urls: list[str], tta: bool) -> dict:
    if len(urls) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(urls)} exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}",
        )

    start = time.perf_counter()
    successes, failures = _fetch_many(urls)

    results_by_model: dict[str, list[dict]] = {k: [] for k in model_keys}
    if successes:
        indices, images = zip(*successes)
        for key in model_keys:
            preds = state[key].predict_batch(list(images), tta=tta)
            for orig_idx, pred in zip(indices, preds):
                pred["url"] = urls[orig_idx]
                results_by_model[key].append(pred)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "count": len(urls),
        "succeeded": len(successes),
        "failed": len(failures),
        "failures": [f.model_dump() for f in failures],
        "results": results_by_model if len(model_keys) > 1 else results_by_model[model_keys[0]],
        "latency_ms": elapsed_ms,
        "tta": tta,
    }


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Kreeda ICH-Next Model API",
        "version": app.version,
        "endpoints": [
            "GET  /health",
            "POST /predict/gender",
            "POST /predict/category",
            "POST /predict",
            "POST /predict/gender/batch",
            "POST /predict/category/batch",
            "POST /predict/batch",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models": {
            "gender": {"loaded": "gender" in state, "classes": state["gender"].classes},
            "category": {"loaded": "category" in state, "classes": state["category"].classes},
        },
        "max_batch_size": MAX_BATCH_SIZE,
    }


@app.post("/predict/gender")
def predict_gender(req: PredictRequest):
    return _run_single("gender", req.url, req.tta)


@app.post("/predict/category")
def predict_category(req: PredictRequest):
    return _run_single("category", req.url, req.tta)


@app.post("/predict")
def predict_both(req: PredictRequest):
    try:
        image = load_image(req.url)
    except ImageLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    start = time.perf_counter()
    return {
        "url": req.url,
        "gender": state["gender"].predict(image, tta=req.tta),
        "category": state["category"].predict(image, tta=req.tta),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }


@app.post("/predict/gender/batch")
def predict_gender_batch(req: BatchRequest):
    return _run_batch(["gender"], req.urls, req.tta)


@app.post("/predict/category/batch")
def predict_category_batch(req: BatchRequest):
    return _run_batch(["category"], req.urls, req.tta)


@app.post("/predict/batch")
def predict_both_batch(req: BatchRequest):
    return _run_batch(["gender", "category"], req.urls, req.tta)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(int(os.getenv("RELOAD", "0"))),
    )
