# Kreeda ICH-Next Model API

FastAPI service exposing two image classifiers used in the ICH-Next pipeline:

- **Gender** — EfficientNet-B0, 3 classes (`men`, `women`, `other`)
- **Category** — EfficientNet-B3, 9 classes (`Western RTW`, `Ethnic RTW`, `Other`, `Streetwear`, `Sarees`, `Sportswear`, `Intimates & Lounge`, `Home Furnishing`, `Beauty`)

Images are fetched on-demand from either `https://` URLs or `s3://bucket/key` URIs.

---

## Project layout

```
model_api/
├── api.py                    # FastAPI entrypoint (routes only)
├── classifiers/              # Inference modules
│   ├── __init__.py
│   ├── gender.py             # GenderInference
│   ├── category.py           # CategoryInference
│   └── image_loader.py       # load_image(url) — s3:// + http(s)
├── ml_models/                # Model checkpoints
│   ├── gender_classifier_best.pt
│   └── category_classifier_best.pt
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and create a virtualenv

```bash
git clone <repo-url>
cd model_api

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU: install a CUDA-matched torch build first. See https://pytorch.org/get-started/locally/.

### 3. Configure env

```bash
cp .env.example .env
# edit .env — at minimum set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
```

Env vars are loaded via `python-dotenv` at process start. All vars are optional except AWS creds (only required if you pass `s3://` URLs).

| Var | Default | Purpose |
| --- | --- | --- |
| `AWS_ACCESS_KEY_ID` | — | S3 access |
| `AWS_SECRET_ACCESS_KEY` | — | S3 access |
| `AWS_REGION` | `us-east-1` | S3 region |
| `HOST` | `0.0.0.0` | uvicorn host |
| `PORT` | `8000` | uvicorn port |
| `DEVICE` | `cpu` | `cpu`, `cuda`, `cuda:0`, ... |
| `MODELS_DIR` | `ml_models` | Directory containing `.pt` files |
| `GENDER_CHECKPOINT` | `gender_classifier_best.pt` | Gender weights filename |
| `CATEGORY_CHECKPOINT` | `category_classifier_best.pt` | Category weights filename |
| `MAX_BATCH_SIZE` | `32` | Max URLs per batch request |
| `FETCH_WORKERS` | `8` | Parallel download workers |
| `HTTP_TIMEOUT_SECONDS` | `20` | Per-request HTTP timeout |
| `MAX_IMAGE_BYTES` | `26214400` (25 MB) | Per-image size cap |

### 4. Place the model weights

Copy the two `.pt` files into `ml_models/`:

```
ml_models/gender_classifier_best.pt
ml_models/category_classifier_best.pt
```

---

## Running

### Dev

```bash
python api.py
# or: uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use `--workers 1` unless you have GPU memory to load the models multiple times. For horizontal scale, run multiple containers behind a load balancer.

Once running, interactive docs: `http://<host>:8000/docs`.

---

## API reference

All `POST` endpoints take JSON. Single endpoints take `{ "url": ..., "tta": bool }`; batch endpoints take `{ "urls": [...], "tta": bool }`.

`tta` (test-time augmentation) runs 5 augmented passes and averages — slower, slightly more accurate.

### `GET /health`
Sanity check. Reports loaded classes and device.

### Single-image inference

| Method | Path | Response |
| --- | --- | --- |
| POST | `/predict/gender` | One gender prediction |
| POST | `/predict/category` | One category prediction |
| POST | `/predict` | Both, one image download |

**Example**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "s3://my-bucket/uploads/img123.jpg", "tta": false}'
```

```json
{
  "url": "s3://my-bucket/uploads/img123.jpg",
  "gender": {
    "prediction": "women",
    "confidence": 0.974,
    "probabilities": {"men": 0.012, "women": 0.974, "other": 0.014},
    "tta": false
  },
  "category": {
    "prediction": "Western RTW",
    "confidence": 0.832,
    "probabilities": { "...": "..." },
    "tta": false
  },
  "latency_ms": 142.7
}
```

### Batch inference

| Method | Path | Response |
| --- | --- | --- |
| POST | `/predict/gender/batch` | List of gender predictions |
| POST | `/predict/category/batch` | List of category predictions |
| POST | `/predict/batch` | Both models, keyed by name |

Downloads run concurrently (`FETCH_WORKERS`); inference runs as a single batched forward pass per model. Failed URLs are reported in `failures[]` and skipped — the batch does not fail as a whole.

**Example**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "s3://my-bucket/a.jpg",
      "https://example.com/b.jpg",
      "s3://my-bucket/missing.jpg"
    ],
    "tta": false
  }'
```

```json
{
  "count": 3,
  "succeeded": 2,
  "failed": 1,
  "failures": [
    {"url": "s3://my-bucket/missing.jpg", "error": "S3 fetch failed ..."}
  ],
  "results": {
    "gender":   [ { "url": "s3://...a.jpg", "prediction": "women", "...": "..." }, { "...": "..." } ],
    "category": [ { "url": "s3://...a.jpg", "prediction": "Sarees", "...": "..." }, { "...": "..." } ]
  },
  "latency_ms": 418.2,
  "tta": false
}
```

---

## Deploying on a fresh VM

```bash
# 1. System deps (Ubuntu example)
sudo apt update && sudo apt install -y python3 python3-venv python3-pip

# 2. App
git clone <repo-url>
cd model_api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Weights — copy the two .pt files into ml_models/
scp gender_classifier_best.pt   vm:~/model_api/ml_models/
scp category_classifier_best.pt vm:~/model_api/ml_models/

# 4. Env
cp .env.example .env  # edit AWS creds + DEVICE

# 5. Run (systemd / pm2 / docker recommended for prod)
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Notes

- First request after startup will be slightly slower — models are loaded in the FastAPI lifespan, so startup takes a few seconds.
- S3 credentials are only needed if you actually pass `s3://` URLs. Plain `https://` URLs work without them.
- On GPU, set `DEVICE=cuda` and make sure your torch build matches the CUDA driver.
