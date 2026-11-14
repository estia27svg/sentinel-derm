"""FastAPI application serving the triage model and the static dashboard."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.inference import TriageService

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
ARTIFACTS = ROOT / "artifacts"
IMAGES = ROOT / "data" / "processed" / "images"
SAMPLE_IMAGES = ARTIFACTS / "sample_images"

MAX_BYTES = 12 * 1024 * 1024
MAX_SIDE = 1600

(ARTIFACTS / "plots").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sentinel Derm", version="1.0.0")
service: TriageService | None = None
service_error: str | None = None

try:
    service = TriageService(artifacts=str(ARTIFACTS), image_root=str(IMAGES))
except Exception as error:  # surfaced through /api/health instead of crashing the server
    service_error = f"{type(error).__name__}: {error}"


def _read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@app.get("/api/health")
def health():
    return {
        "ok": service is not None,
        "error": service_error,
        "model": service.model_label if service else None,
        "classes": service.metrics["classifier"]["support"] if service and service.metrics else None,
    }


@app.get("/api/metrics")
def metrics():
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="metrics.json not found - run training/train.py first")
    return _read_json(path)


@app.get("/api/samples")
def samples():
    path = ARTIFACTS / "samples.json"
    if not path.exists():
        return {"samples": []}
    payload = _read_json(path)
    for sample in payload:
        sample["image_url"] = f"/api/samples/{sample['image_id']}/image"
    return {"samples": payload}


@app.get("/api/samples/{image_id}/image")
def sample_image(image_id: str):
    safe = "".join(character for character in image_id if character.isalnum() or character in "_-")
    for folder in (SAMPLE_IMAGES, IMAGES):
        path = folder / f"{safe}.jpg"
        if path.exists():
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="image not found")


@app.get("/api/demo/ood-image")
def ood_image():
    bundled = SAMPLE_IMAGES / "ood_demo.jpg"
    if bundled.exists():
        return FileResponse(bundled, media_type="image/jpeg")
    candidates = sorted((ROOT / "data" / "ood" / "cifar").glob("*.jpg"))
    if not candidates:
        raise HTTPException(status_code=404, detail="OOD demo image not found")
    return FileResponse(candidates[0], media_type="image/jpeg")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts are not ready. {service_error or ''}".strip(),
        )
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    payload = await file.read()
    if len(payload) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB limit.")
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {error}") from error

    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    try:
        result = service.analyze(image)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error
    return JSONResponse(result)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
app.mount("/plots", StaticFiles(directory=str(ARTIFACTS / "plots")), name="plots")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
