"""Tests for the smartphone guidance mode and its separation from the model."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app

client = TestClient(app)


def _image(size=(512, 512), colour=(190, 150, 120), noise=18) -> bytes:
    rng = np.random.default_rng(7)
    base = np.zeros((size[1], size[0], 3), dtype=np.float64)
    base[:] = colour
    base += rng.normal(0, noise, base.shape)
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return buffer.getvalue()


def _post(payload: bytes, name="photo.jpg", content_type="image/jpeg"):
    return client.post("/api/guidance", files={"file": (name, payload, content_type)})


def test_guidance_returns_checks_without_a_diagnosis():
    response = _post(_image())
    assert response.status_code == 200
    payload = response.json()

    assert payload["mode"] == "smartphone"
    assert payload["scope"] == "guidance_only_no_diagnosis"
    assert payload["status"] in {"guidance_ready", "guidance_retake", "guidance_unclear"}
    assert {check["code"] for check in payload["checks"]} == {"readability", "focus", "lighting", "skin"}
    assert "probabilities" not in payload
    assert "prediction_set" not in payload
    assert "triage" not in payload
    assert "gradcam" not in payload


def test_tiny_image_is_flagged_for_retake():
    response = _post(_image(size=(48, 48)))
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "guidance_retake"
    readability = next(check for check in payload["checks"] if check["code"] == "readability")
    assert readability["state"] == "fail"
    assert readability["detail"] == "small"


def test_dark_image_reports_lighting_problem():
    response = _post(_image(colour=(18, 14, 12), noise=4))
    assert response.status_code == 200
    payload = response.json()
    lighting = next(check for check in payload["checks"] if check["code"] == "lighting")
    assert lighting["detail"] == "dark"


def test_non_skin_image_is_unclear_but_still_guidance():
    response = _post(_image(colour=(30, 60, 210), noise=20))
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "guidance_unclear"
    skin = next(check for check in payload["checks"] if check["code"] == "skin")
    assert skin["state"] == "warn"


def test_guidance_rejects_non_image_upload():
    response = _post(b"not an image", name="notes.txt", content_type="text/plain")
    assert response.status_code == 415
