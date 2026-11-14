"""End-to-end checks of the serving API used by the interface."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "artifacts" / "sample_images").glob("*.jpg"))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_and_metrics(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert 0.0 <= payload["conformal"]["coverage"] <= 1.0
    assert set(payload["classifier"]["support"]) == {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


def test_sample_case_returns_structured_result(client):
    assert SAMPLES, "no sample images in artifacts/sample_images"
    response = client.post(
        "/api/analyze",
        files={"file": ("case.jpg", SAMPLES[0].read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] in {"confident", "ambiguous", "uncertain"}
    assert payload["explanation_code"] in {"confident", "ambiguous", "uncertain"}
    assert sum(item["probability"] for item in payload["probabilities"]) == pytest.approx(1.0, abs=1e-4)
    assert len(payload["probabilities"]) == 7
    assert payload["triage"]["code"] in {"class", "uncertain_refer"}
    if payload["status"] == "uncertain":
        assert payload["triage"]["next_steps"] == ["consult_clinician"]
    if payload["status"] == "ambiguous":
        assert len(payload["prediction_set"]) >= 2
        assert "consult_clinician" in payload["triage"]["next_steps"]


def test_low_quality_image_is_rejected_with_actionable_steps(client):
    tiny = Image.new("RGB", (48, 48), (190, 150, 130))
    response = client.post("/api/analyze", files={"file": ("tiny.jpg", _jpeg_bytes(tiny), "image/jpeg")})
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "low_quality"
    assert payload["explanation_code"] == "low_quality"
    assert "resolution" in [issue["code"] for issue in payload["quality"]["issue_codes"]]
    assert "retake_focused" in payload["triage"]["next_steps"]
    assert payload["triage"]["code"] == "not_assessed"


def test_out_of_domain_image_is_rejected(client):
    demo = client.get("/api/demo/ood-image")
    assert demo.status_code == 200
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", demo.content, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "out_of_distribution"
    assert payload["explanation_code"] == "out_of_distribution"
    assert "dermoscopy_only" in payload["triage"]["next_steps"]


def test_non_image_upload_is_rejected(client):
    response = client.post("/api/analyze", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 415
