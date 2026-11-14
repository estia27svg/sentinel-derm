"""Educational guidance for ordinary smartphone photos.

This module deliberately does not load or call the dermoscopy model. It only
performs transparent image checks and returns structured guidance codes so the
interface can explain what is and is not being assessed.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageFilter

MIN_DIMENSION = 96
SHARPNESS_WARN = 12.0
SHARPNESS_FAIL = 6.0
DARK_MEAN = 45.0
BRIGHT_MEAN = 225.0
LOW_CONTRAST = 12.0
MIN_SKIN_FRACTION = 0.03


def _sharpness(image: Image.Image) -> float:
    gray = image.convert("L")
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    if edges.shape[0] > 4 and edges.shape[1] > 4:
        edges = edges[2:-2, 2:-2]
    return float(edges.var())


def _skin_fraction(image: Image.Image) -> float:
    small = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32)
    ycbcr = np.asarray(small.convert("YCbCr"), dtype=np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cb, cr = ycbcr[..., 1], ycbcr[..., 2]
    skin = (
        (cb >= 77)
        & (cb <= 127)
        & (cr >= 133)
        & (cr <= 173)
        & (red > 95)
        & (green > 40)
        & (blue > 20)
        & ((np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])) > 15)
        & (np.abs(red - green) > 15)
        & (red > green)
        & (red > blue)
    )
    return float(skin.mean())


def build_guidance(image: Image.Image) -> dict:
    started = time.perf_counter()
    width, height = image.size
    gray = image.convert("L")
    luminance = np.asarray(gray, dtype=np.float64)
    brightness = float(luminance.mean())
    contrast = float(luminance.std())
    sharpness = _sharpness(image)
    skin_fraction = _skin_fraction(image)

    checks = []

    readable = min(width, height) >= MIN_DIMENSION
    checks.append(
        {
            "code": "readability",
            "state": "pass" if readable else "fail",
            "detail": "" if readable else "small",
            "params": {"width": width, "height": height, "minimum": MIN_DIMENSION},
        }
    )

    if sharpness < SHARPNESS_FAIL:
        focus_state, focus_detail = "fail", "blurred"
    elif sharpness < SHARPNESS_WARN:
        focus_state, focus_detail = "warn", "blurred"
    else:
        focus_state, focus_detail = "pass", ""
    checks.append(
        {"code": "focus", "state": focus_state, "detail": focus_detail, "params": {"sharpness": round(sharpness, 1)}}
    )

    if brightness < DARK_MEAN:
        light_state, light_detail = "warn", "dark"
    elif brightness > BRIGHT_MEAN:
        light_state, light_detail = "warn", "bright"
    elif contrast < LOW_CONTRAST:
        light_state, light_detail = "warn", "flat"
    else:
        light_state, light_detail = "pass", ""
    checks.append(
        {
            "code": "lighting",
            "state": light_state,
            "detail": light_detail,
            "params": {"brightness": round(brightness, 1), "contrast": round(contrast, 1)},
        }
    )

    skin_present = skin_fraction >= MIN_SKIN_FRACTION
    checks.append(
        {
            "code": "skin",
            "state": "pass" if skin_present else "warn",
            "detail": "" if skin_present else "none",
            "params": {"fraction": round(skin_fraction, 3)},
        }
    )

    if not readable or focus_state == "fail" or light_detail in {"dark", "bright"}:
        status = "guidance_retake"
    elif not skin_present:
        status = "guidance_unclear"
    else:
        status = "guidance_ready"

    return {
        "mode": "smartphone",
        "scope": "guidance_only_no_diagnosis",
        "status": status,
        "quality": {
            "width": width,
            "height": height,
            "sharpness": round(sharpness, 1),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "skin_like_fraction": round(skin_fraction, 3),
            "minimum": MIN_DIMENSION,
        },
        "checks": checks,
        "disclaimer_code": "smartphone_model_not_validated",
        "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
    }
