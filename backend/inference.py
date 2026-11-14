"""Inference pipeline: backbone -> linear head -> conformal set -> OOD gate -> Grad-CAM."""

from __future__ import annotations

import base64
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from common import CLASS_INFO, CLASS_NAMES, URGENCY_ORDER
from training.backbone import FeatureExtractor, build_transform

CAM_STOPS = np.array(
    [
        [10, 15, 60],
        [30, 110, 200],
        [70, 200, 210],
        [250, 220, 70],
        [230, 70, 50],
    ],
    dtype=np.float32,
)


class TriageService:
    def __init__(
        self,
        artifacts: str = "artifacts",
        image_root: str = "data/processed/images",
        quality_path: str = "data/processed/quality.json",
        image_size: int = 224,
    ) -> None:
        self.artifacts = artifacts
        self.image_root = image_root
        self.image_size = image_size

        head = np.load(os.path.join(artifacts, "head.npz"), allow_pickle=False)
        self.coef = torch.from_numpy(head["coef"].astype(np.float32))
        self.intercept = torch.from_numpy(head["intercept"].astype(np.float32))
        self.scaler_mean = head["scaler_mean"].astype(np.float64)
        self.scaler_scale = head["scaler_scale"].astype(np.float64)
        self.backbone_name = str(head["backbone"]) if "backbone" in head else "efficientnet_b0"

        with open(os.path.join(artifacts, "conformal.json"), encoding="utf-8") as handle:
            calibration = json.load(handle)
        self.alpha = float(calibration["alpha"])
        self.quantiles = np.asarray(calibration["quantiles"], dtype=np.float64)
        self.n_calibration = int(calibration.get("n_calibration", 0))

        ood_payload = np.load(os.path.join(artifacts, "ood.npz"), allow_pickle=False)
        self.ood_means = ood_payload["means"].astype(np.float64)
        self.ood_precision = ood_payload["precision"].astype(np.float64)
        # Serve at the 99th percentile (1% in-distribution rejection). The 95th
        # percentile is reported in the evaluation as the stricter operating point.
        self.ood_threshold = float(ood_payload["threshold_99"])

        self.min_dimension = 96
        self.sharpness_floor = 15.0
        bundled_quality = os.path.join(artifacts, "quality.json")
        if not os.path.exists(quality_path) and os.path.exists(bundled_quality):
            quality_path = bundled_quality
        if os.path.exists(quality_path):
            with open(quality_path, encoding="utf-8") as handle:
                quality = json.load(handle)
            self.min_dimension = int(quality.get("min_dimension", 96))
            self.sharpness_floor = float(quality.get("sharpness_p01", 15.0))

        self.backbone = FeatureExtractor(self.backbone_name, pretrained=True)
        self.backbone.eval()
        self.transform = build_transform(image_size)
        self.model_label = self._describe_backbone()

        metrics_path = os.path.join(artifacts, "metrics.json")
        self.metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as handle:
                self.metrics = json.load(handle)

    def _describe_backbone(self) -> str:
        names = {
            "efficientnet_b0": "EfficientNet-B0",
            "mobilenet_v3_large": "MobileNetV3-Large",
            "resnet18": "ResNet-18",
            "convnext_tiny": "ConvNeXt-Tiny",
        }
        return names.get(self.backbone_name, self.backbone_name)

    # ------------------------------------------------------------------ quality

    def _quality(self, image: Image.Image) -> dict:
        width, height = image.size
        gray = image.convert("L")
        edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
        # the convolution overshoots on the image border, so ignore a 2px frame
        if edges.shape[0] > 4 and edges.shape[1] > 4:
            edges = edges[2:-2, 2:-2]
        sharpness = float(edges.var())
        issues = []
        issue_codes = []
        if min(width, height) < self.min_dimension:
            issues.append(f"resolution {width}x{height} is below the {self.min_dimension}px minimum")
            issue_codes.append(
                {
                    "code": "resolution",
                    "width": width,
                    "height": height,
                    "minimum": self.min_dimension,
                }
            )
        if sharpness < self.sharpness_floor:
            issues.append("image appears out of focus")
            issue_codes.append(
                {
                    "code": "focus",
                    "sharpness": round(sharpness, 2),
                    "floor": round(self.sharpness_floor, 2),
                }
            )
        return {
            "ok": not issues,
            "width": width,
            "height": height,
            "sharpness": round(sharpness, 2),
            "sharpness_floor": round(self.sharpness_floor, 2),
            "issues": issues,
            "issue_codes": issue_codes,
        }

    # ------------------------------------------------------------ model passes

    def _logits_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.scaler_mean).to(pooled.dtype).to(pooled.device)
        scale = torch.from_numpy(self.scaler_scale).to(pooled.dtype).to(pooled.device)
        normalised = (pooled - mean) / scale
        return normalised @ self.coef.T + self.intercept

    def _ood_score(self, pooled: np.ndarray) -> float:
        scaled = (pooled.astype(np.float64) - self.scaler_mean) / self.scaler_scale
        distances = np.empty(len(self.ood_means), dtype=np.float64)
        for index, centre in enumerate(self.ood_means):
            delta = scaled - centre
            distances[index] = float(delta @ self.ood_precision @ delta)
        return float(distances.min())

    def _gradcam(self, feature_map: torch.Tensor, logits: torch.Tensor, target: int) -> np.ndarray:
        score = logits[0, target]
        gradients = torch.autograd.grad(score, feature_map, retain_graph=False)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().numpy()
        cam = cam - cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam = cam / maximum
        return cam

    @staticmethod
    def _colorize(cam: np.ndarray) -> np.ndarray:
        positions = np.linspace(0.0, 1.0, len(CAM_STOPS))
        channels = [np.interp(cam, positions, CAM_STOPS[:, channel]) for channel in range(3)]
        return np.stack(channels, axis=-1).astype(np.uint8)

    def _overlay(self, image: Image.Image, cam: np.ndarray) -> str:
        width = self.image_size
        height = max(1, int(round(width * image.height / image.width)))
        base = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR)).astype(np.float32)
        cam_image = Image.fromarray((cam * 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
        cam = np.asarray(cam_image, dtype=np.float32) / 255.0
        coloured = self._colorize(cam).astype(np.float32)
        weight = (cam > 0.15)[..., None] * 0.55
        blended = base * (1 - weight) + coloured * weight
        buffer = io.BytesIO()
        Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(buffer, "PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    # --------------------------------------------------------------- public API

    def analyze(self, image: Image.Image, include_cam: bool = True) -> dict:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        image = image.convert("RGB")
        quality = self._quality(image)

        tensor = self.transform(image).unsqueeze(0)
        preprocess_ms = (time.perf_counter() - started) * 1000

        if quality["ok"] and include_cam:
            tensor.requires_grad_(True)
            pooled, feature_map = self.backbone(tensor)
            logits = self._logits_from_pooled(pooled)
        else:
            with torch.inference_mode():
                pooled, feature_map = self.backbone(tensor)
                logits = self._logits_from_pooled(pooled)
        model_ms = (time.perf_counter() - started) * 1000 - preprocess_ms

        probs = torch.softmax(logits.detach(), dim=1)[0].numpy().astype(np.float64)
        pooled_np = pooled.detach()[0].numpy()
        ood_score = self._ood_score(pooled_np)

        in_distribution = ood_score <= self.ood_threshold
        prediction_set = [index for index in range(len(CLASS_NAMES)) if probs[index] >= 1.0 - self.quantiles[index]]

        if not quality["ok"]:
            status = "low_quality"
        elif not in_distribution:
            status = "out_of_distribution"
        elif len(prediction_set) == 0:
            status = "uncertain"
        elif len(prediction_set) == 1:
            status = "confident"
        else:
            status = "ambiguous"

        triage = self._triage(status, prediction_set)
        explanation, explanation_code, explanation_params = self._explain(
            status, prediction_set, probs, quality, ood_score
        )

        cam_data = None
        if include_cam and status in {"confident", "ambiguous"}:
            target = int(np.argmax(probs))
            cam = self._gradcam(feature_map, logits, target)
            cam_data = self._overlay(image, cam)
        cam_ms = (time.perf_counter() - started) * 1000 - preprocess_ms - model_ms

        probabilities = sorted(
            (
                {
                    "code": name,
                    "name": CLASS_INFO[name]["full"],
                    "short": CLASS_INFO[name]["short"],
                    "probability": round(float(probs[index]), 4),
                    "in_set": index in prediction_set,
                }
                for index, name in enumerate(CLASS_NAMES)
            ),
            key=lambda item: item["probability"],
            reverse=True,
        )

        total_ms = (time.perf_counter() - started) * 1000
        return {
            "status": status,
            "quality": quality,
            "ood": {
                "in_distribution": in_distribution,
                "score": round(ood_score, 3),
                "threshold": round(self.ood_threshold, 3),
            },
            "prediction_set": [CLASS_NAMES[index] for index in prediction_set],
            "probabilities": probabilities,
            "triage": triage,
            "explanation": explanation,
            "explanation_code": explanation_code,
            "explanation_params": explanation_params,
            "gradcam": cam_data,
            "guarantee": {
                "alpha": self.alpha,
                "target_coverage": round(1.0 - self.alpha, 3),
                "n_calibration": self.n_calibration,
                "method": "class-conditional split conformal prediction",
            },
            "timings_ms": {
                "preprocess": round(preprocess_ms, 1),
                "model": round(model_ms, 1),
                "gradcam": round(max(cam_ms, 0.0), 1),
                "total": round(total_ms, 1),
            },
            "model": {
                "backbone": self.model_label,
                "classes": CLASS_NAMES,
            },
        }

    def _triage(self, status: str, prediction_set: list[int]) -> dict:
        if status == "low_quality":
            return {
                "urgency": "none",
                "code": "not_assessed",
                "next_steps": ["retake_focused", "even_lighting", "consult_if_changing"],
                "label": "Not assessed",
                "advice": "No diagnosis was produced for this image.",
            }
        if status == "out_of_distribution":
            return {
                "urgency": "none",
                "code": "not_assessed",
                "next_steps": ["dermoscopy_only", "consult_clinician"],
                "label": "Not assessed",
                "advice": "No diagnosis was produced for this image.",
            }
        if not prediction_set:
            return {
                "urgency": "urgent",
                "code": "uncertain_refer",
                "next_steps": ["consult_clinician"],
                "label": "Uncertain - refer",
                "advice": "The model could not form a reliable differential. Escalate for clinical review.",
            }
        urgency = max((CLASS_INFO[CLASS_NAMES[index]]["urgency"] for index in prediction_set), key=lambda key: URGENCY_ORDER[key])
        selected = [
            (index, CLASS_INFO[CLASS_NAMES[index]])
            for index in prediction_set
            if CLASS_INFO[CLASS_NAMES[index]]["urgency"] == urgency
        ]
        index, info = selected[0]
        return {
            "urgency": urgency,
            "code": "class",
            "class_code": CLASS_NAMES[index],
            "next_steps": [] if len(prediction_set) == 1 else ["consult_clinician"],
            "label": info["urgency_label"],
            "advice": info["advice"],
        }

    @staticmethod
    def _explain(status, prediction_set, probs, quality, ood_score):
        if status == "low_quality":
            text = "The image did not pass the input quality check, so no diagnosis was generated. " + "; ".join(quality["issues"]).capitalize() + "."
            return text, "low_quality", {"issues": [item["code"] for item in quality.get("issue_codes", [])]}
        if status == "out_of_distribution":
            return (
                "This image does not resemble dermoscopic skin images the model was trained on, "
                "so it was rejected instead of being force-fitted to a diagnosis. "
                "In a deployment this would prompt the user to retake the photo or see a clinician.",
                "out_of_distribution",
                {},
            )
        names = [CLASS_NAMES[index] for index in prediction_set]
        if status == "uncertain":
            return (
                "Every candidate fell below its calibrated threshold, so the prediction set is empty. "
                "The model is signalling that none of its known diagnoses explains this image well.",
                "uncertain",
                {},
            )
        if status == "confident":
            name = CLASS_INFO[names[0]]["full"]
            probability = probs[prediction_set[0]]
            return (
                f"The 90% prediction set contains a single diagnosis: {name} ({probability:.0%} model probability). "
                "Class-conditional conformal calibration means that, across the validated test set, the true label "
                "falls inside such sets at least 90% of the time.",
                "confident",
                {"class_code": names[0], "probability": round(float(probability), 4)},
            )
        listed = ", ".join(CLASS_INFO[name]["short"] for name in names)
        text = (
            f"At the 90% coverage level the model cannot separate {listed}. "
            "Rather than guessing one label, it returns all of them as a prediction set and recommends clinical review - "
            "this is the intended behaviour on genuinely ambiguous lesions."
        )
        return text, "ambiguous", {"class_codes": names}
