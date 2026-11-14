# Sentinel Derm

Uncertainty-aware dermoscopy research prototype for **DSH Hacks V2** (AI x Healthcare).

The app has two deliberately separate modes:

- **Phone photo guidance (default).** Ordinary smartphone photos are checked for readability, focus, lighting and skin-like colour, then receive educational guidance. The AI model is **not** run on these photos, and the mode produces no diagnosis or cancer probability. The model was trained on dermoscopic images and is not validated for smartphone photography.
- **Experimental dermoscopy analysis.** Only for dermoscopic images. It returns the set of diagnoses that cannot be ruled out at a **90% coverage guarantee**, abstains when the differential is ambiguous, and rejects images that are blurry or not dermoscopic skin. It is a research prototype, not a clinical tool.

## How the dermoscopy mode works

1. **Frozen backbone** - an ImageNet-pretrained EfficientNet-B0 turns each image into a 1,280-dimensional embedding. A multinomial logistic head is fitted on the training split only.
2. **Class-conditional conformal calibration** - on a separate calibration split, a per-class quantile of the nonconformity score `1 - p(y|x)` is computed with the finite-sample level `ceil((n+1)(1-alpha))/n`. At test time a class enters the prediction set when its probability clears that class's threshold. Calibrating per class keeps the rare melanoma class from being systematically under-covered.
3. **Abstention** - a singleton set is a committed answer; two or more candidates is reported as ambiguity; an empty set means no known diagnosis explains the image. The interface never hides this.
4. **Out-of-domain gate** - inputs are scored by Mahalanobis distance to the nearest class-conditional embedding centre, with the rejection threshold at the 99th percentile of in-distribution calibration scores. The gate rejects non-skin photos instead of force-fitting a diagnosis.
5. **Quality gate** - minimum 96 px on the short side and an edge-variance sharpness floor calibrated at the 1st percentile of the training set.
6. **Grad-CAM** - highlights the region driving the top class, as a user-facing sanity check.

## Held-out results

All numbers come from 2,025 test images the model never saw during training, calibration, or model selection. Splits are made by **lesion**, not by image, so near-duplicate photographs cannot leak between sets.

| Metric | Value |
| --- | --- |
| Accuracy | 74.4% |
| Macro F1 (7 classes) | 45.6% |
| Calibration error (ECE) | 3.7% |
| Empirical coverage at alpha=0.1 | **90.8%** (target 90%) |
| Average prediction-set size | 2.90 |
| Singleton (committed) rate | 28.6% |
| Selective accuracy when committed | 94.3% |
| Melanoma coverage | 88.9% |
| Out-of-domain AUROC (CIFAR-10 negatives) | 0.99997 |

Per-class set coverage on the test split: akiec 94.1%, bcc 98.2%, bkl 91.7%, df 78.3%, mel 88.9%, nv 90.1%, vasc 100%. Dermatofibroma has only 23 test cases, so its 78.3% coverage reflects the finite-sample limit rather than a different calibration rule.

The dataset is HAM10000: 10,015 dermoscopic images, deduplicated and split by lesion into 6,022 train / 1,968 calibration / 2,025 test. EfficientNet-B0 is frozen; the head uses `C=0.003`, no class weighting, selected on training-only cross-validation.

## Run locally

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
python run.py                 # http://127.0.0.1:8000
```

`python run.py --port 9000` changes the port. The server loads `artifacts/` at import time; `GET /api/health` reports whether the model is ready.

Artifacts needed at runtime are committed except the cached embeddings:

```
artifacts/head.npz          logistic head + scaler
artifacts/conformal.json    per-class thresholds
artifacts/ood.npz           Mahalanobis centres + thresholds
artifacts/quality.json      sharpness floor
artifacts/metrics.json      held-out evaluation
artifacts/samples.json      twelve demo cases
artifacts/sample_images/    demo JPEGs + one non-skin photo
artifacts/plots/            evaluation figures for the Evidence view
```

## Reproduce training

```bash
python scripts/download_data.py          # HAM10000 mirror
python training/prepare_data.py          # dedupe, lesion-level splits, 256 px JPEGs
python training/extract_features.py      # cached EfficientNet-B0 embeddings
python scripts/make_ood_set.py           # CIFAR-10 negatives
python training/extract_features.py --source data/ood --out-name embeddings_ood
python training/train.py --skip-search --C 0.003 --class-weight none \
    --ood-embeddings artifacts/embeddings_ood.npz
python -m pytest -q                      # conformal unit tests
```

## Deploy to Render

`render.yaml` defines a free-plan Python web service pinned to Python 3.12. `requirements.txt` pins CPU-only `torch` and `torchvision` from the PyTorch index, so the build never downloads CUDA/NVIDIA wheels. The build command is `pip install --upgrade pip && pip install -r requirements.txt`; the start command is `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`; the health check is `/api/health`.

If the Render service was created manually instead of from the Blueprint, set the same build command, start command, and `PYTHON_VERSION=3.12.7` in the Render dashboard. Push the repository to GitHub and point Render at it, or create a Blueprint from `render.yaml`. No dataset is required in the image: the model artifacts and demo images are bundled.

## Known limitations

- **Skin-tone coverage.** HAM10000 is predominantly lighter-skinned. Performance on darker skin is not established, and models in this space are known to degrade there.
- **Dermoscopy only.** The model was trained on specialist dermoscopic images, not smartphone photos. The out-of-domain gate mitigates this but does not solve it.
- **Seven classes.** The real differential is far wider; a confidently classified image can still be something the model has no class for.
- **Melanoma recall is 29.1% at the argmax threshold.** The conformal set, not the argmax, is the intended output: melanoma falls inside the 90% set for 88.9% of melanoma test cases.
- **Out-of-domain AUROC is optimistic.** CIFAR-10 is an easy negative set; real-world non-dermoscopic phone photos are harder.
- **Not a medical device.** This is a research prototype submitted to a student hackathon and must not be used for clinical decisions.

## Data and licence

HAM10000 (Tschandl, Rosendahl, Kittler, 2018) is used for research. CIFAR-10 is used only as an out-of-domain negative set. No patient-identifiable data is included; all demo images come from the public HAM10000 dataset.
