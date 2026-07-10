"""Lightweight preprocessing tests (no PyTorch / Gradio import)."""

import cv2
import numpy as np


def _analyze_image(img):
    img_uint8 = img.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 128.0
    noise = float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    return np.array([brightness, contrast, noise, edge_density])


def test_analyze_image_returns_four_metrics():
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    metrics = _analyze_image(img)
    assert metrics.shape == (4,)
    assert np.all(np.isfinite(metrics))


def test_clahe_preserves_image_shape():
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)
    assert enhanced.shape == img.shape
