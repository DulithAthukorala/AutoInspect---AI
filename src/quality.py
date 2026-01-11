from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    score: float                 # 0..1
    flags: list[str]
    metrics: Dict[str, float]
    user_message: str


def _to_gray_np(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    return np.asarray(g, dtype=np.float32)


def _variance_of_laplacian(gray: np.ndarray) -> float:
    """
    Fast vectorized Laplacian variance (no OpenCV).
    Higher => sharper.
    """
    g = gray
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0

    center = g[1:-1, 1:-1]
    lap = (
        g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1]
        - 4.0 * center
    )
    return float(lap.var())


def assess_image_quality(img: Image.Image) -> QualityResult:
    """
    Phase-1 quality gate:
      - computes metrics on a resized copy (fast + stable)
      - hard reject ONLY when truly unusable
      - otherwise pass with warning flags

    Goal: reduce false rejects.
    """
    flags: list[str] = []

    # Original size (for reporting / rules)
    w, h = img.size

    # ---- Compute quality metrics on smaller image (speed + stability) ----
    max_side = 800
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        q_img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    else:
        q_img = img

    gray = _to_gray_np(q_img)

    brightness = float(gray.mean())          # 0..255
    contrast = float(gray.std())
    blur_v = _variance_of_laplacian(gray)

    metrics = {
        "width": float(w),
        "height": float(h),
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "blur_var_laplacian": blur_v,
    }

    # ---- Soft flags (warnings only) ----
    if w < 640 or h < 480:
        flags.append("LOW_RESOLUTION")

    if brightness < 50:
        flags.append("DARK")
    elif brightness > 205:
        flags.append("BRIGHT")

    if blur_v < 60:
        flags.append("BLURRY")

    # ---- Hard reject rules (ONLY extreme) ----
    hard_reject = False

    # Too small to be useful at all
    if w < 420 or h < 320:
        hard_reject = True
        flags.append("HARD_FAIL_TOO_SMALL")

    # Extremely dark/bright
    if brightness < 35:
        hard_reject = True
        flags.append("HARD_FAIL_TOO_DARK")
    if brightness > 220:
        hard_reject = True
        flags.append("HARD_FAIL_TOO_BRIGHT")

    # Extremely blurry
    if blur_v < 25:
        hard_reject = True
        flags.append("HARD_FAIL_TOO_BLURRY")

    ok = not hard_reject

    # ---- Score (simple, explainable) ----
    score = 1.0
    if "LOW_RESOLUTION" in flags:
        score -= 0.25
    if "DARK" in flags or "BRIGHT" in flags:
        score -= 0.20
    if "BLURRY" in flags:
        score -= 0.35
    if hard_reject:
        score = min(score, 0.25)
    score = max(0.0, min(1.0, score))

    if ok:
        msg = "Image quality looks acceptable."
    else:
        support = os.getenv("SUPPORT_CONTACT", "+94-XX-XXXXXXX")
        msg = (
            "Image quality is too low for reliable assessment. "
            "Please retake the photo (good lighting, full car in frame, not blurry). "
            f"If you cannot retake it, contact customer support: {support}."
        )

    return QualityResult(ok=ok, score=score, flags=flags, metrics=metrics, user_message=msg)
