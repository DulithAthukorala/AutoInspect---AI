from __future__ import annotations

from typing import Optional

import numpy as np

from src.logic import CaseEvidence, DamageInstance


def _resize_nn(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Nearest-neighbor resize for masks without OpenCV.
    mask: 2D array
    """
    in_h, in_w = mask.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return mask

    y_idx = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
    x_idx = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
    return mask[y_idx][:, x_idx]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are boolean masks
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def extract_evidence(yolo_result, image_id: str, vehicle_mask: Optional[np.ndarray]):
    """
    Evidence extraction rules (Phase-1):
    - Always compute damage area ratios deterministically.
    - Prefer vehicle-relative area when vehicle_mask exists.
    - Otherwise fall back to image-relative area (NOT zero).
      (Logic layer already avoids NO_DAMAGE if visibility is insufficient.)
    - Compute overlaps on aligned masks (orig_shape).
    """
    damages: list[DamageInstance] = []
    overlaps: dict[tuple[int, int], float] = {}

    orig_h, orig_w = yolo_result.orig_shape[:2]
    img_area = int(orig_h * orig_w)

    # Vehicle pixels + framing signal
    if vehicle_mask is None:
        vehicle_area = None
        vehicle_area_ratio = None
    else:
        # Ensure vehicle mask aligned to image shape
        if vehicle_mask.shape != (orig_h, orig_w):
            vehicle_mask = (_resize_nn(vehicle_mask.astype(np.uint8), orig_h, orig_w) > 0).astype(np.uint8)
        vehicle_area = int(vehicle_mask.sum())
        vehicle_area_ratio = float(vehicle_area / max(img_area, 1))

        # If vehicle mask exists but is basically empty, treat as missing
        if vehicle_area <= 50:
            vehicle_area = None
            vehicle_area_ratio = None

    # No masks => no damages
    if yolo_result.masks is None or yolo_result.boxes is None:
        return CaseEvidence(
            image_id=image_id,
            damages=[],
            overlaps=None,
            vehicle_area_ratio=vehicle_area_ratio,
        )

    masks = yolo_result.masks.data.cpu().numpy()      # (N, Hm, Wm)
    classes = yolo_result.boxes.cls.cpu().numpy()
    confidences = yolo_result.boxes.conf.cpu().numpy()
    names = yolo_result.names

    bin_masks: list[np.ndarray] = []

    for mask, cls_id, conf in zip(masks, classes, confidences):
        # Align to original image
        if mask.shape != (orig_h, orig_w):
            mask = _resize_nn(mask, orig_h, orig_w)

        bin_mask = (mask > 0.5)
        damage_pixels = int(bin_mask.sum())

        # Area ratio: vehicle-relative preferred, else image-relative
        if vehicle_area is not None and vehicle_area > 0:
            area_ratio = float(damage_pixels / max(vehicle_area, 1))
        else:
            area_ratio = float(damage_pixels / max(img_area, 1))

        damages.append(
            DamageInstance(
                damage_type=names[int(cls_id)],
                confidence=float(conf),
                area_ratio=area_ratio,
            )
        )
        bin_masks.append(bin_mask)

    # Overlaps (IoU) for consistency checks
    n = len(bin_masks)
    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                iou = _mask_iou(bin_masks[i], bin_masks[j])
                if iou > 0:
                    overlaps[(i, j)] = float(iou)

    return CaseEvidence(
        image_id=image_id,
        damages=damages,
        overlaps=overlaps if overlaps else None,
        vehicle_area_ratio=vehicle_area_ratio,
    )
