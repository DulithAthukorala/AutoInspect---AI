import numpy as np
from src.logic import CaseEvidence, DamageInstance

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are boolean masks
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)

def extract_evidence(yolo_result, image_id: str, vehicle_mask: np.ndarray | None):
    damages: list[DamageInstance] = []
    overlaps: dict[tuple[int, int], float] = {}

    img_h, img_w = yolo_result.orig_shape[:2]
    img_area = img_h * img_w

    # Vehicle pixels
    if vehicle_mask is None:
        vehicle_area = None
        vehicle_area_ratio = None  # unknown
    else:
        vehicle_area = int(vehicle_mask.sum())
        vehicle_area_ratio = vehicle_area / img_area

    if yolo_result.masks is None:
        return CaseEvidence(
            image_id=image_id,
            damages=[],
            overlaps=None,
            vehicle_area_ratio=vehicle_area_ratio
        )

    masks = yolo_result.masks.data.cpu().numpy()
    classes = yolo_result.boxes.cls.cpu().numpy()
    confidences = yolo_result.boxes.conf.cpu().numpy()
    names = yolo_result.names

    bin_masks: list[np.ndarray] = []

    for mask, cls_id, conf in zip(masks, classes, confidences):
        bin_mask = (mask > 0.5)
        damage_pixels = int(bin_mask.sum())

        # IMPORTANT: Only compute area_ratio if vehicle mask exists.
        # If vehicle mask is missing, we keep the detection but mark area_ratio as 0
        # so logic doesn't accidentally treat image-relative area as vehicle-relative.
        if vehicle_area is None or vehicle_area <= 0:
            area_ratio_vehicle = 0.0
        else:
            area_ratio_vehicle = damage_pixels / max(vehicle_area, 1)

        damages.append(
            DamageInstance(
                damage_type=names[int(cls_id)],
                confidence=float(conf),
                area_ratio=float(area_ratio_vehicle),
            )
        )
        bin_masks.append(bin_mask)

    # Compute overlaps (IoU) between damage masks for consistency checks
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
        vehicle_area_ratio=vehicle_area_ratio
    )
