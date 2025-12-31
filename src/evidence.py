import numpy as np
from src.logic import CaseEvidence, DamageInstance

def extract_evidence(yolo_result, image_id: str, vehicle_mask: np.ndarray | None):
    damages = []
    img_h, img_w = yolo_result.orig_shape[:2]
    img_area = img_h * img_w

    # vehicle pixels (fallback to full image if missing)
    if vehicle_mask is None:
        vehicle_area = img_area
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

    for mask, cls_id, conf in zip(masks, classes, confidences):
        bin_mask = (mask > 0.5)
        damage_pixels = int(bin_mask.sum())

        area_ratio_vehicle = damage_pixels / max(vehicle_area, 1)

        damages.append(
            DamageInstance(
                damage_type=names[int(cls_id)],
                confidence=float(conf),
                area_ratio=float(area_ratio_vehicle),
            )
        )

    return CaseEvidence(
        image_id=image_id,
        damages=damages,
        overlaps=None,  # keep overlaps for REAL overlaps only
        vehicle_area_ratio=vehicle_area_ratio
    )
