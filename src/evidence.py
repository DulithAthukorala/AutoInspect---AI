import numpy as np
from src.logic import CaseEvidence, DamageInstance

def extract_evidence(yolo_result, image_id: str):
    damages = []

    if yolo_result.masks is None:
        return CaseEvidence(
            image_id=image_id,
            damages=[],
            overlaps=None
        )

    masks = yolo_result.masks.data.cpu().numpy()
    classes = yolo_result.boxes.cls.cpu().numpy()
    confidences = yolo_result.boxes.conf.cpu().numpy()
    names = yolo_result.names

    img_h, img_w = yolo_result.orig_shape[:2]

    for mask, cls_id, conf in zip(masks, classes, confidences):
        mask_area = np.sum(mask > 0.5)
        area_ratio = mask_area / (img_h * img_w)

        damages.append(
            DamageInstance(
                damage_type=names[int(cls_id)],
                confidence=float(conf),
                area_ratio=float(area_ratio)
            )
        )

    return CaseEvidence(
        image_id=image_id,
        damages=damages,
        overlaps=None
    )
