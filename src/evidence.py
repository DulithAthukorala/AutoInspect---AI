import numpy as np

def extract_evidence(yolo_result):
    evidence = {
        "num_damages": 0,
        "damages": [],
        "total_damage_ratio": 0.0
    }

    if yolo_result.masks is None:
        return evidence

    # converting YOLO PyTorch tensor results to numpy arrays
    masks = yolo_result.masks.data.cpu().numpy()        # (N, H, W)
    classes = yolo_result.boxes.cls.cpu().numpy()       # (N,)
    confidences = yolo_result.boxes.conf.cpu().numpy()  # (N,)
    names = yolo_result.names

    for mask, cls_id, conf in zip(masks, classes, confidences):
        mask_area = np.sum(mask > 0.5)  # pixels predicted as damage
        mask_h, mask_w = mask.shape
        area_ratio = mask_area / (mask_h * mask_w)

        damage = {
            "type": names[int(cls_id)],
            "confidence": float(conf),
            "area_ratio": float(area_ratio)
        }

        evidence["damages"].append(damage)
        evidence["total_damage_ratio"] += area_ratio

    evidence["num_damages"] = len(evidence["damages"])
    return evidence
