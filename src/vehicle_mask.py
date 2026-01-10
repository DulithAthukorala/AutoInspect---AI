import numpy as np
from ultralytics import YOLO
from typing import Optional


class VehicleMasker:
    """
    Vehicle segmentation wrapper.
    Responsibility:
    - detect vehicle
    - return binary mask (H x W)
    - return None if vehicle not confidently detected
    """

    # COCO vehicle class IDs
    VEHICLE_CLASS_IDS = {
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
    }

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict_vehicle_mask(self, image) -> Optional[np.ndarray]:
        """
        Returns:
        - binary vehicle mask (uint8 {0,1})
        - None if no reliable vehicle found
        """
        results = self.model(image, conf=0.25, verbose=False)
        r = results[0]

        if r.masks is None or r.boxes is None:
            return None

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        vehicle_masks = []

        for mask, cls_id, conf in zip(masks, classes, confidences):
            if int(cls_id) in self.VEHICLE_CLASS_IDS and conf >= 0.35:
                vehicle_masks.append(mask > 0.5)

        if not vehicle_masks:
            return None

        # Union all vehicle masks
        combined = np.logical_or.reduce(vehicle_masks)
        return combined.astype(np.uint8)
