from typing import Optional

import numpy as np
from ultralytics import YOLO


class VehicleMasker:
    """
    Vehicle segmentation wrapper.
    - detects vehicle-like classes
    - returns a binary mask (H x W) aligned to original image
    - returns None if no reliable vehicle found
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
        results = self.model(image, conf=0.25, verbose=False)
        r = results[0]

        if r.masks is None or r.boxes is None:
            return None

        masks = r.masks.data.cpu().numpy()  # (N, Hm, Wm)
        classes = r.boxes.cls.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        orig_h, orig_w = r.orig_shape[:2]

        vehicle_masks = []
        for mask, cls_id, conf in zip(masks, classes, confidences):
            if int(cls_id) in self.VEHICLE_CLASS_IDS and float(conf) >= 0.35:
                bin_mask = (mask > 0.5)

                # Ensure mask matches original image shape
                if bin_mask.shape != (orig_h, orig_w):
                    # Resize via nearest-neighbor using numpy indexing (no cv2)
                    bin_mask = self._resize_nn(bin_mask.astype(np.uint8), orig_h, orig_w) > 0

                vehicle_masks.append(bin_mask)

        if not vehicle_masks:
            return None

        combined = np.logical_or.reduce(vehicle_masks)
        return combined.astype(np.uint8)

    @staticmethod
    def _resize_nn(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """
        Nearest-neighbor resize for binary masks without OpenCV.
        """
        in_h, in_w = mask.shape[:2]
        if in_h == out_h and in_w == out_w:
            return mask

        y_idx = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
        x_idx = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
        return mask[y_idx][:, x_idx]
