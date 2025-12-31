from ultralytics import YOLO
import numpy as np

VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle"}  # adjust as you want

class VehicleMasker:
    def __init__(self, model_path: str = "yolov8n-seg.pt"):
        self.model = YOLO(model_path)

    def predict_vehicle_mask(self, image, conf: float = 0.25) -> np.ndarray | None:
        """
        Returns a boolean mask (H,W) for the main vehicle, or None if not found.
        """
        res = self.model(image, conf=conf)[0]
        if res.masks is None:
            return None

        masks = res.masks.data.cpu().numpy()            # (N,H,W) float
        classes = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names

        vehicle_idxs = []
        for i, cls_id in enumerate(classes):
            if names[int(cls_id)] in VEHICLE_NAMES:
                vehicle_idxs.append(i)

        if not vehicle_idxs:
            return None

        # union all vehicle masks
        vehicle = np.zeros(masks[0].shape, dtype=bool)
        for i in vehicle_idxs:
            vehicle |= (masks[i] > 0.5)

        return vehicle
