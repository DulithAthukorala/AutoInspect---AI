from ultralytics import YOLO
from typing import Any


class DamageDetector:
    """
    Thin inference wrapper.
    Responsibility:
    - run YOLO
    - expose raw model outputs
    - DO NOT make decisions
    """

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.model_path = model_path  # used for audit / replay metadata

    def predict(self, image: Any):
        """
        Returns raw YOLO result (single image).
        Detection confidence lives here.
        Other confidence components are computed downstream.
        """
        results = self.model(
            image,
            conf=0.25,
            verbose=False
        )
        return results[0]
