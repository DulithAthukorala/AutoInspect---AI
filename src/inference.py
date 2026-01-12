from ultralytics import YOLO
from typing import Any


class DamageDetector:
    """
    - import YOLO model
    - make predictions on images
    """

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.model_path = model_path  # used for logging

    def predict(self, image: Any):
        """
        Run the model on the input image.
        Returns results of that image
        results[0] -> bcz results is a list
        """
        results = self.model(
            image,
            conf=0.25,
            verbose=False
        )
        return results[0]
