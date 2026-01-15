from typing import Any
from ultralytics import YOLO

from src.core.weights import ensure_weights


class DamageDetector:
    """
    Loads YOLO weights (local OR auto-download via HF) and runs predictions.
    """

    def __init__(self, model_path: str | None = None):
        # model_path is optional now.
        # If it's missing or doesn't exist, ensure_weights() will download it to /app/models/best.pt
        if model_path:
            # if user passed a path, set MODEL_PATH so ensure_weights uses it
            import os
            os.environ["MODEL_PATH"] = model_path

        resolved_path = ensure_weights()   # <-- THIS is what you were missing
        self.model_path = resolved_path
        self.model = YOLO(resolved_path)

    def predict(self, image: Any):
        results = self.model(image, conf=0.25, verbose=False)
        return results[0]
