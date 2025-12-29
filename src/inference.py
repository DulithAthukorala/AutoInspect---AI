from ultralytics import YOLO

class DamageDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image):
        results = self.model(
            image,
            conf=0.25,
            verbose=False
        )
        return results[0]
