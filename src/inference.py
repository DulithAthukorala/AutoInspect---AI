from ultralytics import YOLO
import numpy as np

class DamageDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path):
        results = self.model(image_path, conf=0.25)
        return results[0]  #results is a list, we return the first item 
