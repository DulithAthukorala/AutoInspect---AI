import cv2
from src.inference import DamageDetector
from src.evidence import extract_evidence

detector = DamageDetector("runs/segment/train/weights/best.pt")

img_path = "yolo_data/images/val/003996.jpg"
img = cv2.imread(img_path)

result = detector.predict(img)
evidence = extract_evidence(result)

print(evidence)
