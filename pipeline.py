# pipeline.py (MINIMAL)
from pathlib import Path

from src.inference import DamageDetector
from src.evidence import extract_evidence, _bbox_area_xyxy
from src.logic import decide_case
from src.explain import generate_explanation, explain_change

def run(image, image_id, weights_path):
    detector = DamageDetector(weights_path)

    yolo_res = detector.predict(image)
    evidence = extract_evidence(yolo_res, image_id=image_id)

    decision = decide_case(evidence)

    explanation = generate_explanation(evidence, decision)
    change = explain_change(evidence, decision)

    return evidence, decision, explanation, change


if __name__ == "__main__":
    WEIGHTS = "runs/segment/train/weights/best.pt"   # <-- change if needed
    IMAGE = "data/CarDD_COCO/test2017/000012.jpg" 
    image_id = Path(IMAGE).name

    evidence, decision, explanation, change = run(
        image=IMAGE,
        image_id=image_id,
        weights_path=WEIGHTS
    )

    print(explanation)
    print("\nCHANGE:")
    print(change)
