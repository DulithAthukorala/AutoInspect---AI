import io
import os
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from src.inference import DamageDetector
from src.vehicle_mask import VehicleMasker
from src.evidence import extract_evidence
from src.logic import decide_case, CaseEvidence, Decision
from src.explain import generate_explanation


# ----------------------------
# Config
# ----------------------------
DEFAULT_WEIGHTS = "runs/segment/train/weights/best.pt"
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", DEFAULT_WEIGHTS)
VEHICLE_WEIGHTS = os.getenv("VEHICLE_WEIGHTS", "yolov8n-seg.pt")


# ----------------------------
# Response Schemas
# ----------------------------
class DamageOut(BaseModel):
    damage_type: str
    confidence: float
    area_ratio: float  # vs vehicle


class DecisionOut(BaseModel):
    severity: str
    route: str
    confidence_score: float
    estimated_cost_lkr: int
    cost_range_lkr: Optional[Tuple[int, int]] = None
    pricing_mode: Optional[str] = None
    flags: Optional[List[str]] = None
    reasons: List[str]


class AssessResponse(BaseModel):
    image_id: str
    vehicle_area_ratio: Optional[float]
    damages: List[DamageOut]
    decision: DecisionOut
    explanation: str


# ----------------------------
# Model singletons (loaded once)
# ----------------------------
_detector: Optional[DamageDetector] = None
_vehicle_masker: Optional[VehicleMasker] = None


def get_models() -> tuple[DamageDetector, VehicleMasker]:
    global _detector, _vehicle_masker
    if _detector is None:
        _detector = DamageDetector(WEIGHTS_PATH)
    if _vehicle_masker is None:
        _vehicle_masker = VehicleMasker(VEHICLE_WEIGHTS)
    return _detector, _vehicle_masker


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="AutoInspect AI API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


def to_jsonable_decision(d: Decision) -> Dict[str, Any]:
    return {
        "severity": d.severity,
        "route": d.route,
        "confidence_score": float(d.confidence_score),
        "estimated_cost_lkr": int(d.estimated_cost_lkr),
        "cost_range_lkr": getattr(d, "cost_range_lkr", None),
        "pricing_mode": getattr(d, "pricing_mode", None),
        "flags": getattr(d, "flags", None),
        "reasons": list(d.reasons),
    }


@app.post("/assess", response_model=AssessResponse)
async def assess_image(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG image.")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    image_id = file.filename or "upload"

    detector, vehicle_masker = get_models()

    # 1) Vehicle mask
    vehicle_mask = vehicle_masker.predict_vehicle_mask(img)

    # 2) Damage prediction
    yolo_res = detector.predict(img)

    # 3) Evidence (numeric)
    evidence = extract_evidence(yolo_res, image_id=image_id, vehicle_mask=vehicle_mask)

    # 4) Decision (deterministic)
    decision = decide_case(evidence)

    # 5) Explanation (human-facing, derived only from evidence + decision)
    explanation = generate_explanation(evidence, decision)

    payload = {
        **to_jsonable_evidence(evidence),
        "decision": to_jsonable_decision(decision),
        "explanation": explanation,
    }
    return JSONResponse(payload)

