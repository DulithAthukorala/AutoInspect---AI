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

# (Phase-1 soon) Image quality thresholds (placeholder; we’ll implement in src/quality.py)
ENABLE_QUALITY_GATE = os.getenv("ENABLE_QUALITY_GATE", "0") == "1"


# ----------------------------
# Response Schemas
# ----------------------------
class DamageOut(BaseModel):
    damage_type: str
    confidence: float
    area_ratio: float  # damage / vehicle


class DecisionOut(BaseModel):
    # IMPORTANT: must match logic.py
    severity: Optional[str]  # can be None when visibility is insufficient
    route: str
    confidence_score: float

    # IMPORTANT: pricing must be None unless route == AUTO
    estimated_cost_lkr: Optional[int] = None
    cost_range_lkr: Optional[Tuple[int, int]] = None
    pricing_mode: str

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
    # Enforce Phase-1 guarantee at API boundary too (double safety)
    if d.route != "AUTO":
        est = None
        rng = None
    else:
        est = d.estimated_cost_lkr
        rng = getattr(d, "cost_range_lkr", None)

    return {
        "severity": d.severity,
        "route": d.route,
        "confidence_score": float(d.confidence_score),
        "estimated_cost_lkr": est,
        "cost_range_lkr": rng,
        "pricing_mode": getattr(d, "pricing_mode", "PENDING_REVIEW" if d.route != "AUTO" else "AUTO_POINT"),
        "flags": getattr(d, "flags", None),
        "reasons": list(d.reasons),
    }


def to_jsonable_evidence(e: CaseEvidence) -> Dict[str, Any]:
    return {
        "image_id": e.image_id,
        "vehicle_area_ratio": getattr(e, "vehicle_area_ratio", None),
        "damages": [
            {
                "damage_type": x.damage_type,
                "confidence": float(x.confidence),
                "area_ratio": float(x.area_ratio),
            }
            for x in e.damages
        ],
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

    # (Phase-1 soon) Image-quality gate should happen HERE, before any YOLO.
    # We'll implement src/quality.py next.
    if ENABLE_QUALITY_GATE:
        # placeholder response until quality.py is added
        # raise HTTPException(status_code=400, detail="Quality gate not implemented yet.")
        pass

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
