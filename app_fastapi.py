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


