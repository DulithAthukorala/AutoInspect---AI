import io
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

from src.inference import DamageDetector
from src.vehicle_mask import VehicleMasker
from src.evidence import extract_evidence
from src.logic import decide_case, CaseEvidence, Decision
from src.explain import generate_explanation

from src.db import init_db, insert_case, get_case
from src.storage import save_uploaded_image


# ----------------------------
# Config
# ----------------------------
DEFAULT_WEIGHTS = "runs/segment/train/weights/best.pt"
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", DEFAULT_WEIGHTS)
VEHICLE_WEIGHTS = os.getenv("VEHICLE_WEIGHTS", "yolov8n-seg.pt")

# Phase-1 (simple versioning; later we’ll move to YAML)
THRESHOLDS_VERSION = os.getenv("THRESHOLDS_VERSION", "v1")


# ----------------------------
# Response Schemas
# ----------------------------
class DamageOut(BaseModel):
    damage_type: str
    confidence: float
    area_ratio: float  # vs vehicle


class DecisionOut(BaseModel):
    severity: Optional[str]
    route: str
    confidence_score: float

    # Pricing must be None unless AUTO
    estimated_cost_lkr: Optional[int] = None
    cost_range_lkr: Optional[Tuple[int, int]] = None

    pricing_mode: str
    flags: Optional[List[str]] = None
    reasons: List[str]


class AssessResponse(BaseModel):
    case_id: str
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


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


def _overlaps_to_jsonable(overlaps: Optional[Dict[Tuple[int, int], float]]) -> Optional[Dict[str, float]]:
    if not overlaps:
        return None
    # tuple keys -> "i-j"
    return {f"{i}-{j}": float(v) for (i, j), v in overlaps.items()}


def to_jsonable_decision(d: Decision) -> Dict[str, Any]:
    # Enforce Phase-1 guarantee: no pricing unless AUTO
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
        "pricing_mode": getattr(d, "pricing_mode", "AUTO_POINT" if d.route == "AUTO" else "PENDING_REVIEW"),
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
        "overlaps": _overlaps_to_jsonable(getattr(e, "overlaps", None)),
    }


def run_pipeline(img: Image.Image, image_id: str) -> Dict[str, Any]:
    detector, vehicle_masker = get_models()

    # 1) Vehicle mask
    vehicle_mask = vehicle_masker.predict_vehicle_mask(img)

    # 2) Damage prediction
    yolo_res = detector.predict(img)

    # 3) Evidence
    evidence = extract_evidence(yolo_res, image_id=image_id, vehicle_mask=vehicle_mask)

    # 4) Decision
    decision = decide_case(evidence)

    # 5) Explanation
    explanation = generate_explanation(evidence, decision)

    return {
        **to_jsonable_evidence(evidence),
        "decision": to_jsonable_decision(decision),
        "explanation": explanation,
        "meta": {
            "weights_path": WEIGHTS_PATH,
            "vehicle_weights": VEHICLE_WEIGHTS,
            "thresholds_version": THRESHOLDS_VERSION,
        }
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
    case_id = str(uuid.uuid4())

    # Run pipeline
    payload = run_pipeline(img, image_id=image_id)

    # Save image + DB record
    image_path, image_hash = save_uploaded_image(case_id, img, raw)
    insert_case(
        case_id=case_id,
        image_path=image_path,
        image_sha256=image_hash,
        weights_path=WEIGHTS_PATH,
        vehicle_weights=VEHICLE_WEIGHTS,
        thresholds_version=THRESHOLDS_VERSION,
        response_json={**payload, "case_id": case_id},
    )

    # Return response
    out = {**payload, "case_id": case_id}
    return JSONResponse(out)


@app.get("/case/{case_id}")
def read_case(case_id: str):
    row = get_case(case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    return JSONResponse(row.response_json)


@app.get("/report/{case_id}")
def download_report(case_id: str):
    row = get_case(case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Case not found.")

    # Downloadable JSON
    content = row.response_json
    filename = f"autoinspect_report_{case_id}.json"
    return Response(
        content=JSONResponse(content).body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/replay/{case_id}")
def replay_case(case_id: str):
    """
    Re-runs the full pipeline on the stored image to verify reproducibility.
    (True reproducibility requires pinned model + thresholds versions.)
    """
    row = get_case(case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Case not found.")

    try:
        img = Image.open(row.image_path).convert("RGB")
    except Exception:
        raise HTTPException(status_code=500, detail="Stored image could not be loaded for replay.")

    # Re-run
    image_id = row.response_json.get("image_id", "replay")
    payload = run_pipeline(img, image_id=image_id)

    # Return replay output (do not overwrite DB in Phase-1)
    return JSONResponse({
        "case_id": case_id,
        "replay": payload,
        "original_meta": {
            "weights_path": row.weights_path,
            "vehicle_weights": row.vehicle_weights,
            "thresholds_version": row.thresholds_version,
            "image_sha256": row.image_sha256,
            "created_at": row.created_at,
        }
    })
