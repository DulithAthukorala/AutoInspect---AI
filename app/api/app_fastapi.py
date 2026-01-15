import io
import os
import uuid
import traceback
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

from src.quality import assess_image_quality
from src.inference import DamageDetector
from src.vehicle_mask import VehicleMasker
from src.evidence import extract_evidence
from src.logic import (
    decide_case,
    CaseEvidence,
    Decision,
    filter_meaningful_damages,
)
from src.explain import generate_explanation
from src.db import init_db, insert_case, get_case
from src.storage import save_uploaded_image


# ----------------------------
# Config
# ----------------------------
DEFAULT_WEIGHTS = "/app/models/best.pt"
WEIGHTS_PATH = os.getenv("MODEL_PATH") or os.getenv("WEIGHTS_PATH", DEFAULT_WEIGHTS)
VEHICLE_WEIGHTS = os.getenv("VEHICLE_WEIGHTS", "yolov8n-seg.pt")

THRESHOLDS_VERSION = os.getenv("THRESHOLDS_VERSION", "v1")
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "+94-77-817-1672")

# Prevent random OOM/500 on large images
MAX_INFER_SIDE = int(os.getenv("MAX_INFER_SIDE", "1280"))


# ----------------------------
# Response Schemas
# ----------------------------
class DamageOut(BaseModel):
    damage_type: str
    confidence: float
    area_ratio: float


class DecisionOut(BaseModel):
    severity: Optional[str]
    route: str
    confidence_score: float
    estimated_cost_lkr: Optional[int] = None
    cost_range_lkr: Optional[Tuple[int, int]] = None
    pricing_mode: str
    flags: Optional[List[str]] = None
    reasons: List[str]


class QualityOut(BaseModel):
    ok: bool
    score: float
    flags: List[str]
    metrics: Dict[str, float]
    user_message: str


class AssessResponse(BaseModel):
    case_id: str
    image_id: str
    quality: QualityOut

    vehicle_area_ratio: Optional[float]
    damages: List[DamageOut]
    decision: DecisionOut
    explanation: str


class BatchTotalOut(BaseModel):
    pricing_mode: str  # AUTO_POINT / AUTO_RANGE / PENDING_REVIEW
    estimated_total_lkr: Optional[int] = None
    total_range_lkr: Optional[Tuple[int, int]] = None
    reason: str


class AssessBatchResponse(BaseModel):
    batch_id: str
    items: List[AssessResponse]
    final_total: BatchTotalOut


# ----------------------------
# Model singletons
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
# Helpers
# ----------------------------
def _resize_for_inference(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((int(w * scale), int(h * scale)))


def _overlaps_to_jsonable(overlaps: Optional[Dict[Tuple[int, int], float]]) -> Optional[Dict[str, float]]:
    if not overlaps:
        return None
    return {f"{i}-{j}": float(v) for (i, j), v in overlaps.items()}


def to_jsonable_decision(d: Decision) -> Dict[str, Any]:
    # Don’t return pricing unless AUTO
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


def _quality_to_jsonable(q) -> Dict[str, Any]:
    return {
        "ok": bool(q.ok),
        "score": float(q.score),
        "flags": list(q.flags),
        "metrics": dict(q.metrics),
        "user_message": str(q.user_message),
    }


def _manual_payload(image_id: str, quality_json: Dict[str, Any], reason: str, flags: Optional[List[str]] = None) -> Dict[str, Any]:
    flags = flags or []
    decision = Decision(
        severity=None,
        route="MANUAL_REVIEW",
        confidence_score=0.0,
        estimated_cost_lkr=None,
        cost_range_lkr=None,
        pricing_mode="PENDING_REVIEW",
        reasons=[
            reason,
            "Please upload a clear full-car photo (front/side), good lighting, not blurry.",
            f"If it keeps happening, contact support: {SUPPORT_CONTACT}.",
        ],
        flags=flags,
    )
    return {
        "quality": quality_json,
        "image_id": image_id,
        "vehicle_area_ratio": None,
        "damages": [],
        "overlaps": None,
        "decision": to_jsonable_decision(decision),
        "explanation": reason,
        "meta": {
            "weights_path": WEIGHTS_PATH,
            "vehicle_weights": VEHICLE_WEIGHTS,
            "thresholds_version": THRESHOLDS_VERSION,
            "max_infer_side": MAX_INFER_SIDE,
        },
    }


def _unwrap_yolo_result(x):
    """Ensure detector output is a single Results object."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) > 0:
        if hasattr(x[0], "masks") or hasattr(x[0], "boxes"):
            return x[0]
    return x


def _unwrap_vehicle_mask(x):
    """If vehicle masker returns (mask, something), take mask."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return x[0]
    return x


def aggregate_decisions(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    # If any manual -> total must be manual (professional rule)
    for d in decisions:
        if d.get("route") == "MANUAL_REVIEW" or d.get("pricing_mode") == "PENDING_REVIEW":
            return {
                "pricing_mode": "PENDING_REVIEW",
                "estimated_total_lkr": None,
                "total_range_lkr": None,
                "reason": "At least one image needs manual review, so total cannot be auto-calculated.",
            }

    total_point = 0
    total_lo = 0
    total_hi = 0
    any_range = False

    for d in decisions:
        pm = d.get("pricing_mode")

        if pm == "AUTO_POINT" and d.get("estimated_cost_lkr") is not None:
            val = int(d["estimated_cost_lkr"])
            total_point += val
            total_lo += val
            total_hi += val

        elif pm == "AUTO_RANGE" and d.get("cost_range_lkr"):
            lo, hi = d["cost_range_lkr"]
            total_lo += int(lo)
            total_hi += int(hi)
            any_range = True

        elif pm == "NONE":
            continue

    if any_range:
        return {
            "pricing_mode": "AUTO_RANGE",
            "estimated_total_lkr": None,
            "total_range_lkr": (int(total_lo), int(total_hi)),
            "reason": "Total is a range because at least one damage requires replacement.",
        }

    return {
        "pricing_mode": "AUTO_POINT",
        "estimated_total_lkr": int(total_point),
        "total_range_lkr": None,
        "reason": "Total is a point estimate because all damages are repair-based point estimates.",
    }


def run_pipeline(img: Image.Image, image_id: str, quality_json: Dict[str, Any]) -> Dict[str, Any]:
    detector, vehicle_masker = get_models()

    # 1) Vehicle mask (may be None)
    vehicle_mask = _unwrap_vehicle_mask(vehicle_masker.predict_vehicle_mask(img))

    # 2) Damage prediction
    yolo_raw = detector.predict(img)
    yolo_res = _unwrap_yolo_result(yolo_raw)

    if yolo_res is None:
        return _manual_payload(
            image_id=image_id,
            quality_json=quality_json,
            reason="Model returned no result for this image (prediction output was None).",
            flags=["PIPELINE_NO_YOLO_RESULT"],
        )

    # 3) Evidence extraction
    evidence = extract_evidence(yolo_res, image_id=image_id, vehicle_mask=vehicle_mask)

    # 4) Decision + explanation
    decision = decide_case(evidence)
    explanation = generate_explanation(evidence, decision)

    return {
        "quality": quality_json,
        **to_jsonable_evidence(evidence),
        "decision": to_jsonable_decision(decision),
        "explanation": explanation,
        "meta": {
            "weights_path": WEIGHTS_PATH,
            "vehicle_weights": VEHICLE_WEIGHTS,
            "thresholds_version": THRESHOLDS_VERSION,
            "max_infer_side": MAX_INFER_SIDE,
        },
    }


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="AutoInspect AI API", version="1.0.0")


@app.on_event("startup")
def _startup():

    init_db()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess", response_model=AssessResponse)
async def assess_image(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG image.")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # Resize for stability
    img = _resize_for_inference(img, max_side=MAX_INFER_SIDE)

    image_id = file.filename or "upload"
    case_id = str(uuid.uuid4())

    # Quality check
    q = assess_image_quality(img)
    q_json = _quality_to_jsonable(q)

    # If low quality -> return 200 MANUAL_REVIEW payload (no client crash)
    if not q.ok:
        payload = _manual_payload(
            image_id=image_id,
            quality_json=q_json,
            reason="Image quality is insufficient for reliable automatic assessment.",
            flags=["LOW_QUALITY_IMAGE"] + list(q_json.get("flags", [])),
        )

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
        return JSONResponse({**payload, "case_id": case_id})

    # Run pipeline (guard crashes so UI gets a real reason)
    try:
        payload = run_pipeline(img, image_id=image_id, quality_json=q_json)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PIPELINE_CRASH",
                "message": str(e),
                "traceback": tb,
                "hint": "Common: model output mismatch / None returns / OOM.",
            },
        )

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

    return JSONResponse({**payload, "case_id": case_id})


@app.post("/assess_batch", response_model=AssessBatchResponse)
async def assess_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least 1 image.")

    batch_id = str(uuid.uuid4())
    items: List[Dict[str, Any]] = []
    decision_dicts: List[Dict[str, Any]] = []

    for f in files:
        if f.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
            # hard fail: client sent bad file
            raise HTTPException(status_code=400, detail="All uploads must be JPG/PNG images.")

        try:
            raw = await f.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail=f"Could not read image file: {f.filename}")

        img = _resize_for_inference(img, max_side=MAX_INFER_SIDE)

        image_id = f.filename or "upload"
        case_id = str(uuid.uuid4())

        # Quality check per image
        q = assess_image_quality(img)
        q_json = _quality_to_jsonable(q)

        if not q.ok:
            payload = _manual_payload(
                image_id=image_id,
                quality_json=q_json,
                reason="Image quality is insufficient for reliable automatic assessment.",
                flags=["LOW_QUALITY_IMAGE"] + list(q_json.get("flags", [])),
            )
        else:
            try:
                payload = run_pipeline(img, image_id=image_id, quality_json=q_json)
            except Exception as e:
                # convert crash into a manual payload (don’t break the whole batch)
                payload = _manual_payload(
                    image_id=image_id,
                    quality_json=q_json,
                    reason=f"Pipeline crash on this image: {str(e)}",
                    flags=["PIPELINE_CRASH_ON_ITEM"],
                )

        # Save each case (keeps history + audit trail)
        image_path, image_hash = save_uploaded_image(case_id, img, raw)
        insert_case(
            case_id=case_id,
            image_path=image_path,
            image_sha256=image_hash,
            weights_path=WEIGHTS_PATH,
            vehicle_weights=VEHICLE_WEIGHTS,
            thresholds_version=THRESHOLDS_VERSION,
            response_json={**payload, "case_id": case_id, "batch_id": batch_id},
        )

        item = {**payload, "case_id": case_id}
        items.append(item)

        # Collect decisions for aggregation
        decision_dicts.append(item["decision"])

    final_total = aggregate_decisions(decision_dicts)

    return JSONResponse(
        {
            "batch_id": batch_id,
            "items": items,
            "final_total": final_total,
        }
    )


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

    content = row.response_json
    filename = f"autoinspect_report_{case_id}.json"
    return Response(
        content=JSONResponse(content).body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )