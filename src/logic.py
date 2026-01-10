from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class DamageInstance:
    """
    One detected instance.

    area_ratio: damage_pixels / vehicle_pixels   (IMPORTANT)
    confidence: YOLO conf in [0,1]
    """
    damage_type: str
    confidence: float
    area_ratio: float


@dataclass(frozen=True)
class CaseEvidence:
    """
    Evidence from a single image.

    overlaps: optional dict of (i,j)->overlap_ratio among masks (0..1)
    vehicle_area_ratio: vehicle_pixels / image_pixels  (framing signal)
    """
    image_id: str
    damages: List[DamageInstance]
    overlaps: Optional[Dict[Tuple[int, int], float]] = None
    vehicle_area_ratio: Optional[float] = None


@dataclass(frozen=True)
class Decision:
    """
    Decision support output.

    Pricing rules (PHASE-1):
      - If route != "AUTO", pricing MUST be None (no fake certainty).
      - If severity == "NO_DAMAGE", pricing is 0.
    pricing_mode:
      - "NONE" (no damage)
      - "AUTO_POINT" (auto-approved with point estimate)
      - "AUTO_RANGE" (auto-approved but we prefer range messaging)
      - "PENDING_REVIEW" (manual review / needs human)
    """
    severity: Optional[str]  # can be None when we cannot safely assert a severity
    route: str
    confidence_score: float
    estimated_cost_lkr: Optional[int]
    cost_range_lkr: Optional[Tuple[int, int]]
    pricing_mode: str
    reasons: List[str]
    flags: List[str]


# ============================================================
# Configuration: thresholds (tuneable, deterministic)
# ============================================================

# --- Noise suppression ---
MIN_CONF_KEEP_DEFAULT = 0.35
MIN_AREA_KEEP = 0.0008  # 0.08% of vehicle area
NO_DAMAGE_AREA_THRESHOLD = 0.0012  # 0.12% of vehicle area

PER_CLASS_MIN_CONF_KEEP = {
    "lamp broken": 0.55,
    "scratch": 0.40,
    "dent": 0.40,
}

PER_CLASS_MIN_AREA_KEEP = {
    "scratch": 0.0010,
    "dent": 0.0010,
}

# --- Severity score thresholds ---
# score = sum(area_ratio * type_weight)
SEVERITY_THRESHOLDS = {
    "LOW": 0.010,
    "MEDIUM": 0.030,  # > MEDIUM -> HIGH
}

# --- Routing thresholds ---
MIN_CONF_FOR_AUTO = 0.55
MIN_TOTAL_AREA_FOR_TRUST = 0.0020
MAX_OVERLAP_FOR_TRUST = 0.60

# Vehicle framing thresholds (vehicle / image)
MIN_VEHICLE_AREA_RATIO = 0.08
MAX_VEHICLE_AREA_RATIO = 0.90


# ============================================================
# Damage taxonomy + weights
# ============================================================

DAMAGE_TYPE_WEIGHT = {
    "scratch": 1.0,
    "dent": 1.5,
    "crack": 2.0,
    "glass": 2.5,
    "glass_shatter": 3.0,
    "lamp broken": 2.2,
    "tire flat": 3.0,
}


def type_weight(damage_type: str) -> float:
    return DAMAGE_TYPE_WEIGHT.get(str(damage_type).lower(), 1.2)


def is_glass(damage_type: str) -> bool:
    t = str(damage_type).lower()
    return ("glass" in t) or ("windshield" in t) or ("window" in t)


def is_tire(damage_type: str) -> bool:
    t = str(damage_type).lower()
    return ("tire" in t) or ("tyre" in t)


def is_lamp(damage_type: str) -> bool:
    t = str(damage_type).lower()
    return ("lamp" in t) or ("headlight" in t) or ("tail light" in t)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _normalize(d: DamageInstance) -> DamageInstance:
    return DamageInstance(
        damage_type=str(d.damage_type),
        confidence=_clamp(d.confidence, 0.0, 1.0),
        area_ratio=_clamp(d.area_ratio, 0.0, 1.0),
    )


# ============================================================
# Core scoring helpers
# ============================================================

def filter_meaningful_damages(damages: List[DamageInstance]) -> Tuple[List[DamageInstance], List[str]]:
    """
    Drop low-confidence / tiny-area detections deterministically.
    Critical types (glass/tire/lamp) are allowed with slightly lower area threshold,
    but still must pass confidence gate.
    """
    flags: List[str] = []
    kept: List[DamageInstance] = []

    for d0 in damages:
        d = _normalize(d0)
        name = d.damage_type.lower()

        min_conf = PER_CLASS_MIN_CONF_KEEP.get(name, MIN_CONF_KEEP_DEFAULT)
        min_area = PER_CLASS_MIN_AREA_KEEP.get(name, MIN_AREA_KEEP)

        # Critical components: allow smaller area but still require confidence
        if is_glass(d.damage_type) or is_tire(d.damage_type) or is_lamp(d.damage_type):
            min_area = min(min_area, 0.0005)

        if d.confidence < min_conf:
            continue
        if d.area_ratio < min_area:
            continue

        kept.append(d)

    if len(kept) < len(damages):
        flags.append("NOISE_FILTER_APPLIED")

    return kept, flags


def aggregate_confidence(damages: List[DamageInstance]) -> float:
    """Area-weighted confidence (fallback to mean confidence if total area is 0)."""
    if not damages:
        return 0.0
    total_area = sum(d.area_ratio for d in damages)
    if total_area <= 0:
        return sum(d.confidence for d in damages) / len(damages)
    return sum(d.confidence * d.area_ratio for d in damages) / total_area


def weighted_damage_score(damages: List[DamageInstance]) -> float:
    """score = Σ(area_ratio * type_weight)."""
    return sum(d.area_ratio * type_weight(d.damage_type) for d in damages)


def summarize_by_type(damages: List[DamageInstance]) -> Dict[str, Dict[str, float]]:
    """Returns per-type totals: {type: {"area": total_area, "max_conf": max_conf}}"""
    out: Dict[str, Dict[str, float]] = {}
    for d in damages:
        k = d.damage_type.lower()
        if k not in out:
            out[k] = {"area": 0.0, "max_conf": 0.0}
        out[k]["area"] += d.area_ratio
        out[k]["max_conf"] = max(out[k]["max_conf"], d.confidence)
    return out


def vehicle_visibility_ok(vehicle_area_ratio: Optional[float]) -> bool:
    """
    Determines whether we can safely claim NO_DAMAGE.
    If we don't have vehicle_area_ratio, visibility is unknown => NOT OK.
    """
    if vehicle_area_ratio is None:
        return False
    var = float(vehicle_area_ratio)
    return (MIN_VEHICLE_AREA_RATIO <= var <= MAX_VEHICLE_AREA_RATIO)


# ============================================================
# Confidence Decomposition (NEW)
# ============================================================

def coverage_confidence(vehicle_area_ratio: Optional[float]) -> float:
    """
    Coverage confidence based on how well the vehicle fills the frame.
    - None => unknown => 0.0
    - Outside [MIN, MAX] => 0.0 (also handled by routing)
    - Inside => score peaks near TARGET and falls as you get too close/far.
    """
    if vehicle_area_ratio is None:
        return 0.0

    var = float(vehicle_area_ratio)
    if var < MIN_VEHICLE_AREA_RATIO or var > MAX_VEHICLE_AREA_RATIO:
        return 0.0

    # Target framing (tunable). 0.35 means car reasonably visible, not too close.
    TARGET = 0.35
    dist = abs(var - TARGET)
    cov = 1.0 - (dist / TARGET)
    return _clamp(cov, 0.0, 1.0)


def consistency_confidence(overlaps: Optional[Dict[Tuple[int, int], float]]) -> float:
    """
    Consistency confidence from mask overlap (IoU).
    - No overlaps => 1.0 (no contradiction signal)
    - High overlap => lower confidence
    """
    if not overlaps:
        return 1.0
    max_iou = max(float(v) for v in overlaps.values())
    return _clamp(1.0 - max_iou, 0.0, 1.0)


def confidence_breakdown(
    meaningful: List[DamageInstance],
    vehicle_area_ratio: Optional[float],
    overlaps: Optional[Dict[Tuple[int, int], float]],
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (aggregate_conf, breakdown_dict)
    aggregate = detection * coverage * consistency

    Conservative by design:
      - If coverage is unknown => aggregate drops => routes to manual.
    """
    det = aggregate_confidence(meaningful)
    cov = coverage_confidence(vehicle_area_ratio)
    con = consistency_confidence(overlaps)
    agg = _clamp(det * cov * con, 0.0, 1.0)

    return agg, {
        "detection": float(det),
        "coverage": float(cov),
        "consistency": float(con),
        "aggregate": float(agg),
    }


# ============================================================
# Severity
# ============================================================

def assign_severity(damages: List[DamageInstance]) -> Tuple[str, float, List[str]]:
    reasons: List[str] = []
    score = weighted_damage_score(damages)
    reasons.append(f"Weighted damage score = {score:.4f} (sum(area_ratio * type_weight)).")

    if score <= SEVERITY_THRESHOLDS["LOW"]:
        reasons.append(f"Score <= {SEVERITY_THRESHOLDS['LOW']:.3f} => LOW severity.")
        return "LOW", score, reasons

    if score <= SEVERITY_THRESHOLDS["MEDIUM"]:
        reasons.append(f"Score <= {SEVERITY_THRESHOLDS['MEDIUM']:.3f} => MEDIUM severity.")
        return "MEDIUM", score, reasons

    reasons.append(f"Score > {SEVERITY_THRESHOLDS['MEDIUM']:.3f} => HIGH severity.")
    return "HIGH", score, reasons


# ============================================================
# Pricing: component-aware + cosmetic range
# ============================================================

GLASS_REPAIR_LKR = 25000
GLASS_REPLACE_LKR = 90000
LAMP_REPLACE_LKR = 60000
TIRE_REPLACE_LKR = 30000

GLASS_REPLACE_RATIO_THRESHOLD = 0.010  # 1.0% of vehicle area

COSMETIC_RANGE_LKR = {
    "LOW": (0, 25000),
    "MEDIUM": (25000, 120000),
    "HIGH": (120000, 400000),
}


def cosmetic_cost_range(severity: str, score: float) -> Tuple[int, int, int, List[str]]:
    """
    Returns (low, high, point, reasons) based on severity + score.
    Point is a deterministic interpolation within range.
    """
    reasons: List[str] = []
    lo, hi = COSMETIC_RANGE_LKR[severity]

    if severity == "LOW":
        denom = max(SEVERITY_THRESHOLDS["LOW"], 1e-6)
        t = _clamp(score / denom, 0.0, 1.0)
    elif severity == "MEDIUM":
        denom = max(SEVERITY_THRESHOLDS["MEDIUM"] - SEVERITY_THRESHOLDS["LOW"], 1e-6)
        t = _clamp((score - SEVERITY_THRESHOLDS["LOW"]) / denom, 0.0, 1.0)
    else:
        t = _clamp(score / 0.10, 0.0, 1.0)

    point = int(round(lo + t * (hi - lo)))
    reasons.append(f"Cosmetic cost band for {severity}: {lo}-{hi} LKR (point={point}).")
    return lo, hi, point, reasons


def component_pricing(damages: List[DamageInstance]) -> Tuple[int, List[str], List[str]]:
    """
    Returns (component_cost, reasons, flags).
    Component cost is additive (glass + tire + lamp).
    """
    reasons: List[str] = []
    flags: List[str] = []

    by_type = summarize_by_type(damages)
    comp_cost = 0

    # Glass
    glass_area = 0.0
    for k, v in by_type.items():
        if is_glass(k):
            glass_area += v["area"]
    if glass_area > 0:
        if glass_area >= GLASS_REPLACE_RATIO_THRESHOLD:
            comp_cost += GLASS_REPLACE_LKR
            reasons.append(f"Glass ratio={glass_area:.4f} => replacement => +{GLASS_REPLACE_LKR} LKR.")
            flags.append("GLASS_REPLACE")
        else:
            comp_cost += GLASS_REPAIR_LKR
            reasons.append(f"Glass ratio={glass_area:.4f} => repair => +{GLASS_REPAIR_LKR} LKR.")
            flags.append("GLASS_REPAIR")

    # Tire
    if any(is_tire(d.damage_type) for d in damages):
        comp_cost += TIRE_REPLACE_LKR
        reasons.append(f"Tire issue detected => +{TIRE_REPLACE_LKR} LKR.")
        flags.append("TIRE_REPLACE")

    # Lamp
    if any(is_lamp(d.damage_type) for d in damages):
        comp_cost += LAMP_REPLACE_LKR
        reasons.append(f"Lamp issue detected => +{LAMP_REPLACE_LKR} LKR.")
        flags.append("LAMP_REPLACE")

    return comp_cost, reasons, flags


# ============================================================
# Routing
# ============================================================

def should_route_manual(
    damages: List[DamageInstance],
    conf_score: float,
    overlaps: Optional[Dict[Tuple[int, int], float]] = None,
    vehicle_area_ratio: Optional[float] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Deterministic routing:
      - low confidence => manual
      - tiny area (but not "no damage") => manual
      - high overlap => manual
      - poor framing => manual
    """
    reasons: List[str] = []
    flags: List[str] = []

    if vehicle_area_ratio is not None:
        var = float(vehicle_area_ratio)
        if var < MIN_VEHICLE_AREA_RATIO:
            reasons.append(f"Vehicle coverage too small (vehicle_area_ratio={var:.2f}) => MANUAL_REVIEW.")
            flags.append("FRAMING_TOO_FAR_OR_NOT_VEHICLE")
            return True, reasons, flags
        if var > MAX_VEHICLE_AREA_RATIO:
            reasons.append(f"Vehicle coverage too large (vehicle_area_ratio={var:.2f}) => MANUAL_REVIEW.")
            flags.append("FRAMING_TOO_CLOSE")
            return True, reasons, flags

    if float(conf_score) < MIN_CONF_FOR_AUTO:
        reasons.append(f"Aggregate confidence {conf_score:.2f} < {MIN_CONF_FOR_AUTO:.2f} => MANUAL_REVIEW.")
        flags.append("LOW_CONFIDENCE")
        return True, reasons, flags

    total_area = sum(d.area_ratio for d in damages)
    if total_area < MIN_TOTAL_AREA_FOR_TRUST:
        reasons.append(f"Total damage area {total_area:.4f} < {MIN_TOTAL_AREA_FOR_TRUST:.4f} => MANUAL_REVIEW.")
        flags.append("LOW_TOTAL_AREA")
        return True, reasons, flags

    if overlaps:
        max_ov = max(overlaps.values()) if overlaps else 0.0
        if max_ov > MAX_OVERLAP_FOR_TRUST:
            reasons.append(f"Max overlap {max_ov:.2f} > {MAX_OVERLAP_FOR_TRUST:.2f} => MANUAL_REVIEW.")
            flags.append("HIGH_OVERLAP")
            return True, reasons, flags

    reasons.append("Meets confidence/area/overlap/framing thresholds => AUTO.")
    return False, reasons, flags


# ============================================================
# Main Decision Function
# ============================================================

def decide_case(evidence: CaseEvidence) -> Decision:
    reasons: List[str] = []
    flags: List[str] = []

    # 0) Normalize + filter noise
    meaningful, f = filter_meaningful_damages(evidence.damages)
    flags.extend(f)

    # 1) Confidence decomposition (meaningful only)
    conf_score, conf_parts = confidence_breakdown(
        meaningful=meaningful,
        vehicle_area_ratio=evidence.vehicle_area_ratio,
        overlaps=evidence.overlaps,
    )
    reasons.append(
        "Confidence breakdown: "
        f"detection={conf_parts['detection']:.2f}, "
        f"coverage={conf_parts['coverage']:.2f}, "
        f"consistency={conf_parts['consistency']:.2f} "
        f"=> aggregate={conf_parts['aggregate']:.2f}."
    )

    # Vehicle validity / framing guard:
    # If vehicle is not clearly detected, do not make claims (including NO_DAMAGE).
    if evidence.vehicle_area_ratio is not None and evidence.vehicle_area_ratio < MIN_VEHICLE_AREA_RATIO:
        return Decision(
            severity=None,
            route="MANUAL_REVIEW",
            confidence_score=float(conf_score),
            estimated_cost_lkr=None,
            cost_range_lkr=None,
            pricing_mode="PENDING_REVIEW",
            reasons=[
                "Vehicle not clearly detected in the photo.",
                f"vehicle_area_ratio={evidence.vehicle_area_ratio:.2f} is too low.",
                "Please upload a clear full-car photo.",
            ],
            flags=flags + ["NOT_A_VALID_VEHICLE_VIEW"],
        )

    # -------- NO DAMAGE SHORT-CIRCUIT (SAFE) --------
    total_area = sum(d.area_ratio for d in meaningful)
    has_critical = any(is_glass(d.damage_type) or is_tire(d.damage_type) or is_lamp(d.damage_type) for d in meaningful)

    vehicle_ok = vehicle_visibility_ok(evidence.vehicle_area_ratio)

    # If vehicle visibility is OK and damage is below noise threshold, we can assert NO_DAMAGE.
    if vehicle_ok and total_area < NO_DAMAGE_AREA_THRESHOLD and not has_critical:
        return Decision(
            severity="NO_DAMAGE",
            route="AUTO",
            confidence_score=float(conf_score),
            estimated_cost_lkr=0,
            cost_range_lkr=(0, 0),
            pricing_mode="NONE",
            reasons=[
                "No meaningful damage detected.",
                f"Total damage area ({total_area:.4f}) is below the noise threshold.",
            ],
            flags=flags + ["NO_DAMAGE"],
        )

    # If we detected nothing but visibility is unknown/insufficient, do NOT claim NO_DAMAGE.
    if (not vehicle_ok) and len(meaningful) == 0:
        return Decision(
            severity=None,
            route="MANUAL_REVIEW",
            confidence_score=float(conf_score),
            estimated_cost_lkr=None,
            cost_range_lkr=None,
            pricing_mode="PENDING_REVIEW",
            reasons=[
                "No damages detected, but vehicle visibility/framing is insufficient to confirm NO_DAMAGE.",
                f"vehicle_area_ratio={evidence.vehicle_area_ratio}",
                "Please upload a clear full-car photo.",
            ],
            flags=flags + ["VISIBILITY_INSUFFICIENT_FOR_NO_DAMAGE"],
        )

    # 2) Severity
    severity, score, sev_reasons = assign_severity(meaningful)
    reasons.extend(sev_reasons)

    # 3) Pricing candidates (only used if AUTO)
    cos_lo, cos_hi, cos_point, cos_reasons = cosmetic_cost_range(severity, score)
    reasons.extend(cos_reasons)

    comp_cost, comp_reasons, comp_flags = component_pricing(meaningful)
    reasons.extend(comp_reasons)
    flags.extend(comp_flags)

    total_lo = cos_lo + comp_cost
    total_hi = cos_hi + comp_cost
    total_point = cos_point + comp_cost

    # 4) Routing
    manual, route_reasons, route_flags = should_route_manual(
        damages=meaningful,
        conf_score=conf_score,
        overlaps=evidence.overlaps,
        vehicle_area_ratio=evidence.vehicle_area_ratio,
    )
    reasons.extend(route_reasons)
    flags.extend(route_flags)

    if manual:
        # PHASE-1 RULE: If manual, do NOT return ANY pricing numbers.
        return Decision(
            severity=severity,
            route="MANUAL_REVIEW",
            confidence_score=float(conf_score),
            estimated_cost_lkr=None,
            cost_range_lkr=None,
            pricing_mode="PENDING_REVIEW",
            reasons=reasons,
            flags=flags,
        )

    # AUTO: pricing is allowed (point + range for transparency)
    return Decision(
        severity=severity,
        route="AUTO",
        confidence_score=float(conf_score),
        estimated_cost_lkr=int(total_point),
        cost_range_lkr=(int(total_lo), int(total_hi)),
        pricing_mode="AUTO_POINT",
        reasons=reasons,
        flags=flags,
    )
