from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class DamageInstance:
    """
    One detected Damage.
    Confidence: YOLO conf in [0,1]
    area_ratio:
      - preferred: damage_pixels / vehicle_pixels
      - fallback (if vehicle mask missing): damage_pixels / image_pixels
    """
    damage_type: str
    confidence: float
    area_ratio: float


@dataclass(frozen=True)
class CaseEvidence:
    """
    Evidence of a single image.
        - image id
        - list of Damages detected
        - overlaps between damages
        - vehicle area ratio
    """
    image_id: str
    damages: List[DamageInstance]
    overlaps: Optional[Dict[Tuple[int, int], float]] = None
    vehicle_area_ratio: Optional[float] = None


@dataclass(frozen=True)
class Decision:
    """
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
# Thresholds
# ============================================================

# min total damage area ratio to consider "no damage"
NO_DAMAGE_AREA_THRESHOLD = 0.0012

# defaults (were missing in your file)
MIN_CONF_KEEP_DEFAULT = 0.50
MIN_AREA_KEEP_DEFAULT = 0.0005

# min confidence to keep a damage
PER_CLASS_MIN_CONF_KEEP = {
    "scratch": 0.40,
    "dent": 0.40,
    "crack": 0.45,
    "glass shatter": 0.50,
    "lamp broken": 0.55,
    "tire flat": 0.50,
}

# min area ratio to keep a damage
PER_CLASS_MIN_AREA_KEEP = {
    "scratch": 0.0005,
    "dent": 0.0005,
    "crack": 0.0005,
    "glass shatter": 0.0020,
    "lamp broken": 0.00015,
    "tire flat": 0.0020,
}

# Severity score thresholds
SEVERITY_THRESHOLDS = {
    "LOW": 0.010,
    "MEDIUM": 0.030,  # > MEDIUM -> HIGH
}

# Routing thresholds
MIN_CONF_FOR_AUTO = 0.55
MIN_TOTAL_AREA_FOR_TRUST = 0.0020
MAX_OVERLAP_FOR_TRUST = 0.60

# Vehicle framing thresholds (vehicle / image)
MIN_VEHICLE_AREA_RATIO = 0.08
MAX_VEHICLE_AREA_RATIO = 1.00

# damage type weights for scoring
DAMAGE_TYPE_WEIGHT = {
    "scratch": 1.0,
    "dent": 1.5,
    "crack": 2.0,
    "glass shatter": 3.0,
    "lamp broken": 2.2,
    "tire flat": 3.0,
}


# ============================================================
# Damage Identification & Normalization Helpers
# ============================================================

def type_weight(damage_type: str) -> float:
    return DAMAGE_TYPE_WEIGHT.get(str(damage_type).lower(), 1.2)

def is_scratch(damage_type: str) -> bool:
    t = str(damage_type).lower()
    return ("scratch" in t)

def is_dent(damage_type: str) -> bool:
    t = str(damage_type).lower()
    return ("dent" in t)

def is_crack(damage_type: str) -> bool:
    t = str(damage_type).lower()
    return ("crack" in t)

def is_glass(damage_type: str) -> bool:
    t = str(damage_type).lower()
    return ("glass shatter" in t)

def is_lamp(damage_type: str) -> bool:
    t = str(damage_type).lower()
    return ("lamp broken" in t)

def is_tire(damage_type: str) -> bool:
    t = str(damage_type).lower()
    return ("tire flat" in t)

def _clamp(x: float, lo: float, hi: float) -> float: # keeps x within [lo, hi] [0 and 1]
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
    flags = []
    kept = []

    for d0 in damages:
        d = _normalize(d0)
        name = d.damage_type.lower()

        if is_dent(name) or is_scratch(name) or is_crack(name) or is_glass(name) or is_lamp(name) or is_tire(name):
            min_conf = PER_CLASS_MIN_CONF_KEEP.get(name, MIN_CONF_KEEP_DEFAULT)
            min_area = PER_CLASS_MIN_AREA_KEEP.get(name, MIN_AREA_KEEP_DEFAULT)
        else:
            min_conf = MIN_CONF_KEEP_DEFAULT
            min_area = MIN_AREA_KEEP_DEFAULT

        if d.confidence < min_conf:
            continue
        if d.area_ratio < min_area:
            continue

        kept.append(d)

    if len(kept) < len(damages):
        flags.append("NOISE_FILTER_APPLIED")

    return kept, flags


def aggregate_confidence(damages: List[DamageInstance]) -> float:
    if not damages:
        return 0.0

    total_area = sum(d.area_ratio for d in damages)

    if total_area <= 0:
        return sum(d.confidence for d in damages) / len(damages)
    else:
        return sum(d.confidence * d.area_ratio for d in damages) / total_area


def weighted_damage_score(damages: List[DamageInstance]) -> float:
    return sum(d.area_ratio * type_weight(d.damage_type) for d in damages)


def vehicle_visibility_ok(vehicle_area_ratio: Optional[float]) -> bool:
    if vehicle_area_ratio is None:
        return False
    var = float(vehicle_area_ratio)
    return (MIN_VEHICLE_AREA_RATIO <= var <= MAX_VEHICLE_AREA_RATIO)


# ============================================================
# Severity
# ============================================================

def assign_severity(damages: List[DamageInstance]) -> Tuple[str, float, List[str]]:
    reasons: List[str] = []
    score = weighted_damage_score(damages)
    reasons.append(f"Weighted damage score = {score:.4f}")

    if score <= SEVERITY_THRESHOLDS["LOW"]:
        reasons.append(f"Score <= {SEVERITY_THRESHOLDS['LOW']:.3f} => LOW severity.")
        return "LOW", score, reasons

    if score <= SEVERITY_THRESHOLDS["MEDIUM"]:
        reasons.append(f"Score <= {SEVERITY_THRESHOLDS['MEDIUM']:.3f} => MEDIUM severity.")
        return "MEDIUM", score, reasons

    reasons.append(f"Score > {SEVERITY_THRESHOLDS['MEDIUM']:.3f} => HIGH severity.")
    return "HIGH", score, reasons


# ============================================================
# Pricing
# 1) glass/tire/lamp => replace 
# 2) scratch/dent/crack => area-based repair
# ============================================================

MIN_GLASS_REPLACE_LKR = 15000
MAX_GLASS_REPLACE_LKR = 150000

MIN_LAMP_REPLACE_LKR = 10000
MAX_LAMP_REPLACE_LKR = 120000

MIN_TIRE_REPLACE_LKR = 25000
MAX_TIRE_REPLACE_LKR = 60000

# area-based repair rates
REPAIR_COST_RATE_LKR = {
    "scratch": 100000,
    "dent": 150000,
    "crack": 150000,
}

def replacement_only_range(damages: List[DamageInstance]) -> Optional[Tuple[int, int]]:
    if any(is_glass(d.damage_type) for d in damages):
        return (MIN_GLASS_REPLACE_LKR, MAX_GLASS_REPLACE_LKR)
    if any(is_tire(d.damage_type) for d in damages):
        return (MIN_TIRE_REPLACE_LKR, MAX_TIRE_REPLACE_LKR)
    if any(is_lamp(d.damage_type) for d in damages):
        return (MIN_LAMP_REPLACE_LKR, MAX_LAMP_REPLACE_LKR)
    return None

def repair_cost_area_based(d: DamageInstance) -> int:
    t = str(d.damage_type).lower()
    rate = REPAIR_COST_RATE_LKR.get(t, 180000)  # fallback
    return max(int(round(float(d.area_ratio) * float(rate))), 5000)


# ============================================================
# Routing
# ============================================================

def should_route_manual(damages: List[DamageInstance],conf_score: float,overlaps: Optional[Dict[Tuple[int, int], float]] = None,vehicle_area_ratio: Optional[float] = None):
    reasons = []
    flags = []
    if vehicle_area_ratio is None:
        reasons.append("Vehicle is Not Detected => MANUAL_REVIEW Needed.")
        flags.append("VEHICLE_MASK_MISSING")
        return True, reasons, flags

    var = float(vehicle_area_ratio)
    if var < MIN_VEHICLE_AREA_RATIO:
        reasons.append(f"Vehicle coverage is too small => MANUAL_REVIEW Needed.")
        flags.append("FRAMING_TOO_FAR_OR_NOT_VEHICLE")
        return True, reasons, flags
    if var > MAX_VEHICLE_AREA_RATIO:
        reasons.append(f"Vehicle coverage is too large => MANUAL_REVIEW Needed.")
        flags.append("FRAMING_TOO_CLOSE")
        return True, reasons, flags

    if float(conf_score) < MIN_CONF_FOR_AUTO:
        reasons.append(f"Our Model is Uncertain About the Damage => MANUAL_REVIEW Needed.")
        flags.append("LOW_CONFIDENCE")
        return True, reasons, flags

    total_area = sum(d.area_ratio for d in damages)
    if total_area < MIN_TOTAL_AREA_FOR_TRUST:
        reasons.append(f"Our Model is Uncertain About the Damage => MANUAL_REVIEW Needed.")
        flags.append("LOW_TOTAL_AREA")
        return True, reasons, flags

    if overlaps:
        max_ov = max(overlaps.values()) if overlaps else 0.0
        if max_ov > MAX_OVERLAP_FOR_TRUST:
            reasons.append(f"Too Much Overlaps in the Damage => MANUAL_REVIEW Needed.")
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

    meaningful, f = filter_meaningful_damages(evidence.damages)
    flags.extend(f)

    conf_score = aggregate_confidence(meaningful)

    total_area = sum(d.area_ratio for d in meaningful)
    has_critical = any(is_glass(d.damage_type) or is_tire(d.damage_type) or is_lamp(d.damage_type) for d in meaningful)
    vehicle_vis_ok = vehicle_visibility_ok(evidence.vehicle_area_ratio)

    # ============================================================
    ## No Damage
    # ===========================================================
    if vehicle_vis_ok and total_area < NO_DAMAGE_AREA_THRESHOLD and not has_critical:
        return Decision(
            severity="NO_DAMAGE",
            route="AUTO",
            confidence_score=conf_score,
            estimated_cost_lkr=0,
            cost_range_lkr=(0, 0),
            pricing_mode="NONE",
            reasons=[
                "No meaningful damage detected.",
                f"Total damage area ({total_area:.4f}) is below the noise threshold.",
            ],
            flags=flags + ["NO_DAMAGE"],
        )

    if (not vehicle_vis_ok) and len(meaningful) == 0:
        return Decision(
            severity=None,
            route="MANUAL_REVIEW",
            confidence_score=conf_score,
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

    # ============================================================
    # Damage Present: Severity + Pricing + Routing
    # ============================================================

    severity, score, sev_reasons = assign_severity(meaningful)
    reasons.extend(sev_reasons)

    # Replacement-only: if the damage is glass/lamp/tire 
    rep = replacement_only_range(meaningful)
    if rep is not None:
        lo, hi = rep
        reasons.append("Parts Needs to Be replaced, Estimated Range shown.")

        manual, route_reasons, route_flags = should_route_manual(
            damages=meaningful,
            conf_score=conf_score,
            overlaps=evidence.overlaps,
            vehicle_area_ratio=evidence.vehicle_area_ratio,
        )
        reasons.extend(route_reasons)
        flags.extend(route_flags)

        if manual:
            return Decision(
                severity=severity,
                route="MANUAL_REVIEW",
                confidence_score=conf_score,
                estimated_cost_lkr=None,
                cost_range_lkr=None,
                pricing_mode="PENDING_REVIEW",
                reasons=reasons,
                flags=flags,
            )

        return Decision(
            severity=severity,
            route="AUTO",
            confidence_score=conf_score,
            estimated_cost_lkr=None,
            cost_range_lkr=(int(lo), int(hi)),
            pricing_mode="AUTO_RANGE",
            reasons=reasons,
            flags=flags,
        )

    # Repairable: if the damage is scratch/dent/crack
    if meaningful:
        d = meaningful[0]
        est = repair_cost_area_based(d)
        reasons.append(f"The damage is repairable => estimated cost={est} LKR.")

        manual, route_reasons, route_flags = should_route_manual(
            damages=meaningful,
            conf_score=conf_score,
            overlaps=evidence.overlaps,
            vehicle_area_ratio=evidence.vehicle_area_ratio,
        )
        reasons.extend(route_reasons)
        flags.extend(route_flags)

        if manual:
            return Decision(
                severity=severity,
                route="MANUAL_REVIEW",
                confidence_score=conf_score,
                estimated_cost_lkr=None,
                cost_range_lkr=None,
                pricing_mode="PENDING_REVIEW",
                reasons=reasons,
                flags=flags,
            )

        return Decision(
            severity=severity,
            route="AUTO",
            confidence_score=conf_score,
            estimated_cost_lkr=int(est),
            cost_range_lkr=None,
            pricing_mode="AUTO_POINT",
            reasons=reasons,
            flags=flags,
        )

    # Fallback: no meaningful damages after filtering
    return Decision(
        severity=None,
        route="MANUAL_REVIEW",
        confidence_score=conf_score,
        estimated_cost_lkr=None,
        cost_range_lkr=None,
        pricing_mode="PENDING_REVIEW",
        reasons=reasons + ["No meaningful damages detected after filtering."],
        flags=flags + ["NO_MEANINGFUL_DAMAGE"],
    )
