from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

@dataclass(frozen=True)  # Nice way to auto-generate __init__ 
class DamageInstance:
    damage_type: str        # e.g., "scratch", "dent", "crack"
    confidence: float       #0..1
    area_ratio: float  


@dataclass(frozen=True)
class CaseEvidence:
    image_id: str
    damages: List[DamageInstance]
    overlaps: Optional[Dict[Tuple[int, int], float]] = None  # optional: overlaps between damages (i,j)->overlap_ratio



@dataclass(frozen=True)
class Decision:
    severity: str          # e.g., "LOW", "MEDIUM", "HIGH"
    estimated_cost_lkr: int
    route: str             # e.g., "AUTO" or "MANUAL_REVIEW"
    confidence_score: float  #0..1
    reasons: List[str]






# Class related functions

DAMAGE_TYPE_WEIGHT = {
    "scratch": 1.0,
    "dent": 1.5,
    "crack": 2.0,
    "glass": 2.5,
}

def type_weight(damage_type: str) -> float:
    return DAMAGE_TYPE_WEIGHT.get(damage_type.lower(), 1.2)



def aggregate_confidence(damages: List[DamageInstance]) -> float:
    if not damages:
        return 0.0

    total_area = sum(d.area_ratio for d in damages)
    if total_area > 0:
        return sum(d.confidence * d.area_ratio for d in damages) / total_area
    return sum(d.confidence for d in damages) / len(damages)

def weighted_damage_score(damages: List[DamageInstance]) -> float:
    """
     main measurable signal used for severity.
    """
    return sum(d.area_ratio * type_weight(d.damage_type) for d in damages)






# Severity thresholds based Functions
SEVERITY_THRESHOLDS = {
    "LOW": 0.010,       # <= 1.0% total damage area 
    "MEDIUM": 0.030,    # <= 3.0% total damage area
    # > 3% => HIGH
}


def assign_severity(damages: List[DamageInstance]) -> Tuple[str, float, List[str]]:
    """
    Returns (severity, score, reasons)
    """
    reasons = []
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








# Cost estimation related functions

BASE_COST_LKR = {
    "LOW": 15000,
    "MEDIUM": 45000,
    "HIGH": 120000,
}

# Additional cost per 1% of (weighted) area
COST_PER_PERCENT_LKR = {
    "LOW": 8000,
    "MEDIUM": 12000,
    "HIGH": 16000,
}


def estimate_cost_lkr(severity: str, weighted_score: float) -> Tuple[int, List[str]]:
    """
    Simple, explainable cost:
      base_cost(severity) + (weighted_score * 100) * cost_per_percent(severity)
    because weighted_score is a fraction; multiply by 100 for percent.
    """
    reasons = []
    base = BASE_COST_LKR[severity]
    per_pct = COST_PER_PERCENT_LKR[severity]
    extra = int(round((weighted_score * 100.0) * per_pct))
    cost = int(base + extra)

    reasons.append(f"Base cost for {severity} = {base} LKR.")
    reasons.append(f"Extra cost = (weighted_score*100)*{per_pct} = {extra} LKR.")
    reasons.append(f"Estimated cost = {cost} LKR.")
    return cost, reasons










# Routing related functions

MIN_CONF_FOR_AUTO = 0.55
MIN_TOTAL_AREA_FOR_TRUST = 0.002   # if extremely tiny, we can be uncertain (false positives)
MAX_OVERLAP_FOR_TRUST = 0.60       # heavy overlaps may indicate confusion/duplicate masks


def should_route_manual(damages: List[DamageInstance],conf_score: float, overlaps: Optional[Dict[Tuple[int, int], float]] = None):
    """
    Route to manual review if:
      - confidence is low
      - damages are extremely tiny
      - overlaps are very high (possible duplicate/confusion)
      - no damages detected
    """
    reasons = []

    if not damages:
        reasons.append("No damages detected => MANUAL_REVIEW (needs confirmation).")
        return True, reasons

    total_area = sum(d.area_ratio for d in damages)
    if conf_score < MIN_CONF_FOR_AUTO:
        reasons.append(f"Aggregate confidence {conf_score:.2f} < {MIN_CONF_FOR_AUTO:.2f} => MANUAL_REVIEW.")
        return True, reasons

    if total_area < MIN_TOTAL_AREA_FOR_TRUST:
        reasons.append(f"Total damage area {total_area:.4f} < {MIN_TOTAL_AREA_FOR_TRUST:.4f} => MANUAL_REVIEW.")
        return True, reasons

    if overlaps:
        max_ov = max(overlaps.values()) if overlaps else 0.0
        if max_ov > MAX_OVERLAP_FOR_TRUST:
            reasons.append(f"Max overlap {max_ov:.2f} > {MAX_OVERLAP_FOR_TRUST:.2f} => MANUAL_REVIEW.")
            return True, reasons

    reasons.append("Meets confidence/area/overlap thresholds => AUTO.")
    return False, reasons







# Main decision function
def decide_case(evidence: CaseEvidence) -> Decision:
    """
    Main entry point: evidence -> decision
    """
    reasons: List[str] = []

    conf_score = aggregate_confidence(evidence.damages)
    reasons.append(f"Aggregate confidence score = {conf_score:.2f} (area-weighted).")

    severity, score, sev_reasons = assign_severity(evidence.damages)
    reasons.extend(sev_reasons)

    cost, cost_reasons = estimate_cost_lkr(severity, score)
    reasons.extend(cost_reasons)

    manual, route_reasons = should_route_manual(evidence.damages, conf_score, evidence.overlaps)
    reasons.extend(route_reasons)

    route = "MANUAL_REVIEW" if manual else "AUTO"

    return Decision(
        severity=severity,
        estimated_cost_lkr=cost,
        route=route,
        confidence_score=float(conf_score),
        reasons=reasons
    )