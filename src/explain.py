from __future__ import annotations
from typing import List

from .logic import CaseEvidence, Decision, weighted_damage_score, SEVERITY_THRESHOLDS


def _bullet(lines: List[str]) -> str:
    """Helper: join bullets nicely."""
    return "\n".join(f"- {x}" for x in lines)


def summarize_detections(evidence: CaseEvidence) -> List[str]:
    """
    Returns short bullet lines describing what was detected.
    Uses ONLY evidence fields.
    """
    if not evidence.damages:
        return ["No damage instances were detected."]

    out = []
    for d in evidence.damages:
        out.append(
            f"{d.damage_type} (conf={d.confidence:.2f}, area={d.area_ratio*100:.2f}% of image)"
        )
    return out



def generate_explanation(evidence: CaseEvidence, decision: Decision) -> str:
    """
    Main function: creates a human explanation.
    Uses ONLY structured facts from evidence + decision.
    """
    lines: List[str] = []

    # Header summary
    lines.append(f"Image: {evidence.image_id}")
    lines.append(f"Severity: {decision.severity}")
    lines.append(f"Estimated cost: {decision.estimated_cost_lkr} LKR")
    lines.append(f"Routing: {decision.route} (confidence={decision.confidence_score:.2f})")
    lines.append("")

    # What was detected
    lines.append("Detected damage instances:")
    lines.append(_bullet(summarize_detections(evidence)))
    lines.append("")

    # Why this decision happened (already computed deterministically in logic.py)
    lines.append("Decision reasons:")
    lines.append(_bullet(decision.reasons))
    lines.append("")

    return "\n".join(lines)
