from __future__ import annotations

from typing import List, Tuple

from .logic import CaseEvidence, Decision


def _bullet(lines: List[str]) -> str:
    return "\n".join(f"- {x}" for x in lines)


def _fmt_lkr(x: int) -> str:
    return f"{x:,} LKR"


def _fmt_range(rng: Tuple[int, int]) -> str:
    lo, hi = rng
    return f"{_fmt_lkr(int(lo))} – {_fmt_lkr(int(hi))}"


def _friendly_severity(sev: str) -> str:
    sev = (sev or "").upper()
    return {"LOW": "Minor", "MEDIUM": "Moderate", "HIGH": "Serious"}.get(sev, sev)


def _friendly_route(route: str) -> str:
    route = (route or "").upper()
    return {"AUTO": "Automatic", "MANUAL_REVIEW": "Human review needed"}.get(route, route)


def summarize_detections(evidence: CaseEvidence) -> List[str]:
    """
    Human-facing summary of detections.
    IMPORTANT: area_ratio is vs vehicle (per your evidence.py change).
    """
    if not evidence.damages:
        return ["No meaningful damage was detected."]

    out: List[str] = []
    for d in evidence.damages:
        out.append(
            f"{d.damage_type} (confidence {d.confidence:.2f}, covers ~{d.area_ratio*100:.2f}% of the vehicle)"
        )
    return out


def _counterfactual_text(decision: Decision) -> str:
    """
    Simple, human counterfactual.
    Uses only decision output (no new reasoning).
    """
    if decision.severity == "HIGH":
        return "If the detected damage covered a smaller area, the damage level could drop to Moderate."
    if decision.severity == "MEDIUM":
        return "If the detected damage covered a smaller area, the damage level could drop to Minor."
    return "If a larger or more serious damage was detected, the damage level could increase."


def generate_explanation(evidence: CaseEvidence, decision: Decision) -> str:
    """
    Creates a clean, human-first report.
    Rules:
    - Never leak a misleading point cost when manual review is needed.
    - Prefer cost range for transparency.
    - Always include: what, why, uncertainty, counterfactual.
    """
    lines: List[str] = []

    friendly_sev = _friendly_severity(decision.severity)
    friendly_route = _friendly_route(decision.route)

    # ----------------------------
    # Header (human-first)
    # ----------------------------
    lines.append("Summary")
    lines.append(_bullet([
        f"Image: {evidence.image_id}",
        f"Damage level: {friendly_sev}",
        f"Next step: {friendly_route} (confidence={decision.confidence_score:.2f})",
    ]))

    # ----------------------------
    # Cost messaging (policy-safe)
    # ----------------------------
    # Only show point estimate for AUTO. For MANUAL_REVIEW, show range only (or nothing).
    if decision.route == "AUTO":
        if hasattr(decision, "cost_range_lkr") and decision.cost_range_lkr != (0, 0):
            lines.append("")
            lines.append("Estimated repair cost")
            lines.append(_bullet([
                f"Estimated cost: {_fmt_lkr(int(decision.estimated_cost_lkr))}",
                f"Likely range: {_fmt_range(decision.cost_range_lkr)}",
            ]))
        else:
            lines.append("")
            lines.append("Estimated repair cost")
            lines.append(_bullet([f"Estimated cost: {_fmt_lkr(int(decision.estimated_cost_lkr))}"]))
    else:
        # Manual review: never show point estimate
        lines.append("")
        lines.append("Repair cost")
        if hasattr(decision, "cost_range_lkr") and decision.cost_range_lkr != (0, 0):
            lines.append(_bullet([
                "A human review is required before we can provide a final amount.",
                f"Provisional range (based on detected signals): {_fmt_range(decision.cost_range_lkr)}",
            ]))
        else:
            lines.append(_bullet([
                "A human review is required before we can provide a repair cost estimate.",
            ]))

    # ----------------------------
    # What was detected
    # ----------------------------
    lines.append("")
    lines.append("What we detected")
    lines.append(_bullet(summarize_detections(evidence)))

    # ----------------------------
    # Why this decision happened
    # ----------------------------
    lines.append("")
    lines.append("Why we decided this")
    # Keep reasons, but make them readable
    if decision.reasons:
        lines.append(_bullet(decision.reasons))
    else:
        lines.append(_bullet(["No additional decision details available."]))

    # ----------------------------
    # Uncertainty / flags
    # ----------------------------
    lines.append("")
    lines.append("Uncertainty")
    if decision.route == "MANUAL_REVIEW":
        lines.append(_bullet([
            "We are not confident enough to auto-approve this case.",
            "A human reviewer can confirm damage type and pricing from the photo(s).",
        ]))
    else:
        lines.append(_bullet([
            "Confidence is high enough to process automatically.",
            "If you disagree with the result, upload another photo from a different angle.",
        ]))

    # Include flags if present (developer-friendly but still readable)
    if hasattr(decision, "flags") and decision.flags:
        lines.append("")
        lines.append("Signals we used (technical)")
        lines.append(_bullet([f"{f}" for f in decision.flags]))

    # ----------------------------
    # Counterfactual (what-if)
    # ----------------------------
    lines.append("")
    lines.append("What would change the result?")
    lines.append(_bullet([_counterfactual_text(decision)]))

    return "\n".join(lines)
