from __future__ import annotations

from typing import Optional, Tuple

from src.logic import CaseEvidence, Decision


def _fmt_lkr(x: int) -> str:
    return f"{int(x):,} LKR"


def _fmt_range(rng: Optional[Tuple[int, int]]) -> str:
    """
    SAFE: rng can be None.
    """
    if not rng:
        return "N/A"
    lo, hi = rng
    return f"{_fmt_lkr(lo)} – {_fmt_lkr(hi)}"


def generate_explanation(evidence: CaseEvidence, decision: Decision) -> str:
    """
    SAFE + Phase-1 policy:
    - If route == MANUAL_REVIEW: DO NOT output any pricing numbers (even ranges).
    - If route == AUTO: pricing allowed.
    - Never crash if cost_range_lkr is None.
    """
    lines: list[str] = []

    # --- Header (user-facing) ---
    if decision.route == "AUTO":
        if decision.severity == "NO_DAMAGE":
            lines.append("✅ No meaningful damage detected in this photo.")
        else:
            sev = decision.severity or "UNKNOWN"
            lines.append(f"✅ Automatic assessment: {sev} damage signals detected.")
    else:
        lines.append("⚠️ This case needs human review to be safe and accurate.")

    # --- Confidence summary ---
    lines.append(f"Confidence score: {float(decision.confidence_score):.2f}")

    # --- Pricing policy ---
    if decision.route == "AUTO":
        # Point estimate (safe if present)
        if decision.estimated_cost_lkr is not None:
            lines.append(f"Estimated repair cost: {_fmt_lkr(int(decision.estimated_cost_lkr))}")

        # Range only if present AND non-empty
        rng = getattr(decision, "cost_range_lkr", None)
        if rng and isinstance(rng, (tuple, list)) and len(rng) == 2:
            lines.append(f"Likely range: {_fmt_range((int(rng[0]), int(rng[1])))}")

    else:
        # Manual review => NO COST NUMBERS AT ALL
        lines.append("We can’t provide a reliable estimate from this photo alone.")
        lines.append("Tip: upload a full-car photo (front + side) in good lighting for an automatic result.")

    # --- Reasons (keep short, but useful) ---
    if getattr(decision, "reasons", None):
        lines.append("")
        lines.append("Why this decision:")
        for r in decision.reasons[:8]:
            lines.append(f"- {r}")

    return "\n".join(lines).strip()
