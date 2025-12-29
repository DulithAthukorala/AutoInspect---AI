import os
from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import DamageDetector
from src.evidence import extract_evidence
from src.logic import decide_case
from src.explain import generate_explanation, explain_change


# ---------- Config ----------
DEFAULT_WEIGHTS = "runs/segment/train/weights/best.pt"
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", DEFAULT_WEIGHTS)

# Human words (no dev labels)
SEVERITY_WORDS = {
    "LOW": "Minor",
    "MEDIUM": "Moderate",
    "HIGH": "Serious",
}

ROUTE_WORDS = {
    "AUTO": "Automatic (no human needed)",
    "MANUAL_REVIEW": "Human review needed",
}

SURE_WORDS = [
    (0.00, 0.40, "Not sure"),
    (0.40, 0.70, "Somewhat sure"),
    (0.70, 1.01, "Pretty sure"),
]


def sure_text(score: float) -> str:
    for lo, hi, label in SURE_WORDS:
        if lo <= score < hi:
            return label
    return "Unknown"


@st.cache_resource
def load_detector(weights_path: str) -> DamageDetector:
    # Loads model once (very important for performance)
    return DamageDetector(weights_path)


# ---------- UI ----------
st.set_page_config(page_title="AutoInspect AI", layout="wide")
st.title("AutoInspect AI 🚗")
st.caption("Upload a car image. The system will detect damage, estimate severity/cost, and explain it in simple English.")

with st.sidebar:
    st.header("Settings")
    st.text_input("Model weights path", value=WEIGHTS_PATH, key="weights_path")
    show_tech = st.toggle("Show technical details", value=False)
    st.divider()
    st.write("Tip: For deployment, set an env var `WEIGHTS_PATH` to your `best.pt`.")

uploaded = st.file_uploader("Upload an image (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded:
    image_id = uploaded.name
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader("Image")
        st.image(img, caption=image_id, use_container_width=True)

    # --- Run pipeline ---
    detector = load_detector(st.session_state["weights_path"])
    yolo_res = detector.predict(img)  # pass PIL image, not a path
    evidence = extract_evidence(yolo_res, image_id=image_id)
    decision = decide_case(evidence)

    explanation = generate_explanation(evidence, decision)
    change = explain_change(evidence, decision)

    # --- Human-friendly summary ---
    severity_word = SEVERITY_WORDS.get(decision.severity, decision.severity)
    route_word = ROUTE_WORDS.get(decision.route, decision.route)
    sure_word = sure_text(float(decision.confidence_score))

    with col2:
        st.subheader("Result (simple)")
        st.metric("Damage level", severity_word)
        st.metric("Estimated repair cost", f"{decision.estimated_cost_lkr:,} LKR")
        st.metric("What happens next", route_word)
        st.write(f"**How sure are we?** {sure_word}")

        st.divider()
        st.subheader("Why?")
        # keep it readable: show the explain.py output (already structured)
        st.text(explanation)

        st.subheader("Change")
        st.write(change)

    # --- Optional technical details ---
    if show_tech:
        st.divider()
        st.subheader("Technical details (for developers)")

        st.write("**Detected instances**")
        if not evidence.damages:
            st.info("No damage instances detected.")
        else:
            rows = []
            for d in evidence.damages:
                rows.append({
                    "damage_type": d.damage_type,
                    "confidence": round(float(d.confidence), 3),
                    "area_%": round(float(d.area_ratio) * 100.0, 3),
                })
            st.dataframe(rows, use_container_width=True)

        st.write("**Decision reasons**")
        for r in decision.reasons:
            st.write("-", r)

else:
    st.info("Upload an image to start.")
