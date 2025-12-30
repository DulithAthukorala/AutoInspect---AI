import os
from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import DamageDetector
from src.evidence import extract_evidence
from src.logic import decide_case
from src.explain import generate_explanation


# ---------- Config ----------
DEFAULT_WEIGHTS = "runs/segment/train/weights/best.pt"
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", DEFAULT_WEIGHTS)

# Human-friendly words
SEVERITY_WORDS = {
    "LOW": "Minor",
    "MEDIUM": "Moderate",
    "HIGH": "Serious",
}

ROUTE_WORDS = {
    "AUTO": "Automatic (No Human review needed)",
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


@st.cache_resource # to avoid reloading model on every interaction
def load_detector(weights_path: str) -> DamageDetector:
    return DamageDetector(weights_path)


# ---------- UI ----------
st.set_page_config(page_title="AutoInspect AI", layout="wide")
st.title("AutoInspect AI - Automated car Damage Detector")
st.markdown("Upload Your Damaged Car Image for Automated Damage Assessment and Repair Cost Estimation")

with st.sidebar:
    with st.expander("⚙️ Settings"):
        weights_path = st.text_input(
            "Path For the Model",
            value=WEIGHTS_PATH,
            key="weights_path"
        )
        show_tech = st.toggle("Show technical details", value=False)
        if show_tech:
            st.markdown("""_Technical details included in the Bottom For Developers_""")

uploaded = st.file_uploader("Upload an image (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded:
    image_id = uploaded.name   # use filename as image_id
    img = Image.open(uploaded).convert("RGB")   # ensure 3 channels

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

    # --- Human-friendly summary ---
    severity_word = SEVERITY_WORDS.get(decision.severity, decision.severity)
    route_word = ROUTE_WORDS.get(decision.route, decision.route)
    sure_word = sure_text(float(decision.confidence_score))

    with col2:
        st.subheader("Our Estimation Of Your Car Damage")
        st.metric("Damage level", severity_word)
        if route_word == "Human review needed":
            st.metric("Estimated repair cost", f"{decision.estimated_cost_lkr:,} LKR","(Final cost may vary after human review)")
        else:
            st.metric("Estimated repair cost", f"{decision.estimated_cost_lkr:,} LKR")
        if route_word == "Human review needed":
            st.warning("⚠️ This case requires human review due to the uncertainty of the damage.")
            st.write("Our Customer Support team will get back to you shortly.")
        else:
            st.success("✅ This case can be processed automatically.")
            st.write(f"**How sure are we?** {sure_word}")

        st.divider()
        more = st.button("Click Here to See More Details")
        if more:
            show_tech = True










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
