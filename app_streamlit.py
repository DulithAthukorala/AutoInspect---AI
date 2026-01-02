import os
import streamlit as st
from PIL import Image

from src.inference import DamageDetector
from src.vehicle_mask import VehicleMasker
from src.evidence import extract_evidence
from src.logic import decide_case
from src.explain import generate_explanation



DEFAULT_WEIGHTS = "runs/segment/train/weights/best.pt"
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", DEFAULT_WEIGHTS)

SEVERITY_WORDS = {"LOW": "Minor", "MEDIUM": "Moderate", "HIGH": "Serious"}

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


def _fmt_lkr(x: int) -> str:
    return f"{int(x):,} LKR"


def _fmt_range(rng) -> str:
    lo, hi = rng
    return f"{_fmt_lkr(lo)} – {_fmt_lkr(hi)}"


def filter_reasons_for_manual_review(reasons: list[str]) -> list[str]:
    """
    If MANUAL_REVIEW, do NOT leak cost numbers anywhere.
    """
    cost_keywords = [
        "Base cost",
        "Extra cost",
        "Estimated cost",
        "Estimated repair cost",
        "cost set to",
        "replacement",
        "repair pricing",
        "baseline cost",
        "LKR",
        "band",
        "range",
        "point=",
        "Cosmetic cost",
    ]
    out = []
    for r in reasons:
        if any(k.lower() in r.lower() for k in cost_keywords):
            continue
        out.append(r)
    return out


@st.cache_resource
def load_damage_detector(weights_path: str) -> DamageDetector:
    return DamageDetector(weights_path)


@st.cache_resource
def load_vehicle_masker() -> VehicleMasker:
    return VehicleMasker("yolov8n-seg.pt")


st.set_page_config(page_title="AutoInspect AI", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
      .title { font-size: 2.2rem; font-weight: 850; margin-bottom: 0.15rem; }
      .subtitle { font-size: 1.05rem; color: rgba(49, 51, 63, 0.70); margin-bottom: 0.8rem; }
      .card {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 18px;
        padding: 16px 16px 10px 16px;
        background: rgba(255,255,255,0.70);
        box-shadow: 0 6px 24px rgba(0,0,0,0.04);
      }
      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.9rem;
        border: 1px solid rgba(49,51,63,0.12);
        margin-right: 8px;
        margin-bottom: 8px;
      }
      .pill-ok { background: rgba(0, 200, 0, 0.08); }
      .pill-warn { background: rgba(255, 165, 0, 0.10); }
      .pill-info { background: rgba(0, 120, 255, 0.08); }
      .muted { color: rgba(49, 51, 63, 0.70); }
      .small { font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">AutoInspect AI 🚗</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Claim triage + pre-estimate from a single photo (demo). Clear photos = better results.</div>',
    unsafe_allow_html=True,
)

# ---------- Session state ----------
if "weights_path" not in st.session_state:
    st.session_state.weights_path = WEIGHTS_PATH

if "show_tech" not in st.session_state:
    st.session_state.show_tech = False

if "audience" not in st.session_state:
    st.session_state.audience = "Customer"


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input("Damage model weights path", key="weights_path")

    st.radio(
        "View mode",
        options=["Customer", "Engineer"],
        key="audience",
        help="Customer = simple. Engineer = full evidence + reasons + flags.",
    )

    st.toggle("Show technical details", key="show_tech")

    st.divider()
    st.markdown("**Photo tips**")
    st.markdown(
        "- Full car in frame\n"
        "- Good lighting (avoid glare)\n"
        "- Take 2 angles (front + side)\n"
        "- Keep image sharp"
    )











# Image uploader
uploaded = st.file_uploader("📤 Upload an image (jpg / jpeg / png)", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload an image to start.")
    st.stop()

image_id = uploaded.name
img = Image.open(uploaded).convert("RGB")







# Layout with two columns in the UI
col1, col2 = st.columns([1.15, 0.85], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Uploaded image")
    st.image(img, caption=image_id, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Run pipeline ----------
detector = load_damage_detector(st.session_state.weights_path)
vehicle_masker = load_vehicle_masker()

vehicle_mask = vehicle_masker.predict_vehicle_mask(img)
yolo_res = detector.predict(img)

evidence = extract_evidence(yolo_res, image_id=image_id, vehicle_mask=vehicle_mask)
decision = decide_case(evidence)

# explanation should already obey “no cost leak on manual review”
explanation = generate_explanation(evidence, decision)

severity_word = SEVERITY_WORDS.get(decision.severity, decision.severity)
sure_word = sure_text(float(decision.confidence_score))

# If manual review, filter reasons for UI (double safety)
display_reasons = decision.reasons
if decision.route == "MANUAL_REVIEW":
    display_reasons = filter_reasons_for_manual_review(decision.reasons)

# Handle new logic fields safely (works even if older Decision)
cost_range = getattr(decision, "cost_range_lkr", None)
pricing_mode = getattr(decision, "pricing_mode", None)
flags = getattr(decision, "flags", [])


# ---------- Result panel ----------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Result")

    if decision.route == "MANUAL_REVIEW":
        st.markdown(
            f'<span class="pill pill-warn">Needs human review</span>'
            f'<span class="pill pill-info">Confidence: {sure_word}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="pill pill-ok">Automatic</span>'
            f'<span class="pill pill-info">Confidence: {sure_word}</span>',
            unsafe_allow_html=True,
        )

    st.metric("Damage level", severity_word)

    # Cost policy:
    # - MANUAL_REVIEW: do NOT show point cost; optionally show range if available
    # - AUTO: show point + (optional) range
    if decision.route == "AUTO":
        st.metric("Estimated repair cost", _fmt_lkr(decision.estimated_cost_lkr))
        if cost_range and cost_range != (0, 0):
            st.caption(f"Likely range: {_fmt_range(cost_range)}")
    else:
        st.warning("⚠️ Human review needed. We’ll provide a final estimate after manual inspection.")
        if cost_range and cost_range != (0, 0):
            st.caption(f"Provisional range (signals only): {_fmt_range(cost_range)}")

    # Friendly “no damage” moment
    if decision.route == "AUTO" and decision.estimated_cost_lkr == 0:
        st.success("✅ No visible damage detected.")
        st.caption("If you believe damage exists, upload a clearer full-car photo from another angle.")

    # Tabs to keep UI clean
    tab_customer, tab_details = st.tabs(["🧾 Explanation", "🔎 Details"])

    with tab_customer:
        st.write(explanation)

    with tab_details:
        st.markdown(f"**Image:** `{image_id}`")
        st.markdown(f"**Severity:** `{decision.severity}`")
        st.markdown(f"**Routing:** `{decision.route}` (confidence={float(decision.confidence_score):.2f})")

        if evidence.vehicle_area_ratio is not None:
            st.caption(f"vehicle_area_ratio (vehicle / image): {evidence.vehicle_area_ratio:.2f}")

        st.write("**Detected damage instances**")
        if not evidence.damages:
            st.info("No damage instances detected.")
        else:
            rows = [
                {
                    "damage_type": d.damage_type,
                    "confidence": round(float(d.confidence), 3),
                    "area_%_of_vehicle": round(float(d.area_ratio) * 100.0, 3),
                }
                for d in evidence.damages
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

        st.write("**Decision reasons**")
        for r in display_reasons:
            st.write("-", r)

        # Only show flags in Engineer mode
        if st.session_state.audience == "Engineer" and flags:
            st.write("**Flags**")
            st.write(", ".join(flags))

    # Optional “retake photo” panel for manual review
    if decision.route == "MANUAL_REVIEW":
        with st.expander("📸 How to get an automatic result (recommended photos)"):
            st.markdown(
                "- **Full car photo** (front)\n"
                "- **Full car photo** (side)\n"
                "- If glass/tire/lamp: **close-up** + **full car**\n"
                "- Avoid glare and heavy reflections"
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Extra technical section (only if toggled) ----------
if st.session_state.show_tech:
    st.divider()
    st.subheader("Technical details (auditors / developers)")

    st.markdown(
        f"- **image_id:** `{image_id}`\n"
        f"- **severity:** `{decision.severity}`\n"
        f"- **route:** `{decision.route}`\n"
        f"- **confidence:** `{float(decision.confidence_score):.2f}`"
    )

    if cost_range and cost_range != (0, 0):
        st.markdown(f"- **cost_range_lkr:** `{cost_range}`")

    st.write("**Raw reasons (unfiltered)**")
    for r in decision.reasons:
        st.write("-", r)
