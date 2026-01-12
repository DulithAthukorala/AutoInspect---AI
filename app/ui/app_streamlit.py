import os
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st
from PIL import Image


# ----------------------------
# Config
# ----------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "+94-XX-XXXXXXX")

SEVERITY_WORDS = {
    "NO_DAMAGE": "No damage",
    "LOW": "Minor",
    "MEDIUM": "Moderate",
    "HIGH": "Serious",
    None: "Unknown",
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


def _fmt_lkr(x: int) -> str:
    return f"{int(x):,} LKR"


def _fmt_range(rng: Tuple[int, int]) -> str:
    lo, hi = rng
    return f"{_fmt_lkr(lo)} – {_fmt_lkr(hi)}"


def filter_reasons_for_manual_review(reasons: list[str]) -> list[str]:
    """
    UI safety: If MANUAL_REVIEW, do NOT leak cost numbers anywhere.
    """
    cost_keywords = ["LKR", "band", "range", "point=", "Cosmetic cost", "replacement", "repair", "cost"]
    out = []
    for r in reasons:
        if any(k.lower() in r.lower() for k in cost_keywords):
            continue
        out.append(r)
    return out


def post_assess(
    api_url: str, image_bytes: bytes, filename: str, mime: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns: (success_json, error_json)
    - On 200: success_json filled
    - On error (network / 500): error_json filled
    """
    files = {"file": (filename, image_bytes, mime)}
    try:
        r = requests.post(f"{api_url}/assess", files=files, timeout=120)
    except requests.RequestException as e:
        return None, {"error": "NETWORK_ERROR", "message": str(e)}

    if r.status_code == 200:
        return r.json(), None

    # Non-200 = real crash now (since quality no longer returns 400)
    try:
        payload = r.json()
    except Exception:
        payload = {"raw": r.text}

    return None, {"error": "API_ERROR", "status_code": r.status_code, "detail": payload}


def download_report_link(api_url: str, case_id: str) -> str:
    return f"{api_url}/report/{case_id}"


# ----------------------------
# UI
# ----------------------------
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
      .pill-bad { background: rgba(255, 0, 0, 0.08); }
      .pill-info { background: rgba(0, 120, 255, 0.08); }
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
if "show_tech" not in st.session_state:
    st.session_state.show_tech = False
if "audience" not in st.session_state:
    st.session_state.audience = "Customer"
if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input("API URL", key="api_url", help="FastAPI base URL, e.g. http://localhost:8000")

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

# ---------- Upload ----------
uploaded = st.file_uploader("📤 Upload an image (jpg / jpeg / png)", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload an image to start.")
    st.stop()

image_id = uploaded.name
img = Image.open(uploaded).convert("RGB")
raw = uploaded.getvalue()
mime = uploaded.type or "image/jpeg"

col1, col2 = st.columns([1.15, 0.85], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Uploaded image")
    st.image(img, caption=image_id, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Call API ----------
with st.spinner("Running assessment..."):
    data, err = post_assess(st.session_state.api_url, raw, image_id, mime)

# ---------- Only real errors (network / crash) ----------
if err:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Result")
    st.markdown('<span class="pill pill-bad">Request failed</span>', unsafe_allow_html=True)

    if err.get("error") == "NETWORK_ERROR":
        st.error("Could not reach API.")
        st.write(err.get("message", ""))
        st.caption("Check API_URL and make sure the FastAPI server is running.")
    else:
        st.error(f"API error (HTTP {err.get('status_code')})")
        st.json(err.get("detail", err))

    st.info(f"📞 Support: **{SUPPORT_CONTACT}**")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------- Success (includes low-quality => MANUAL_REVIEW) ----------
decision = data["decision"]
evidence_damages = data.get("damages", [])
vehicle_area_ratio = data.get("vehicle_area_ratio", None)
conf_break = data.get("confidence_breakdown", {})
quality = data.get("quality", {})
case_id = data.get("case_id", "")

severity = decision.get("severity", None)
route = decision.get("route", "MANUAL_REVIEW")
confidence_score = float(decision.get("confidence_score", 0.0))

severity_word = SEVERITY_WORDS.get(severity, str(severity))
sure_word = sure_text(confidence_score)

reasons = decision.get("reasons", [])
flags = decision.get("flags", []) or []

# UI safety for manual review
display_reasons = reasons
if route == "MANUAL_REVIEW":
    display_reasons = filter_reasons_for_manual_review(reasons)

cost_point = decision.get("estimated_cost_lkr", None)
cost_range = decision.get("cost_range_lkr", None)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Result")

    # Quality pill
    if isinstance(quality, dict) and quality:
        if quality.get("ok", True):
            st.markdown('<span class="pill pill-ok">Quality: OK</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<span class="pill pill-warn">Quality: Weak</span>'
                f'<span class="pill pill-info">Score: {float(quality.get("score", 0.0)):.2f}</span>',
                unsafe_allow_html=True,
            )

    # Route pill
    if route == "MANUAL_REVIEW":
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

    # Cost policy
    if route == "AUTO":
        if cost_point is not None:
            st.metric("Estimated repair cost", _fmt_lkr(int(cost_point)))
        if cost_range:
            st.caption(f"Likely range: {_fmt_range(tuple(cost_range))}")
    else:
        st.warning("⚠️ Human review needed. We’ll provide a final estimate after manual inspection.")

    # Friendly “no damage”
    if route == "AUTO" and (severity == "NO_DAMAGE" or int(cost_point or 0) == 0):
        st.success("✅ No visible damage detected.")
        st.caption("If you believe damage exists, upload a clearer full-car photo from another angle.")

    tab_customer, tab_details = st.tabs(["🧾 Explanation", "🔎 Details"])

    with tab_customer:
        st.write(data.get("explanation", ""))

        if case_id:
            st.markdown("**Download report**")
            st.link_button("⬇️ Download JSON report", download_report_link(st.session_state.api_url, case_id))

        # If weak quality, give tips
        if isinstance(quality, dict) and quality and not quality.get("ok", True):
            st.info("Tip: retake with full car in frame + good lighting + avoid blur/zoom.")

    with tab_details:
        st.markdown(f"**case_id:** `{case_id}`")
        st.markdown(f"**image:** `{data.get('image_id', image_id)}`")
        st.markdown(f"**routing:** `{route}` (confidence={confidence_score:.2f})")

        if vehicle_area_ratio is not None:
            st.caption(f"vehicle_area_ratio (vehicle / image): {float(vehicle_area_ratio):.2f}")
        else:
            st.caption("vehicle_area_ratio: None (vehicle mask missing / not confident)")

        if conf_break:
            st.write("**Confidence breakdown**")
            st.json(conf_break)

        st.write("**Detected damage instances**")
        if not evidence_damages:
            st.info("No damage instances detected.")
        else:
            rows = []
            for d in evidence_damages:
                rows.append(
                    {
                        "damage_type": d["damage_type"],
                        "confidence": round(float(d["confidence"]), 3),
                        "area_%_(vehicle_or_fallback)": round(float(d["area_ratio"]) * 100.0, 3),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

        st.write("**Decision reasons**")
        for r in display_reasons:
            st.write("-", r)

        if st.session_state.audience == "Engineer" and flags:
            st.write("**Flags**")
            st.write(", ".join(flags))

        if st.session_state.audience == "Engineer" and quality:
            st.write("**Quality**")
            st.json(quality)

    if route == "MANUAL_REVIEW":
        with st.expander("📸 How to get an automatic result (recommended photos)"):
            st.markdown(
                "- **Full car photo** (front)\n"
                "- **Full car photo** (side)\n"
                "- If glass/tire/lamp: **close-up** + **full car**\n"
                "- Avoid glare and heavy reflections"
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Extra tech ----------
if st.session_state.show_tech:
    st.divider()
    st.subheader("Technical details (auditors / developers)")
    st.json(data)
