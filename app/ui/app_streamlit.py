import os
import requests
import streamlit as st
from PIL import Image

# =========================
# CONFIG
# =========================
API_BATCH_URL = os.getenv("API_BATCH_URL", "http://localhost:8000/assess_batch")
TIMEOUT_SEC = 180
MAX_FILES = 6

INSTRUCTION_IMAGE_PATH = "app/ui/hw_to_take.jpg"  # <-- change to your actual filename

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Auto-Inspect AI", page_icon="🚗", layout="wide")
st.title("AUTO-INSPECT AI")
st.subheader("Vehicle Damage Detection with AI")

# =========================
# INSTRUCTIONS (with image)
# =========================
c1, c2 = st.columns([1, 2], gap="large")

with c1:
    if os.path.exists(INSTRUCTION_IMAGE_PATH):
        try:
            img = Image.open(INSTRUCTION_IMAGE_PATH)
            st.image(img, caption="Take Your Damaged Photo Like this", use_container_width=True)
        except Exception:
            st.warning("Couldn't open the instruction image. Check the file is a valid image.")
    else:
        st.info(f"Add your instruction image next to this file: `{INSTRUCTION_IMAGE_PATH}`")

with c2:
    st.markdown(
        """
### 📸 Take Your Pictures like this to get the best Results
- One photo should only include **one damaged part**
- take close-up photos of the damage
- don't take wide shots of the whole vehicle
- Avoid including license plates or personal info
- Good lighting, not blurry, no heavy shadows
- If you have multiple damages → upload **multiple photos** (up to 6)
"""
    )
st.divider()

# =========================
# UPLOAD Pictures
# =========================
files = st.file_uploader(
    f"Upload up to {MAX_FILES} damage photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if files:
    if len(files) > MAX_FILES:
        st.error(f"Upto {MAX_FILES} files allowed. You uploaded {len(files)}. If you have more contact support.")
        st.stop()

    st.write(f"✅ Uploaded: **{len(files)}** image(s)")

    if st.button("🔍 Analyze All", type="primary"):
        with st.spinner("Running batch analysis..."):
            try:
                multipart = [("files", (f.name, f.getvalue(), f.type)) for f in files]
                r = requests.post(API_BATCH_URL, files=multipart, timeout=TIMEOUT_SEC)
                r.raise_for_status()
                data = r.json()
            except requests.Timeout:
                st.error("API timeout. Try smaller images or check your server.")
                st.stop()
            except requests.RequestException as e:
                st.error(f"API request failed: {e}")
                st.stop()

        items = data.get("items", [])
        if not items:
            st.error("No items returned from API.")
            st.json(data)
            st.stop()

        # =========================
        # AUTO-SUM 
        # =========================
        auto_total_point = 0
        auto_total_lo = 0
        auto_total_hi = 0
        any_range = False
        excluded_manual = 0
        excluded_none = 0

        for it in items:
            d = it.get("decision", {})
            pm = d.get("pricing_mode")
            route = d.get("route")

            if route == "MANUAL_REVIEW" or pm == "PENDING_REVIEW":
                excluded_manual += 1
                continue
            if pm == "NONE":
                excluded_none += 1
                continue

            if pm == "AUTO_POINT" and d.get("estimated_cost_lkr") is not None:
                val = int(d["estimated_cost_lkr"])
                auto_total_point += val
                auto_total_lo += val
                auto_total_hi += val

            elif pm == "AUTO_RANGE" and d.get("cost_range_lkr"):
                lo, hi = d["cost_range_lkr"]
                auto_total_lo += int(lo)
                auto_total_hi += int(hi)
                any_range = True

        # =========================
        # SHOW PER-IMAGE CARDS
        # =========================
        st.subheader("Results (per image)")

        cols = st.columns(3, gap="large")  # 3 per row
        for idx, it in enumerate(items):
            col = cols[idx % 3]
            with col:
                image_id = it.get("image_id", f"image_{idx+1}")
                decision = it.get("decision", {})
                route = decision.get("route", "—")
                pm = decision.get("pricing_mode", "—")
                conf = float(decision.get("confidence_score", 0.0))
                sev = decision.get("severity", None)

                # Status tag
                if route == "MANUAL_REVIEW" or pm == "PENDING_REVIEW":
                    st.error(f"🧑‍⚖️ MANUAL REVIEW")
                elif pm == "NONE":
                    st.success("✅ NO DAMAGE")
                else:
                    st.info("⚡ AUTO")

                # Image preview
                # (we use the uploaded file list to render, matched by index)
                st.image(files[idx].getvalue(), caption=image_id, use_container_width=True)

                # Quick facts
                st.write(f"**Severity:** {sev if sev else '—'}")
                st.write(f"**Confidence:** {conf:.2f}")

                # Pricing display
                if pm == "AUTO_POINT" and decision.get("estimated_cost_lkr") is not None:
                    st.write(f"**Estimated cost:** LKR {int(decision['estimated_cost_lkr']):,}")
                elif pm == "AUTO_RANGE" and decision.get("cost_range_lkr"):
                    lo, hi = decision["cost_range_lkr"]
                    st.write(f"**Replacement range:** LKR {int(lo):,} – {int(hi):,}")
                elif pm == "NONE":
                    st.write("**Cost:** LKR 0")
                else:
                    st.write("**Cost:** —")

                # Optional: Reasons (expand)
                with st.expander("Why?"):
                    reasons = decision.get("reasons", [])
                    if reasons:
                        for rr in reasons:
                            st.write(f"- {rr}")
                    else:
                        st.write("No reasons provided.")

        st.divider()

        # =========================
        # FINAL TOTAL (AUTO-SUM)
        # =========================
        st.subheader("Final total (AUTO-approved only)")

        if excluded_manual > 0:
            st.warning(f"⚠️ {excluded_manual} image(s) need MANUAL REVIEW — excluded from auto total.")
        if excluded_none > 0:
            st.info(f"ℹ️ {excluded_none} image(s) were NO DAMAGE — excluded from total.")

        if any_range:
            st.success(f"**Total estimate (range):** LKR {auto_total_lo:,} – {auto_total_hi:,}")
        else:
            st.success(f"**Total estimate (point):** LKR {auto_total_point:,}")

        # Optional: show API’s own final_total too (for transparency)
        with st.expander("API final_total (debug/trace)"):
            st.json(data.get("final_total", {}))
