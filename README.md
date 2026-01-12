# AutoInspect AI 🚗
Car damage detection + claim triage + pre-estimate from a single photo (Phase-1).

> **Disclaimer:** Estimates are heuristic + for demo purposes.

---

## What this project does
Upload a vehicle photo → AutoInspect AI:
1. Checks basic photo quality (blur/brightness/resolution)
2. Detects the vehicle (vehicle mask)
3. Detects Damage instances (YOLO-V8n segmentation)
4. computes vehicle-relative damage area ratios
5. produces:
   - severity (NO DAMAGE / LOW DAMAGE / MEDIUM DAMAGE / HIGH DAMAGE)
   - routing decision (AUTO vs MANUAL_REVIEW)
   - confidence breakdown (detection × coverage × consistency)
   - explanation text
6. stores a reproducible JSON report (case_id) + supports replay

---

## Key features (Phase-1)
- ✅ Damage segmentation inference (Ultralytics YOLO)
- ✅ Vehicle segmentation mask (COCO vehicles)
- ✅ Evidence extraction:
  - damage area ratio (vehicle-relative if mask exists, else image-relative fallback)
  - mask overlap IoU (consistency signal)
- ✅ Deterministic decision engine:
  - noise filtering
  - severity scoring
  - auto vs manual routing
  - safe NO_DAMAGE logic (only when visibility is sufficient)
- ✅ Confidence decomposition:
  - **detection** (area-weighted)
  - **coverage** (vehicle framing)
  - **consistency** (mask overlap)
  - **aggregate**
- ✅ Image quality gate (hard reject only for extreme cases)
- ✅ FastAPI backend:
  - `POST /assess`
  - `GET /case/{case_id}`
  - `GET /report/{case_id}`
  - `POST /replay/{case_id}`
- ✅ Streamlit frontend (talks to API)
- ✅ SQLite case storage (report JSON + metadata)

---

## Demo screenshots
> Add screenshots here later (`/assets/...png`)

---

## Tech stack
- Python
- Ultralytics YOLO
- FastAPI + Uvicorn
- Streamlit
- SQLite
- NumPy + Pillow
- Docker
---

## Project structure
```text
src/
  inference.py
  vehicle_mask.py
  evidence.py
  logic.py
  quality.py
  explain.py
  db.py
  storage.py
app_fastapi.py
streamlit_app.py
