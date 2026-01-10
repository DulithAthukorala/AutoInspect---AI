import hashlib
import os
from pathlib import Path
from typing import Tuple

from PIL import Image


CASES_DIR = Path(os.getenv("CASES_DIR", "data/cases"))
CASES_DIR.mkdir(parents=True, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def save_uploaded_image(case_id: str, img: Image.Image, raw_bytes: bytes) -> Tuple[str, str]:
    """
    Saves image as JPG for consistent replay.
    Returns (image_path, image_sha256).
    """
    image_hash = sha256_bytes(raw_bytes)
    out_path = CASES_DIR / f"{case_id}.jpg"
    img.save(out_path, format="JPEG", quality=95)
    return str(out_path), image_hash
