import json, random
import cv2
import numpy as np
from pathlib import Path

COCO_ROOT = Path(r"data/CarDD_COCO")
ANN_PATH = COCO_ROOT / "annotations" / "instances_train2017.json"
IMG_DIR  = COCO_ROOT / "train2017"

data = json.loads(ANN_PATH.read_text(encoding="utf-8"))
images = data["images"]
anns = data["annotations"]
cats = {c["id"]: c["name"] for c in data["categories"]}

# index annotations by image_id
by_img = {}
for a in anns:
    by_img.setdefault(a["image_id"], []).append(a)

# pick an image that actually has annotations
candidates = [im for im in images if im["id"] in by_img]
im = random.choice(candidates)

img_path = IMG_DIR / im["file_name"]
img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f"Could not read {img_path}")

h, w = img.shape[:2]

for a in by_img[im["id"]]:
    seg = a.get("segmentation", [])
    cid = a.get("category_id")
    name = cats.get(cid, str(cid))

    if isinstance(seg, list):  # polygons
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)  
            pts[:, 0] = np.clip(pts[:, 0], 0, w-1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h-1)
            pts = pts.astype(np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)

    # draw bbox too (optional)
    if "bbox" in a:
        x, y, bw, bh = a["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x+bw), int(y+bh)), (255,0,0), 2)

cv2.imshow("COCO sanity check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Opened:", img_path)
