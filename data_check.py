import json
from pathlib import Path
from collections import Counter

COCO_ROOT = Path(r"data/CarDD_COCO")
ANN_PATH = COCO_ROOT / "annotations" / "instances_train2017.json"

def main():
    data = json.loads(ANN_PATH.read_text(encoding="utf-8"))

    # Basic keys
    print("Top-level keys:", list(data.keys()))

    images = data.get("images", [])
    anns = data.get("annotations", [])
    cats = data.get("categories", [])

    print("\nCounts")
    print(" images:", len(images))
    print(" annotations:", len(anns))
    print(" categories:", len(cats))

    # Categories
    print("\nCategories (id -> name):")
    for c in sorted(cats, key=lambda x: x["id"]):
        print(f'  {c["id"]} -> {c.get("name")}')

    # Annotation fields check
    sample = anns[0]
    print("\nSample annotation keys:", list(sample.keys()))
    print("Has bbox?", "bbox" in sample)
    print("Has segmentation?", "segmentation" in sample)
    print("iscrowd:", sample.get("iscrowd"))

    # Segmentation type check
    seg = sample.get("segmentation")
    if isinstance(seg, list):
        print("\nSegmentation looks like polygons (list). Good for YOLO-seg.")
        print("Example polygon length:", len(seg[0]) if seg and isinstance(seg[0], list) else "N/A")
    elif isinstance(seg, dict):
        print("\nSegmentation is RLE (dict). Still usable, but conversion changes.")
    else:
        print("\nSegmentation missing/unknown format.")

    # How many annotations per category
    cat_counts = Counter(a["category_id"] for a in anns if "category_id" in a)
    print("\nAnnotations per category_id (top 10):")
    for k, v in cat_counts.most_common(10):
        print(" ", k, ":", v)

    # Image-id join sanity
    img_ids = {im["id"] for im in images}
    bad = sum(1 for a in anns if a.get("image_id") not in img_ids)
    print("\nAnnotations with missing image_id match:", bad)

if __name__ == "__main__":
    main()
