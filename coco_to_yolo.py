import json
import shutil
from pathlib import Path


COCO_ROOT = Path(r"data/CarDD_COCO")
OUT_ROOT  = Path(r"yolo_data")        


def make_dirs():
    (OUT_ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / "val").mkdir(parents=True, exist_ok=True)

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_cat_mapping(categories):
    """
    COCO category ids are not always 0 to N-1
    YOLO expects class ids 0..N-1.
    Created this to make it acending order by COCO id.
    """
    cats_sorted = sorted(categories, key=lambda c: c["id"])
    coco_id_to_yolo = {c["id"]: i for i, c in enumerate(cats_sorted)}
    names = [c.get("name", str(c["id"])) for c in cats_sorted]
    return coco_id_to_yolo, names

def normalize_polygon(poly, w, h):
    """
    COCO polygon is [x1,y1,x2,y2,...] in pixel coords.
    YOLO-seg wants normalized coords: x/w, y/h for each point.
    Where is this object relative to the image?
    """
    out = []
    for i in range(0, len(poly), 2):
        x = poly[i] / w
        y = poly[i + 1] / h
        # clamp to [0,1] for safety
        x = 0.0 if x < 0 else 1.0 if x > 1 else x
        y = 0.0 if y < 0 else 1.0 if y > 1 else y
        out.extend([x, y])
    return out

def convert_split(split_name: str, ann_file: Path, img_dir: Path, coco_id_to_yolo: dict):
    data = load_json(ann_file)
    images = data["images"]
    anns = data["annotations"]

    # image_id -> image info
    img_by_id = {im["id"]: im for im in images}

    # image_id -> list of annotations
    ann_by_img = {}
    for a in anns:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    out_img_dir = OUT_ROOT / "images" / split_name
    out_lab_dir = OUT_ROOT / "labels" / split_name

    converted = 0
    skipped_no_seg = 0
    skipped_missing_img = 0

    for image_id, im in img_by_id.items():
        file_name = im["file_name"]
        w, h = im["width"], im["height"]

        src_img_path = img_dir / file_name
        if not src_img_path.exists():
            skipped_missing_img += 1
            continue

        # write label file even if empty (YOLO allows empty = background image)
        label_lines = []

        for a in ann_by_img.get(image_id, []):
            seg = a.get("segmentation", None)
            if seg is None:
                skipped_no_seg += 1
                continue

            # We handle polygon segmentation (list).
            # If it's RLE dict, we skip (we can add RLE support later if needed).
            if not isinstance(seg, list) or len(seg) == 0:
                skipped_no_seg += 1
                continue

            coco_cat_id = a["category_id"]
            if coco_cat_id not in coco_id_to_yolo:
                continue
            cls = coco_id_to_yolo[coco_cat_id]

            # COCO can store multiple polygons per object -> we write each polygon as a separate line.
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue  # need at least 3 points
                norm = normalize_polygon(poly, w, h)

                # YOLOv8 segmentation label line:
                # class x1 y1 x2 y2 ...
                line = str(cls) + " " + " ".join(f"{v:.6f}" for v in norm)
                label_lines.append(line)

        # copy image
        dst_img_path = out_img_dir / file_name
        shutil.copy2(src_img_path, dst_img_path)

        # save labels with same stem
        label_path = out_lab_dir / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        converted += 1

    print(f"\n[{split_name}] done")
    print(" converted images:", converted)
    print(" skipped (missing image file):", skipped_missing_img)
    print(" skipped annotations (no/invalid segmentation):", skipped_no_seg)

def write_data_yaml(class_names):
    yaml_text = f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val

names:
"""
    for i, n in enumerate(class_names):
        yaml_text += f"  {i}: {n}\n"

    (OUT_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print("\nWrote:", OUT_ROOT / "data.yaml")

def main():
    make_dirs()

    train_ann = COCO_ROOT / "annotations" / "instances_train2017.json"
    val_ann   = COCO_ROOT / "annotations" / "instances_val2017.json"
    train_img = COCO_ROOT / "train2017"
    val_img   = COCO_ROOT / "val2017"

    # build mapping from TRAIN file categories (usually same for val)
    train_data = load_json(train_ann)
    coco_id_to_yolo, class_names = build_cat_mapping(train_data["categories"])

    print("Classes (YOLO id -> name):")
    for i, n in enumerate(class_names):
        print(f"  {i}: {n}")

    convert_split("train", train_ann, train_img, coco_id_to_yolo)
    convert_split("val",   val_ann,   val_img,   coco_id_to_yolo)

    write_data_yaml(class_names)

    print("\n✅ Conversion complete.")
    print("Next: run a YOLOv8 quick check training with: yolo segment train data=yolo_data/data.yaml model=yolov8n-seg.pt epochs=1 imgsz=640")

if __name__ == "__main__":
    main()
