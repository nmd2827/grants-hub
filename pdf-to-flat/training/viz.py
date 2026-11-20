"""Draw COCO boxes on a few tiles to check annotations by eye.

Usage: python viz.py data/coco/annotations/muscima_train.json data/coco/train2017 out_dir [count]
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from classes import CLASSES

ann_path, img_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
count = int(sys.argv[4]) if len(sys.argv) > 4 else 4
coco = json.loads(ann_path.read_text())
by_img = {}
for a in coco["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)
out_dir.mkdir(parents=True, exist_ok=True)
random.seed(1)
for im in random.sample(coco["images"], min(count, len(coco["images"]))):
    img = Image.open(img_dir / im["file_name"]).convert("RGB")
    d = ImageDraw.Draw(img)
    for a in by_img.get(im["id"], []):
        x, y, w, h = a["bbox"]
        col = (220, 30, 30) if "notehead" in CLASSES[a["category_id"]] else (30, 120, 220)
        d.rectangle([x, y, x + w, y + h], outline=col)
        d.text((x, max(0, y - 9)), CLASSES[a["category_id"]][:6], fill=col)
    img.save(out_dir / im["file_name"])
    print(out_dir / im["file_name"])
