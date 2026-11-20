"""Turn scans of your copyist's pages into training data.

Step A (tile + pre-label): cuts each staff into model tiles and, if you pass a trained model,
pre-labels every symbol so you only correct mistakes instead of drawing every box.
  python prepare_copyist.py tile --pages scans/*.png --out copyist_work [--onnx music_symbols.onnx]
  -> copyist_work/images/*.png and copyist_work/prelabels_coco.json

Step B (correct): import images + prelabels_coco.json into an annotation tool that reads and
writes COCO 1.0 (for example CVAT). Fix wrong boxes and classes, add missed symbols, then export
as COCO 1.0.

Step C (finalize): split the corrected pages into train/val/test by page and write the files
merge_coco.py and the Kaggle notebook expect.
  python prepare_copyist.py finalize --coco corrected.json --images copyist_work/images --out copyist_dataset
  -> copyist_dataset/{train2017,val2017,test2017}/, copyist_{train,val,test}.json
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from classes import CLASSES, CLASS_ID, coco_categories
from staves import binarize, find_staves, load_page, strip_tiles


def prelabel(session, tile, conf=0.3, iou=0.5):
    a = np.asarray(tile.convert("RGB"), dtype=np.float32)[:, :, ::-1].transpose(2, 0, 1)[None].copy()
    out = session.run(None, {session.get_inputs()[0].name: a})[0][0]
    boxes = []
    for row in out:
        cls = int(np.argmax(row[5:]))
        score = float(row[4] * row[5 + cls])
        if score < conf:
            continue
        cx, cy, w, h = row[:4]
        boxes.append([cx - w / 2, cy - h / 2, w, h, score, cls])
    boxes.sort(key=lambda b: -b[4])
    keep = []
    for b in boxes:  # per-class NMS
        if all(k[5] != b[5] or iou_xywh(k, b) < iou for k in keep):
            keep.append(b)
    return keep


def iou_xywh(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(a[0], b[0])), max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def cmd_tile(args):
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    session = None
    if args.onnx:
        import onnxruntime as ort
        session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    coco = {"images": [], "annotations": [], "categories": coco_categories()}
    iid = aid = 0
    for page_path in sorted(sum([sorted(Path().glob(p)) if not Path(p).exists() else [Path(p)] for p in args.pages], [])):
        gray = load_page(page_path)
        staves = find_staves(binarize(gray))
        print(f"{page_path.name}: {len(staves)} staves")
        for si, (top, bottom, space) in enumerate(staves):
            for tile, x, _, _ in strip_tiles(gray, top, bottom, space):
                iid += 1
                name = f"{page_path.stem}_s{si:02d}_x{x:05d}.png"
                tile.convert("RGB").save(out / "images" / name)
                coco["images"].append({"id": iid, "file_name": name, "width": tile.width, "height": tile.height, "page": page_path.stem})
                if session:
                    for x0, y0, w, h, score, cls in prelabel(session, tile):
                        aid += 1
                        coco["annotations"].append({"id": aid, "image_id": iid, "category_id": cls,
                                                    "bbox": [round(float(v), 1) for v in (x0, y0, w, h)],
                                                    "area": round(float(w * h), 1), "iscrowd": 0, "score": round(score, 3)})
    (out / "prelabels_coco.json").write_text(json.dumps(coco))
    print(f"{iid} tiles, {aid} pre-labels -> {out / 'prelabels_coco.json'}")


def cmd_finalize(args):
    src = json.loads(Path(args.coco).read_text())
    # annotation tools may renumber categories: map by name back to our ids
    cat_name = {c["id"]: c["name"] for c in src["categories"]}
    unknown = sorted({n for n in cat_name.values() if n not in CLASS_ID})
    if unknown:
        print("ignoring unknown categories:", unknown)
    pages = sorted({im.get("page") or im["file_name"].rsplit("_s", 1)[0] for im in src["images"]})
    random.Random(1).shuffle(pages)
    n_val = max(1, round(len(pages) * 0.1))
    n_test = max(1, round(len(pages) * 0.1)) if len(pages) >= 5 else 0
    split_of = {p: ("val" if i < n_val else "test" if i < n_val + n_test else "train") for i, p in enumerate(pages)}
    out = Path(args.out)
    folders = {"train": "train2017", "val": "val2017", "test": "test2017"}
    sets = {s: {"images": [], "annotations": [], "categories": coco_categories()} for s in folders}
    by_img = {}
    for a in src["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for s, f in folders.items():
        (out / f).mkdir(parents=True, exist_ok=True)
    for im in src["images"]:
        page = im.get("page") or im["file_name"].rsplit("_s", 1)[0]
        s = split_of[page]
        name = "copyist_" + Path(im["file_name"]).name
        shutil.copy(Path(args.images) / Path(im["file_name"]).name, out / folders[s] / name)
        sets[s]["images"].append({"id": im["id"], "file_name": name, "width": im["width"], "height": im["height"]})
        for a in by_img.get(im["id"], []):
            n = cat_name.get(a["category_id"])
            if n in CLASS_ID:
                sets[s]["annotations"].append({**a, "category_id": CLASS_ID[n], "iscrowd": 0})
    for s in folders:
        (out / f"copyist_{s}.json").write_text(json.dumps(sets[s]))
        print(f"copyist {s}: {len(sets[s]['images'])} tiles, {len(sets[s]['annotations'])} boxes")
    print(f"Upload {out} as a Kaggle Dataset and set COPYIST_DIR in the notebook.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tile")
    t.add_argument("--pages", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--onnx")
    f = sub.add_parser("finalize")
    f.add_argument("--coco", required=True)
    f.add_argument("--images", required=True)
    f.add_argument("--out", required=True)
    args = ap.parse_args()
    {"tile": cmd_tile, "finalize": cmd_finalize}[args.cmd](args)
