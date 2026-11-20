"""Convert MUSCIMA++ v2 (handwritten) into staff-strip tiles with COCO annotations.

Each staff becomes a strip from 4.5 staff spaces above to 4.5 below it, rescaled so one staff
space is SPACE_PX pixels, then cut into TILE_W x TILE_H tiles. Writes annotations/muscima_{train,val,test}.json;
run merge_coco.py afterwards. That is the same geometry the
browser converter feeds the model.

Usage:
  python prepare_muscima.py --download --root data/muscima --out data/coco
  python prepare_muscima.py --root data/muscima --out data/coco      # data already downloaded

Data license: CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution to MUSCIMA++ and CVC-MUSCIMA).
"""
import argparse
import json
import re
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from classes import CLASS_ID, MUSCIMA_MAP, SPACE_PX, TILE_W, TILE_H, TILE_STRIDE, coco_categories

URL_ANN = "https://github.com/OMR-Research/muscima-pp/releases/download/v2.0/MUSCIMA-pp_v2.0.zip"
URL_IMG = "https://github.com/apacha/OMR-Datasets/releases/download/datasets/CVC_MUSCIMA_PP_Annotated-Images.zip"

# split by writer so no writer appears in more than one split
SPLIT_BY_WRITER = {"val": set(range(41, 46)), "test": set(range(46, 51))}


def split_for(writer):
    for name, ws in SPLIT_BY_WRITER.items():
        if writer in ws:
            return name
    return "train"


def download(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for url, name in [(URL_ANN, "ann.zip"), (URL_IMG, "img.zip")]:
        z = root / name
        if not z.exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(root / name.replace(".zip", ""))


def parse_nodes(xml_path):
    nodes = []
    for n in ET.parse(xml_path).getroot().iter("Node"):
        nodes.append({
            "cls": n.findtext("ClassName"),
            "top": int(n.findtext("Top")), "left": int(n.findtext("Left")),
            "w": int(n.findtext("Width")), "h": int(n.findtext("Height")),
        })
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/muscima")
    ap.add_argument("--out", default="data/coco")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--min-visible", type=float, default=0.5, help="keep a clipped box if this share of it is inside the tile")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    if args.download:
        download(root)
    ann_files = sorted(root.glob("ann/**/annotations/*.xml"))
    img_dir = next(root.glob("img/**/fulls"))
    if not ann_files:
        raise SystemExit("No MUSCIMA++ annotations found; run with --download")

    coco = {s: {"images": [], "annotations": [], "categories": coco_categories()} for s in ("train", "val", "test")}
    folder = {"train": "train2017", "val": "val2017", "test": "test2017"}
    for s in folder.values():
        (out / s).mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    next_img, next_ann = 1, 1
    skipped = {}

    for xf in ann_files:
        doc = xf.stem
        m = re.search(r"W-(\d+)", doc)
        split = split_for(int(m.group(1)) if m else 0)
        img_path = img_dir / f"{doc}.png"
        if not img_path.exists():
            skipped[doc] = "image missing"
            continue
        page = Image.open(img_path).convert("L")
        # CVC-MUSCIMA "ideal" images are white ink on black: normalise to black ink on white
        if sum(page.resize((64, 64)).tobytes()) / 4096 < 128:
            page = Image.eval(page, lambda v: 255 - v)
        nodes = parse_nodes(xf)
        staves = sorted([n for n in nodes if n["cls"] == "staff"], key=lambda n: n["top"])
        symbols = [n for n in nodes if n["cls"] in MUSCIMA_MAP]
        for si, st in enumerate(staves):
            space = st["h"] / 4.0
            if space < 4:
                continue
            y0 = st["top"] - 4.5 * space
            y1 = st["top"] + st["h"] + 4.5 * space
            scale = SPACE_PX / space
            strip = page.crop((0, int(y0), page.width, int(y1)))
            sw, sh = max(1, round(strip.width * scale)), max(1, round(strip.height * scale))
            strip = strip.resize((sw, sh), Image.BILINEAR)
            boxes = []
            for n in symbols:
                bx0, by0 = (n["left"]) * scale, (n["top"] - y0) * scale
                bx1, by1 = (n["left"] + n["w"]) * scale, (n["top"] + n["h"] - y0) * scale
                boxes.append((CLASS_ID[MUSCIMA_MAP[n["cls"]]], bx0, by0, bx1, by1))
            x = 0
            while True:
                tile = Image.new("L", (TILE_W, TILE_H), 255)
                tile.paste(strip.crop((x, 0, min(x + TILE_W, sw), min(TILE_H, sh))), (0, 0))
                anns = []
                for cid, bx0, by0, bx1, by1 in boxes:
                    cx0, cy0 = max(bx0 - x, 0), max(by0, 0)
                    cx1, cy1 = min(bx1 - x, min(TILE_W, sw - x)), min(by1, min(TILE_H, sh))
                    if cx1 <= cx0 or cy1 <= cy0:
                        continue
                    full = (bx1 - bx0) * (by1 - by0)
                    vis = (cx1 - cx0) * (cy1 - cy0)
                    if full <= 0 or vis / full < args.min_visible:
                        continue
                    anns.append((cid, cx0, cy0, cx1 - cx0, cy1 - cy0))
                if anns:
                    name = f"{doc}_s{si:02d}_x{x:05d}.png"
                    tile.convert("RGB").save(out / folder[split] / name)
                    coco[split]["images"].append({"id": next_img, "file_name": name, "width": TILE_W, "height": TILE_H})
                    for cid, bx, by, bw, bh in anns:
                        coco[split]["annotations"].append({"id": next_ann, "image_id": next_img, "category_id": cid,
                                                           "bbox": [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
                                                           "area": round(bw * bh, 1), "iscrowd": 0})
                        next_ann += 1
                    next_img += 1
                if x + TILE_W >= sw:
                    break
                x += TILE_STRIDE

    for s in ("train", "val", "test"):
        p = out / "annotations" / f"muscima_{s}.json"
        p.write_text(json.dumps(coco[s]))
        print(f"{s}: {len(coco[s]['images'])} tiles, {len(coco[s]['annotations'])} boxes")
    if skipped:
        print("skipped:", skipped)


if __name__ == "__main__":
    main()
