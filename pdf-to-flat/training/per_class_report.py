"""Per-class precision / recall of an exported ONNX model on a COCO split (IoU >= 0.5).

Usage: python per_class_report.py --data data/coco --split test2017 --onnx export/music_symbols.onnx [--conf 0.3]
       python per_class_report.py --data data/coco --split test2017 --self-check   # GT vs GT, must print 100%
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from classes import CLASSES


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(a[0], b[0])), max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def score(gt_by_img, pred_by_img, thr=0.5):
    tp, fp, fn = np.zeros(len(CLASSES)), np.zeros(len(CLASSES)), np.zeros(len(CLASSES))
    for img_id, gts in gt_by_img.items():
        preds = sorted(pred_by_img.get(img_id, []), key=lambda p: -p[4])
        used = set()
        for p in preds:
            best, bi = 0, -1
            for i, g in enumerate(gts):
                if i in used or g[4] != p[5]:
                    continue
                v = iou(p, g)
                if v > best:
                    best, bi = v, i
            if best >= thr:
                used.add(bi); tp[p[5]] += 1
            else:
                fp[p[5]] += 1
        for i, g in enumerate(gts):
            if i not in used:
                fn[g[4]] += 1
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test2017")
    ap.add_argument("--onnx")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    coco = json.loads((data / "annotations" / f"instances_{args.split}.json").read_text())
    gt = {}
    for a in coco["annotations"]:
        gt.setdefault(a["image_id"], []).append([*a["bbox"], a["category_id"]])
    preds = {}
    if args.self_check:
        preds = {k: [[*g[:4], 1.0, g[4]] for g in v] for k, v in gt.items()}
    else:
        import onnxruntime as ort
        sys_path = Path(__file__).resolve().parent
        import sys
        sys.path.insert(0, str(sys_path))
        from prepare_copyist import prelabel
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        for im in coco["images"]:
            tile = Image.open(data / args.split / im["file_name"])
            preds[im["id"]] = prelabel(sess, tile, conf=args.conf)
    tp, fp, fn = score(gt, preds)
    print(f"{'class':16s} {'GT':>6s} {'prec':>6s} {'recall':>6s}")
    for i, c in enumerate(CLASSES):
        n = tp[i] + fn[i]
        if n == 0 and fp[i] == 0:
            continue
        p = tp[i] / max(1, tp[i] + fp[i]); r = tp[i] / max(1, n)
        print(f"{c:16s} {int(n):6d} {p:6.1%} {r:6.1%}")
    P = tp.sum() / max(1, tp.sum() + fp.sum()); R = tp.sum() / max(1, tp.sum() + fn.sum())
    print(f"{'ALL':16s} {int((tp + fn).sum()):6d} {P:6.1%} {R:6.1%}")


if __name__ == "__main__":
    main()
