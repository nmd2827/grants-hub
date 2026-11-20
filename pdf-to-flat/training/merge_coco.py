"""Merge per-source COCO files into the instances_*.json files YOLOX reads.

Sources (whichever exist in <out>/annotations): muscima_*, synth_*, copyist_*.
  train = muscima_train + synth_train + copyist_train (copyist pages repeated --copyist-repeat times)
  val   = muscima_val + copyist_val       (real handwriting only, so the score means something)
  test  = muscima_test + copyist_test

Usage: python merge_coco.py --out data/coco [--copyist-repeat 5]
"""
import argparse
import json
from pathlib import Path

from classes import coco_categories

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/coco")
ap.add_argument("--copyist-repeat", type=int, default=5, help="oversample the copyist's own pages")
args = ap.parse_args()
ann = Path(args.out) / "annotations"
plan = {
    "train2017": [("muscima_train", 1), ("synth_train", 1), ("copyist_train", args.copyist_repeat)],
    "val2017": [("muscima_val", 1), ("copyist_val", 1)],
    "test2017": [("muscima_test", 1), ("copyist_test", 1)],
}
for target, sources in plan.items():
    merged = {"images": [], "annotations": [], "categories": coco_categories()}
    iid = aid = 0
    for name, rep in sources:
        p = ann / f"{name}.json"
        if not p.exists():
            continue
        src = json.loads(p.read_text())
        for _ in range(rep):
            remap = {}
            for im in src["images"]:
                iid += 1
                remap[im["id"]] = iid
                merged["images"].append({**im, "id": iid})
            for a in src["annotations"]:
                aid += 1
                merged["annotations"].append({**a, "id": aid, "image_id": remap[a["image_id"]]})
        print(f"{target}: + {name} x{rep} ({len(src['images'])} tiles)")
    (ann / f"instances_{target}.json").write_text(json.dumps(merged))
    print(f"{target}: {len(merged['images'])} tiles, {len(merged['annotations'])} boxes")
