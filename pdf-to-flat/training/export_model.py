"""Export a trained checkpoint to ONNX for the browser (ONNX Runtime Web).

Writes:
  music_symbols.onnx       fp32, boxes already decoded: output [1, N, 5 + num_classes]
                           rows = cx, cy, w, h, objectness, class scores (pixels of the 320x1024 tile)
  music_symbols.fp16.onnx  half-size copy (used by the converter when present)
  music_symbols.json       classes, input size and preprocessing, read by the converter

Usage (from the YOLOX repo root):
  python /path/to/export_model.py --exp /path/to/yolox_handwritten.py --ckpt YOLOX_outputs/music_tiny/best_ckpt.pth --out export/
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classes import CLASSES, SPACE_PX, TILE_H, TILE_W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="export")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
onnx_path = out / "music_symbols.onnx"

subprocess.run([sys.executable, "tools/export_onnx.py", "-f", args.exp, "-c", args.ckpt,
                "--output-name", str(onnx_path), "--decode_in_inference", "-o", "17"], check=True)

try:
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(str(onnx_path))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(out / "music_symbols.fp16.onnx"))
except Exception as e:  # fp16 is optional
    print("fp16 conversion skipped:", e)

meta = {
    "classes": CLASSES,
    "input": {"width": TILE_W, "height": TILE_H, "channels": 3, "layout": "NCHW", "range": "0-255", "order": "BGR",
              "pad_value": 114, "space_px": SPACE_PX, "staff_top_spaces": 4.5},
    "output": "decoded: cx, cy, w, h, objectness, class scores",
    "license": "Trained on MUSCIMA++/CVC-MUSCIMA (CC BY-NC-SA 4.0) and Petaluma synthetic data: non-commercial, share-alike.",
}
(out / "music_symbols.json").write_text(json.dumps(meta, indent=1))
for f in sorted(out.iterdir()):
    print(f"{f.name}: {os.path.getsize(f) / 1e6:.1f} MB")
