"""Generate synthetic handwritten-style staff strips with exact bounding boxes.

Glyphs come from Steinberg's Petaluma font (SIL Open Font License, handwritten "Real Book" style).
Each glyph is drawn with random size, rotation, position and stroke-weight jitter. Stems, beams,
barlines, ledger lines, ties, slurs and hairpins are drawn as wobbly pen strokes, and the strip then
gets blur, noise and thresholding so it looks like a binarised scan. The output geometry matches
prepare_muscima.py: TILE_W x TILE_H tiles, staff space SPACE_PX, staff top at 4.5 spaces.

Usage:
  python synth_petaluma.py --font path/to/Petaluma.otf --out data/coco --train 6000 --val 300
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from classes import CLASS_ID, SPACE_PX, TILE_W, TILE_H, coco_categories

SP = SPACE_PX
TOP = 4.5 * SP                     # y of the top staff line
G = {  # SMuFL code points
    "headBlack": 0xE0A4, "headHalf": 0xE0A3, "headWhole": 0xE0A2,
    "gClef": 0xE050, "fClef": 0xE062, "cClef": 0xE05C, "percClef": 0xE069,
    "sharp": 0xE262, "flat": 0xE260, "natural": 0xE261, "doubleSharp": 0xE263,
    "flag8Up": 0xE240, "flag8Down": 0xE241, "flag16Up": 0xE242, "flag16Down": 0xE243,
    "dot": 0xE1E7, "restWhole": 0xE4E3, "restHalf": 0xE4E4, "restQuarter": 0xE4E5,
    "rest8": 0xE4E6, "rest16": 0xE4E7, "timeSig0": 0xE080, "common": 0xE08A, "cut": 0xE08B,
    "tuplet0": 0xE880, "dynP": 0xE520, "dynM": 0xE521, "dynF": 0xE522, "dynS": 0xE524, "dynZ": 0xE525,
}
DYN_WORDS = ["p", "pp", "mp", "mf", "f", "ff", "sf", "sfz", "fz", "fp"]
SHARP_POS = [8, 5, 9, 6, 3, 7, 4]     # treble key signature positions (half-spaces above bottom line)
FLAT_POS = [4, 7, 3, 6, 2, 5, 1]


def y_of(pos):
    return TOP + 4 * SP - pos * SP / 2


class Strip:
    def __init__(self, font_path, rng):
        self.rng = rng
        self.font_path = font_path
        self.img = Image.new("L", (TILE_W, TILE_H), 255)
        self.d = ImageDraw.Draw(self.img)
        self.boxes = []
        self.fonts = {}
        self.pen = rng.uniform(1.6, 3.0)        # stroke width of the "pen"

    def font(self, size):
        size = int(round(size))
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.font_path, size)
        return self.fonts[size]

    def add(self, cls, box):
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(TILE_W, x1), min(TILE_H, y1)
        if x1 - x0 >= 1 and y1 - y0 >= 1 and cls:
            self.boxes.append((CLASS_ID[cls], x0, y0, x1 - x0, y1 - y0))

    def glyph(self, cp, x, y, cls, scale=1.0, rot=None):
        """Draw a SMuFL glyph with its origin at (x, y) (baseline), jittered like handwriting."""
        r = self.rng
        size = 4 * SP * scale * r.uniform(0.92, 1.1)       # SMuFL: 1 em = 4 staff spaces
        f = self.font(size)
        ch = chr(cp)
        l, t, rr, b = f.getbbox(ch, anchor="ls")
        if rr - l <= 0:
            return None
        pad = 6
        tmp = Image.new("L", (int(rr - l) + 2 * pad, int(b - t) + 2 * pad), 0)
        ImageDraw.Draw(tmp).text((pad - l, pad - t), ch, font=f, fill=255, anchor="ls")
        rot = r.uniform(-6, 6) if rot is None else rot
        tmp = tmp.rotate(rot, resample=Image.BILINEAR, expand=True)
        if r.random() < 0.3:
            tmp = tmp.filter(ImageFilter.MaxFilter(3))   # heavier pen
        bbox = tmp.getbbox()
        if not bbox:
            return None
        ox = int(round(x + l - pad - (tmp.width - (rr - l + 2 * pad)) / 2))
        oy = int(round(y + t - pad - (tmp.height - (b - t + 2 * pad)) / 2))
        self.img.paste(0, (ox, oy), tmp)
        box = (ox + bbox[0], oy + bbox[1], ox + bbox[2], oy + bbox[3])
        self.add(cls, box)
        return box

    def stroke(self, pts, width=None, cls=None):
        """A wobbly pen line through pts."""
        r = self.rng
        w = width or self.pen
        jit = [(x + r.uniform(-0.6, 0.6), y + r.uniform(-0.6, 0.6)) for x, y in pts]
        self.d.line(jit, fill=0, width=max(1, int(round(w))), joint="curve")
        xs, ys = [p[0] for p in jit], [p[1] for p in jit]
        box = (min(xs) - w / 2, min(ys) - w / 2, max(xs) + w / 2, max(ys) + w / 2)
        self.add(cls, box)
        return box

    def line(self, x0, y0, x1, y1, width=None, cls=None, wobble=0.8):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 12))
        ph = self.rng.uniform(0, 6.28)
        pts = []
        for i in range(n + 1):
            t = i / n
            off = wobble * math.sin(ph + t * 5)
            nx, ny = -(y1 - y0), (x1 - x0)
            ln = math.hypot(nx, ny) or 1
            pts.append((x0 + (x1 - x0) * t + off * nx / ln, y0 + (y1 - y0) * t + off * ny / ln))
        return self.stroke(pts, width, cls)

    def curve(self, x0, y0, x1, y1, bulge, cls):
        pts = []
        for i in range(21):
            t = i / 20
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t + bulge * 4 * t * (1 - t)
            pts.append((x, y))
        w = self.pen * self.rng.uniform(0.7, 1.1)
        for i in range(20):  # thicker in the middle, like a real tie
            t = (i + 0.5) / 20
            self.d.line([pts[i], pts[i + 1]], fill=0, width=max(1, int(round(w * (0.6 + 0.8 * math.sin(math.pi * t))))))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        self.add(cls, (min(xs) - 1, min(ys) - 1, max(xs) + 1, max(ys) + 1))


def make_strip(font_path, rng):
    s = Strip(font_path, rng)
    r = rng
    # staff lines (printed manuscript paper: straight but slightly uneven)
    x_start = r.choice([0, 0, r.uniform(10, 120)])
    for k in range(5):
        y = TOP + k * SP + r.uniform(-0.4, 0.4)
        s.d.line([(x_start, y), (TILE_W, y + r.uniform(-1.5, 1.5))], fill=0, width=r.choice([2, 2, 3]))
    x = x_start + r.uniform(8, 20)
    clef = r.choice(["gClef", "gClef", "gClef", "fClef", "cClef", "percClef", None])
    if clef:
        y = {"gClef": y_of(2), "fClef": y_of(6), "cClef": y_of(4), "percClef": y_of(4)}[clef]
        b = s.glyph(G[clef], x, y, clef, rot=r.uniform(-4, 4))
        x = (b[2] if b else x + 2 * SP) + r.uniform(6, 14)
        if clef != "percClef" and r.random() < 0.7:
            n = r.randint(0, 4)
            kind = r.choice(["sharp", "flat"])
            for i in range(n):
                p = (SHARP_POS if kind == "sharp" else FLAT_POS)[i]
                b = s.glyph(G[kind], x, y_of(p), kind)
                x = (b[2] if b else x + SP) + r.uniform(1, 5)
        if r.random() < 0.5:
            x += r.uniform(4, 10)
            if r.random() < 0.2:
                b = s.glyph(G[r.choice(["common", "cut"])], x, y_of(4), None)
                s.boxes[-1:] = []
                if b:
                    s.add("timeSigCommon", b)
                x = (b[2] if b else x + 2 * SP) + 10
            else:
                num, den = r.choice([(2, 4), (3, 4), (4, 4), (6, 8), (12, 8), (5, 4), (2, 2), (3, 8)])
                bx = []
                for val, pos in ((num, 6), (den, 2)):
                    xx = x
                    for dch in str(val):
                        b = s.glyph(G["timeSig0"] + int(dch), xx, y_of(pos), f"digit{dch}", rot=r.uniform(-3, 3))
                        if b:
                            bx.append(b); xx = b[2] + 1
                x = max([b[2] for b in bx] + [x + 2 * SP]) + r.uniform(8, 16)

    last_note = None
    while x < TILE_W - 2 * SP:
        roll = r.random()
        if roll < 0.06:                                            # barline
            bx = x + r.uniform(4, 10)
            s.line(bx, TOP - r.uniform(0, 2), bx + r.uniform(-1.5, 1.5), TOP + 4 * SP + r.uniform(0, 2), cls="barline", wobble=0.5)
            if r.random() < 0.1:
                for yy in (y_of(5), y_of(3)):
                    s.glyph(G["dot"], bx - SP * 0.9, yy, "repeatDot")
            x = bx + r.uniform(12, 26)
            last_note = None
            continue
        if roll < 0.16:                                            # rest
            kind = r.choice(["restQuarter", "rest8th", "rest16th", "restHalf", "restWhole"])
            cp = {"restQuarter": G["restQuarter"], "rest8th": G["rest8"], "rest16th": G["rest16"], "restHalf": G["restHalf"], "restWhole": G["restWhole"]}[kind]
            y = y_of(6) if kind == "restWhole" else y_of(4)
            b = s.glyph(cp, x, y, kind)
            x = (b[2] if b else x + SP) + r.uniform(10, 26)
            last_note = None
            continue
        if roll < 0.20:                                            # dynamics below the staff
            word = r.choice(DYN_WORDS)
            xx = x
            for chx in word:
                b = s.glyph(G["dyn" + chx.upper()], xx, y_of(-5) + r.uniform(-4, 6), "dyn" + chx.upper())
                xx = (b[2] if b else xx + SP) - r.uniform(0, 3)
            if r.random() < 0.4:                                   # hairpin after it
                x0, x1 = xx + 8, min(TILE_W - 4, xx + r.uniform(4, 9) * SP)
                yy, op = y_of(-5) - 4, r.uniform(0.35, 0.6) * SP
                cls = r.choice(["crescendo", "diminuendo"])
                if cls == "crescendo":
                    s.line(x0, yy, x1, yy - op, wobble=0.4); s.line(x0, yy, x1, yy + op, wobble=0.4)
                else:
                    s.line(x0, yy - op, x1, yy, wobble=0.4); s.line(x0, yy + op, x1, yy, wobble=0.4)
                s.add(cls, (x0 - 1, yy - op - 1, x1 + 1, yy + op + 1))
            continue
        # a beamed group (2-4 notes) or a single note
        n = r.choice([1, 1, 1, 2, 3, 4]) if roll < 0.85 else 1
        dur = r.choice(["8", "16"]) if n > 1 else r.choice(["q", "q", "h", "w", "8", "16"])
        up = r.random() < 0.5
        heads = []
        for i in range(n):
            pos = r.randint(-4, 12)
            if r.random() < 0.15 and n == 1:                      # grace note before it
                gy = y_of(r.randint(0, 9))
                gb = s.glyph(G["headBlack"], x, gy, "noteheadGrace", scale=0.62)
                if gb:
                    sx = gb[2] - 1
                    s.line(sx, gy, sx, gy - 2.4 * SP, width=max(1, s.pen * 0.7), cls="stem")
                    s.glyph(G["flag8Up"], sx, gy - 2.4 * SP, "flag8th", scale=0.62)
                    if r.random() < 0.6:
                        s.line(sx - 0.5 * SP, gy - 1.2 * SP, sx + 0.7 * SP, gy - 2.0 * SP, width=1, cls=None)
                    x = gb[2] + r.uniform(8, 14)
            if r.random() < 0.18:                                  # accidental
                acc = r.choice(["sharp", "flat", "natural", "sharp", "flat", "doubleSharp"])
                b = s.glyph(G[acc], x, y_of(pos), acc)
                x = (b[2] if b else x + SP) + r.uniform(2, 6)
            head = {"w": G["headWhole"], "h": G["headHalf"]}.get(dur, G["headBlack"])
            cls = {"w": "noteheadWhole", "h": "noteheadHalf"}.get(dur, "noteheadFull")
            hb = s.glyph(head, x, y_of(pos), cls, rot=r.uniform(-8, 8))
            if not hb:
                break
            # ledger lines
            for lp in list(range(-2, pos - 1, -2)) + list(range(10, pos + 1, 2)):
                ly = y_of(lp) + r.uniform(-0.6, 0.6)
                s.line(hb[0] - 0.35 * SP, ly, hb[2] + 0.35 * SP, ly, width=max(1, s.pen * 0.8), cls="ledgerLine", wobble=0.3)
            heads.append((hb, pos))
            if r.random() < 0.12 and dur not in ("8", "16") or (dur in ("q", "h") and r.random() < 0.1):
                dy = y_of(pos) - (SP / 2 if pos % 2 == 0 else 0)
                s.glyph(G["dot"], hb[2] + r.uniform(4, 8), dy, "augmentationDot")
            if last_note and r.random() < 0.12:                    # tie / slur from the previous note
                (pb, ppos) = last_note
                same = ppos == pos
                y0 = (pb[3] + 3) if not up else (pb[1] - 3)
                y1 = (hb[3] + 3) if not up else (hb[1] - 3)
                s.curve(pb[2] + 2, y0, hb[0] - 2, y1, (0.6 if not up else -0.6) * SP * (1 if same else 1.8), "tie" if same else "slur")
            last_note = (hb, pos)
            x = hb[2] + (r.uniform(18, 34) if n > 1 else r.uniform(22, 60))
        if not heads:
            continue
        if dur == "w":
            continue
        # stems (+ flags or beams)
        stem_up = up if n > 1 else heads[0][1] < 4
        ends = []
        for hb, pos in heads:
            sx = hb[2] - 1 if stem_up else hb[0] + 1
            sy = (hb[1] + hb[3]) / 2
            ey = sy - 3.4 * SP if stem_up else sy + 3.4 * SP
            ends.append((sx, sy, ey))
        if n > 1:
            ey = min(e[2] for e in ends) if stem_up else max(e[2] for e in ends)
            slope = r.uniform(-0.08, 0.08)
            x0 = ends[0][0]
            for sx, sy, _ in ends:
                s.line(sx, sy, sx, ey + slope * (sx - x0), cls="stem", wobble=0.3)
            levels = 2 if dur == "16" else 1
            for lv in range(levels):
                off = lv * 0.75 * SP * (1 if stem_up else -1)
                ya, yb = ey + off, ey + slope * (ends[-1][0] - x0) + off
                th = 0.45 * SP
                poly = [(ends[0][0], ya), (ends[-1][0], yb), (ends[-1][0], yb + (th if stem_up else -th)), (ends[0][0], ya + (th if stem_up else -th))]
                poly = [(px + r.uniform(-0.8, 0.8), py + r.uniform(-0.8, 0.8)) for px, py in poly]
                s.d.polygon(poly, fill=0)
                s.add("beam", (min(p[0] for p in poly), min(p[1] for p in poly), max(p[0] for p in poly), max(p[1] for p in poly)))
            if r.random() < 0.15 and n == 3:                       # tuplet number
                tx = (ends[0][0] + ends[-1][0]) / 2 - 0.4 * SP
                ty = (ey - 0.8 * SP) if stem_up else (ey + 1.8 * SP)
                s.glyph(G["tuplet0"] + 3, tx, ty, "digit3")
        else:
            sx, sy, ey = ends[0]
            s.line(sx, sy, sx, ey, cls="stem", wobble=0.3)
            if dur in ("8", "16"):
                cp = G[("flag8" if dur == "8" else "flag16") + ("Up" if stem_up else "Down")]
                s.glyph(cp, sx, ey, "flag8th" if dur == "8" else "flag16th", rot=r.uniform(-4, 4))
    return s


def degrade(img, rng):
    a = np.asarray(img, dtype=np.float32)
    # smooth elastic wobble (max ~1.5 px) so strokes are not perfectly regular
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    amp, per = rng.uniform(0.3, 1.5), rng.uniform(40, 120)
    ph1, ph2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    xs = np.clip(xx + amp * np.sin(yy / per + ph1), 0, w - 1).astype(np.int32)
    ys = np.clip(yy + amp * np.sin(xx / per + ph2), 0, h - 1).astype(np.int32)
    a = a[ys, xs]
    out = Image.fromarray(a.astype(np.uint8))
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    a = np.asarray(out, dtype=np.float32) + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0, 16), (h, w))
    thr = rng.uniform(110, 170)
    a = np.where(a < thr, 0, 255).astype(np.uint8)          # binarised like the converter's input
    return Image.fromarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True)
    ap.add_argument("--out", default="data/coco")
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for split, count, folder in (("train", args.train, "train2017"), ("val", args.val, "val2017")):
        (out / folder).mkdir(parents=True, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": coco_categories()}
        aid = 1
        for i in range(count):
            s = make_strip(args.font, rng)
            img = degrade(s.img, rng)
            name = f"synth_{split}_{i:05d}.png"
            img.convert("RGB").save(out / folder / name)
            coco["images"].append({"id": i + 1, "file_name": name, "width": TILE_W, "height": TILE_H})
            for cid, x, y, w, h in s.boxes:
                coco["annotations"].append({"id": aid, "image_id": i + 1, "category_id": cid, "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)], "area": round(w * h, 1), "iscrowd": 0})
                aid += 1
        (out / "annotations" / f"synth_{split}.json").write_text(json.dumps(coco))
        print(f"synth {split}: {count} tiles, {aid - 1} boxes")


if __name__ == "__main__":
    main()
