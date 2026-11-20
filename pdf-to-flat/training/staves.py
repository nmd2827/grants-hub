"""Staff finding and strip tiling for scanned pages (Python twin of the converter's logic)."""
import numpy as np
from PIL import Image

from classes import SPACE_PX, TILE_H, TILE_W, TILE_STRIDE


def load_page(path):
    """Open a scan as grayscale with dark ink on a light background."""
    gray = Image.open(path).convert("L")
    if np.asarray(gray.resize((64, 64))).mean() < 128:
        gray = Image.eval(gray, lambda v: 255 - v)
    return gray


def binarize(gray):
    a = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    total, s = a.size, np.dot(np.arange(256), hist)
    wb = sb = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        wb += hist[t]
        if wb == 0 or wb == total:
            continue
        sb += t * hist[t]
        mb, mf = sb / wb, (s - sb) / (total - wb)
        v = wb * (total - wb) * (mb - mf) ** 2
        if v > best:
            best, thr = v, t
    return a <= thr


def find_staves(ink):
    """Return [(top_y, bottom_y, space)] for five-line staves, top to bottom."""
    h, w = ink.shape
    # long horizontal runs per row
    runs = np.zeros(h)
    for y in range(h):
        row = ink[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        L = ends - starts
        runs[y] = L[L > w * 0.04].sum()
    thr = max(runs.max() * 0.3, w * 0.15)
    lines, y = [], 0
    while y < h:
        if runs[y] < thr:
            y += 1
            continue
        y0 = y
        while y < h and runs[y] >= thr:
            y += 1
        lines.append((y0 + y - 1) / 2)
    staves, i = [], 0
    while i + 4 < len(lines):
        g = lines[i:i + 5]
        gaps = np.diff(g)
        if gaps.max() / max(gaps.min(), 1) < 1.35:
            staves.append((g[0], g[4], (g[4] - g[0]) / 4))
            i += 5
        else:
            i += 1
    return staves


def strip_tiles(gray_img, top, bottom, space):
    """Yield (tile_image, x_offset_in_strip, scale, y0_in_page) for one staff."""
    y0 = top - 4.5 * space
    y1 = bottom + 4.5 * space
    scale = SPACE_PX / space
    strip = gray_img.crop((0, int(y0), gray_img.width, int(y1)))
    sw, sh = max(1, round(strip.width * scale)), max(1, round(strip.height * scale))
    strip = strip.resize((sw, sh), Image.BILINEAR)
    x = 0
    while True:
        tile = Image.new("L", (TILE_W, TILE_H), 255)
        tile.paste(strip.crop((x, 0, min(x + TILE_W, sw), min(TILE_H, sh))), (0, 0))
        yield tile, x, scale, int(y0)
        if x + TILE_W >= sw:
            break
        x += TILE_STRIDE
