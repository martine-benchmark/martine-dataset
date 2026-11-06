"""Build the README banner of martine-dataset by re-gridding the project-page thumbnail sheet.

Usage:
    python make_banner.py --miniature MINIATURE.jpg --out assets/banner.png

Default: the 28 objects on a 10x3 grid (2480x750).

MINIATURE.jpg (martine-benchmark.github.io/assets/images/) is a regular 7x4 grid of the 28 objects on black. Its
cells are cut as they are and laid out again, in reading order, on a grid with --cols columns; an incomplete last
row is centred. Nothing is added: same cells, same black background. Requires numpy and Pillow.
"""

import argparse
import math
import os

import numpy as np
from PIL import Image

SRC_COLS, SRC_ROWS = 7, 4
NAMES = ["warrior_head", "vase", "can", "knife", "moka_pot", "surgery_saw", "candle",
         "rock", "colander", "petanque_ball", "sneakers", "spikes_ball", "paw_pad", "wood",
         "pillow", "surgery_scissors", "glass", "lego", "jam_pot", "straw_hat", "hammer_head",
         "whisk", "coin", "bowl", "plate", "plant", "spine", "grid"]


def cells(path):
    img = np.array(Image.open(path).convert("RGB"))
    img[:2, :] = 0  # the sheet has a stray grey line along part of its top and left edges
    img[:, :2] = 0
    h, w = img.shape[:2]
    xs = [round(c * w / SRC_COLS) for c in range(SRC_COLS + 1)]
    ys = [round(r * h / SRC_ROWS) for r in range(SRC_ROWS + 1)]
    cw, ch = min(np.diff(xs)), min(np.diff(ys))
    out = {}
    for r in range(SRC_ROWS):
        for c in range(SRC_COLS):
            out[NAMES[r * SRC_COLS + c]] = img[ys[r]:ys[r] + ch, xs[c]:xs[c] + cw]
    return out, cw, ch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--miniature", required=True, help="MINIATURE.jpg of the project page")
    ap.add_argument("--cols", type=int, default=10, help="columns of the new grid")
    ap.add_argument("--drop", nargs="*", default=[], choices=NAMES, help="objects left out")
    ap.add_argument("--out", default="banner.png")
    a = ap.parse_args()

    cut, cw, ch = cells(a.miniature)
    keep = [n for n in NAMES if n not in a.drop]
    rows = math.ceil(len(keep) / a.cols)
    sheet = np.zeros((rows * ch, a.cols * cw, 3), np.uint8)
    for r in range(rows):
        line = keep[r * a.cols:(r + 1) * a.cols]
        x0 = (a.cols - len(line)) * cw // 2
        for k, name in enumerate(line):
            sheet[r * ch:(r + 1) * ch, x0 + k * cw:x0 + (k + 1) * cw] = cut[name]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(sheet).save(a.out, optimize=True)
    print(f"wrote {a.out} ({sheet.shape[1]}x{sheet.shape[0]}, {len(keep)} objects, {a.cols}x{rows})")


if __name__ == "__main__":
    main()
