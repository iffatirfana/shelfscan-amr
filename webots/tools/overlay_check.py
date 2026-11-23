"""Draw the YOLO labels over the saved images so you can SEE if labels line up.

    python webots/tools/overlay_check.py dataset/images/train dataset/labels/train out_overlay 20

Run this on the first dataset you capture. If boxes are offset or mirrored, the
camera-frame assumption in shelfscan/geometry.py (x fwd, y left, z up) is wrong
for your Webots version: fix geometry.py before training.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shelfscan.robot_utils import CLASSES  # noqa: E402

COL = {"sku_A": "red", "sku_B": "lime", "sku_C": "cyan", "sku_D": "yellow",
       "slot_empty": "magenta", "blocker": "white"}


def main(img_dir, lbl_dir, out_dir, n=20):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(img_dir).glob("*.jpg"))[: int(n)]:
        im = Image.open(p).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        lp = Path(lbl_dir) / (p.stem + ".txt")
        for line in lp.read_text().splitlines() if lp.exists() else []:
            c, cx, cy, w, h = line.split()
            c = CLASSES[int(c)]
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=COL[c], width=2)
            d.text((cx - w / 2 + 2, cy - h / 2 + 2), c, fill=COL[c])
        im.save(out / p.name)
    print(f"wrote overlays to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
