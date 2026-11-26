"""Check a YOLO dataset: per-class box counts, box sizes, and drawn previews.

    python train/preview_labels.py --data datasets_v3 --classes v3 --n 30

Prints per-class box counts and the mean/percentile box size for train and val
(the box size decides imgsz: a class whose boxes average under ~12 px at the
training size has nothing left to learn from), and renders --n random frames
with their boxes drawn into <data>/preview/ so misaligned labels are visible.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shelfscan.robot_utils import class_set                      # noqa: E402

COLOURS = [(255, 64, 64), (64, 200, 255), (120, 255, 120), (255, 200, 60),
           (220, 120, 255), (255, 255, 255)]


def read(lbl):
    out = []
    for line in lbl.read_text().split("\n"):
        p = line.split()
        if len(p) == 5:
            out.append((int(p[0]), *(float(v) for v in p[1:])))
    return out


def stats(data, split, names):
    """(per-class box counter, per-class min-side pixel sizes, n frames)."""
    img_dir, lbl_dir = data / "images" / split, data / "labels" / split
    cnt, px, n = Counter(), {c: [] for c in names}, 0
    if not lbl_dir.exists():
        return cnt, px, n, []
    from PIL import Image
    pairs = []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = img_dir / f"{lbl.stem}.jpg"
        if not img.exists():
            continue
        n += 1
        W, H = Image.open(img).size
        boxes = read(lbl)
        pairs.append((img, lbl))
        for ci, _, _, w, h in boxes:
            cnt[names[ci]] += 1
            px[names[ci]].append(min(w * W, h * H))
    return cnt, px, n, pairs


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def report(data, names, imgsz):
    tot = Counter()
    allpairs = {}
    for split in ("train", "val"):
        cnt, px, n, pairs = stats(data, split, names)
        allpairs[split] = pairs
        print(f"\n{split}: {n} frames, {sum(cnt.values())} boxes")
        print(f"  {'class':<14}{'boxes':>7}  "
              f"{'mean px':>8}{'p10':>7}{'min':>7}  (min side, at source "
              f"resolution / at imgsz {imgsz})")
        for c in names:
            v = px[c]
            m = sum(v) / len(v) if v else float("nan")
            # frames are 640 wide; at imgsz the box shrinks by imgsz/640
            k = imgsz / 640.0
            print(f"  {c:<14}{cnt[c]:>7}  {m:>8.1f}{pct(v, 0.1):>7.1f}"
                  f"{(min(v) if v else float('nan')):>7.1f}   -> {m * k:5.1f} px")
            tot[c] += cnt[c]
    return tot, allpairs


def draw(data, names, pairs, n, seed=0):
    from PIL import Image, ImageDraw
    out = data / "preview"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    pick = random.Random(seed).sample(pairs, min(n, len(pairs)))
    for img, lbl in pick:
        im = Image.open(img).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for ci, cx, cy, w, h in read(lbl):
            x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
            x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
            col = COLOURS[ci % len(COLOURS)]
            d.rectangle([x0, y0, x1, y1], outline=col, width=2)
            d.text((x0 + 3, max(0, y0 - 11)), names[ci], fill=col)
        im.save(out / f"{img.stem}.jpg", quality=90)
    print(f"\nwrote {len(pick)} previews to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets_v3")
    ap.add_argument("--classes", choices=["v2", "v3"], default="v3")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--min-boxes", type=int, default=100,
                    help="warn if any class has fewer boxes than this")
    a = ap.parse_args()
    data = Path(a.data).resolve()
    names = class_set(a.classes)
    tot, pairs = report(data, names, a.imgsz)
    pool = pairs["train"] + pairs["val"]
    if pool:
        draw(data, names, pool, a.n)
    thin = [c for c in names if tot[c] < a.min_boxes]
    print("\nTOTAL " + "  ".join(f"{c}={tot[c]}" for c in names))
    if thin:
        print(f"WARNING: under {a.min_boxes} boxes: {', '.join(thin)}")
    return 1 if thin else 0


if __name__ == "__main__":
    sys.exit(main())
