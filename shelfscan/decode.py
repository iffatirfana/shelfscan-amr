"""Crop-guided code reading for the v3 detector path.

v2 reads codes out of the WHOLE 640x480 frame: the decoders have to find a
plate that covers a few per cent of it, which is why both readers upscale the
entire frame first (shelfscan_audit._read_qr / _read_barcode).

v3 has a detector class for the plate itself, so the search is already done.
This module takes the detected box, pads it, upscales it and sharpens it, and
hands the decoders a crop that is mostly code - which is what they are good at.
When a carton has no `label` box the carton box is used instead, so a missed
plate costs resolution rather than the whole read.

cv2 is imported at module level (it is a plain pip package, no Webots), and
zxingcpp lazily, so the QR path runs on a machine without it.

Nothing here decides anything: it returns payload text or None. Turning a
payload into a SKU, and a SKU into a status, stays where it was.
"""
import cv2
import numpy as np

LABEL_CLS = "label"
BOX_CLS = "cardboard_box"

PAD_FRAC = 0.10      # grow the detected box by this much of its own size
SCALE = 3            # single-shot upscale, for prep_crop's own callers
MIN_CROP_PX = 10     # a crop smaller than this cannot hold a readable code
MAX_CROP_PX = 1600   # ... and one larger than this only costs time

# Scale cascade, the same shape the v2 full-frame reader uses. A single
# upscale decoded QR as well as the full frame did but lost ground on Code128,
# which has no error correction: measured on seeds 101-110, one scale read
# 209/259 codes against the full frame's 217/259. The cost of a miss is a
# re-look, so trying three sizes before giving up is cheap by comparison.
SCALES = (1, 2, 3)


def centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def inside(box, region):
    """Is the centre of `box` inside `region`?"""
    cx, cy = centre(box)
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def pad_box(box, w, h, pad_frac=PAD_FRAC):
    """Grow a box by pad_frac of its own size, clipped to the frame.

    A QR carries its quiet zone inside the printed plate, so the padding is
    not needed to find one - it is there because the detected box can sit a
    pixel or two inside the true plate edge, and a clipped finder pattern or a
    clipped Code128 stop character costs the whole read."""
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    return (max(0, int(round(x0 - px))), max(0, int(round(y0 - py))),
            min(int(w), int(round(x1 + px))), min(int(h), int(round(y1 + py))))


def sharpen(img):
    """Unsharp mask. The upscale is interpolated, so it adds samples but no
    edge; this puts the edge back, which is what a binarizer keys on."""
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.6, blur, -0.6, 0)


def prep_crop(img, box, pad_frac=PAD_FRAC, scale=SCALE, do_sharpen=True):
    """Padded, upscaled, sharpened crop of `box`, or None if it is too small.

    Returns greyscale: both decoders want grey, and converting once here keeps
    the two paths reading exactly the same pixels."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = pad_box(box, w, h, pad_frac)
    if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
        return None
    crop = img[y0:y1, x0:x1]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = min(scale, MAX_CROP_PX / max(1, max(crop.shape[:2])))
    if k > 1.0:
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    return sharpen(crop) if do_sharpen else crop


def scaled(img, k):
    """Upscale by k, never past MAX_CROP_PX on the long side."""
    if k <= 1:
        return img
    k = min(k, MAX_CROP_PX / max(1, max(img.shape[:2])))
    if k <= 1.0:
        return img
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)


def read_qr_crop(crop):
    """Payload of the QR in a crop, or None.

    detectAndDecode, not the Multi form v2 uses on the full frame: a crop holds
    one code, and the single-code call is both faster and more forgiving."""
    if crop is None:
        return None
    try:
        text, pts, _ = cv2.QRCodeDetector().detectAndDecode(crop)
    except cv2.error:
        return None
    return text or None


def read_code128_crop(crop):
    """Payload of the Code128 in a crop, or None.

    Same binarizer cascade as the full-frame reader: FixedThreshold first
    because a rendered barcode on cardboard defeats the local-average one."""
    if crop is None:
        return None
    try:
        import zxingcpp
    except ImportError:
        return None
    fmt = zxingcpp.BarcodeFormat.Code128
    for binz in (zxingcpp.Binarizer.FixedThreshold,
                 zxingcpp.Binarizer.LocalAverage):
        for found in zxingcpp.read_barcodes(crop, formats=fmt, binarizer=binz):
            if found.text:
                return found.text
    return None


def read_crop(crop, label_type, scales=SCALES):
    """Cascade the reader over `scales`, first payload wins.

    read_code128_crop already walks its two binarizers at each size, so this is
    the same 3-scales x 2-binarizers sweep the v2 full-frame reader does - just
    over a crop that is mostly code instead of over the whole frame."""
    if crop is None:
        return None
    read = read_code128_crop if label_type == "barcode" else read_qr_crop
    for k in scales:
        text = read(scaled(crop, k))
        if text:
            return text
    return None


def decode_box(img, box, label_type, pad_frac=PAD_FRAC, scales=SCALES):
    """Decode one detected box. Payload text, or None.

    The crop is built ONCE at native resolution and the cascade upscales that,
    so every size is resampled from the same pixels rather than from an
    already-interpolated image."""
    return read_crop(prep_crop(img, box, pad_frac, scale=1), label_type, scales)


def crop_targets(boxes, label_cls=LABEL_CLS, box_cls=BOX_CLS):
    """Which detected boxes to decode, best first.

    Every `label` box, most confident first - the plate is what the decoders
    want. Then any `cardboard_box` that holds no label box: the detector found
    a carton but not its plate, so the carton crop is the fallback. A carton
    that DOES have a label box is skipped, because decoding it would only
    re-read, at lower resolution, a plate already tried."""
    labels = [b for b, _ in sorted(((b, p) for b, c, p in boxes
                                    if c == label_cls), key=lambda t: -t[1])]
    cartons = [b for b, _ in sorted(((b, p) for b, c, p in boxes
                                     if c == box_cls), key=lambda t: -t[1])]
    out = [(b, label_cls) for b in labels]
    out += [(b, box_cls) for b in cartons
            if not any(inside(lb, b) for lb in labels)]
    return out


def read_codes(img, boxes, label_type, pad_frac=PAD_FRAC, scales=SCALES,
               fallback=None):
    """[(payload, (cx, cy))] for every code read from a detected box.

    The same shape the full-frame v2 reader returns, so the caller maps a
    payload onto a slot exactly as before - by where it sits in the image,
    using the projected map and the robot's pose, never ground truth.

    `fallback(img) -> [(payload, (cx, cy))]` is the whole-frame reader, used
    only when the crops yield NOTHING. A frame with no readable crop is one
    where the detector found no plate, or where the plate will not read even
    close up; either way the full-frame sweep is the last thing to try, and it
    is exactly what v2 would have done. It is skipped as soon as one crop
    reads, because the crop path is then working and the sweep would only cost
    time - which does mean a frame can still miss a SECOND code the sweep
    would have found."""
    out = []
    for box, _cls in crop_targets(boxes):
        text = decode_box(img, box, label_type, pad_frac, scales)
        if text:
            out.append((text, centre(box)))
    if not out and fallback is not None:
        return list(fallback(img))
    return out


def annotate(img, boxes, codes=(), colours=None):
    """Frame with detector boxes and any decoded text drawn on it (demo only).

    Used for the HUD screenshots; it returns a copy and never touches `img`."""
    out = np.ascontiguousarray(img.copy())
    colours = colours or {}
    for box, cls, conf in boxes:
        c = colours.get(cls, (0, 255, 0))
        p0 = (int(box[0]), int(box[1]))
        p1 = (int(box[2]), int(box[3]))
        cv2.rectangle(out, p0, p1, c, 2)
        cv2.putText(out, f"{cls} {conf:.2f}", (p0[0] + 2, max(10, p0[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    for text, (cx, cy) in codes:
        cv2.putText(out, text, (int(cx) - 40, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1,
                    cv2.LINE_AA)
    return out
