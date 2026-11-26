"""Camera-inset overlay for the demo recording (HUD_DISPLAY=1).

What the robot's camera sees, with a box drawn round each label region the
decoder actually tried and the payload it got, pushed to a Webots Display so
it shows live beside the 3D view.

It is a READ-ONLY view, in the same sense `hud.py` is. Nothing here detects or
decodes anything: `draw` is handed the detector boxes and the decoded codes
that `observe()` has ALREADY computed for its own decision, and it only
chooses colours and text. That is deliberate - re-running the detector or the
readers for the benefit of a recording would both cost a second pass and let
the picture disagree with the decision it is illustrating.

cv2 and numpy only, no Webots: `push` takes any object exposing Webots'
imageNew/imagePaste/imageDelete, so all of this is unit-testable.
"""
import cv2
import numpy as np

from .decode import crop_targets
from .hud import sku_label

# BGR, because that is what the camera frame is in. Chosen for a recording
# that is watched once, at speed: the carton is mid-brown and the plate is
# white, so every one of these has to survive against both.
SCAN = (255, 255, 0)       # cyan - a label region found, not yet decoded
GREEN = (0, 255, 0)        # a code was read out of this box
FAIL = (0, 60, 255)        # red-orange - the decoder tried here and failed

# Kept as a name because the audit's own language calls this amber, and an
# earlier version of this module drew it; it is the same state as FAIL.
AMBER = FAIL

SCAN_TEXT = "SCANNING..."
FAIL_TEXT = "DECODE FAIL"

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55          # ~12 px caps on a 640x480 frame
FONT_THICK = 2
BOX_THICK = 3
# A dark keyline drawn just outside the coloured box. Cyan on a white plate
# and green on pale cardboard both wash out otherwise, and the recording is
# the one place this is ever seen.
EDGE = (20, 20, 20)
EDGE_THICK = 1
TEXT_PAD = 6               # gap between a box edge and its caption
CHIP_PAD = 3               # dark plate drawn behind the caption
MAX_TEXT = 22              # a raw payload is longer than the inset is wide


def point_in(pt, box):
    """Is (x, y) inside the box (x0, y0, x1, y1)?"""
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def code_in(box, codes):
    """Payload of the first decoded code whose centre lies in `box`, or None.

    Matching by centre rather than by index because the two decode paths
    report different things: a crop read returns the centre of the box it was
    given, while the whole-frame fallback returns where it actually found the
    code. Containment is the one test that works for both."""
    for text, c in codes:
        if point_in(c, box):
            return text
    return None


def caption(text, sku_of=None):
    """The short string drawn next to a box.

    `sku_of` is the controller's payload -> SKU id function, so the SKU
    vocabulary stays in one place; without it the raw payload is shown,
    clipped, which is what a dev run wants to see."""
    if not text:
        return FAIL_TEXT
    if sku_of is not None:
        sku = sku_of(text)
        return sku_label(sku) if sku else FAIL_TEXT
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT - 1] + "…"


def overlay_items(boxes, codes=(), sku_of=None, pending=False):
    """[(box, colour, caption)] - everything `draw` should put on the frame.

    One entry per box the decoder tried, in `crop_targets` order, so the
    picture shows exactly the regions that were offered to the readers: every
    detected `label` plate, plus any carton that had no plate of its own and
    was therefore decoded whole. Green where a payload came back, amber where
    one did not.

    A code that matches none of those boxes still gets an entry, as a
    zero-size box at its own centre: that is the whole-frame fallback finding
    a code the detector never boxed, and a recording that dropped it would
    show a blank frame for a slot that was in fact read.

    `pending` is the in-between state: the detector has boxed a plate and the
    readers have not answered yet, so every target is cyan and says SCANNING.
    It is a DIFFERENT picture of the same frame, never a different decision -
    see the note on paint_pending in the controller for why it is opt-in."""
    items, used = [], []
    for box, _cls in crop_targets(list(boxes)):
        if pending:
            items.append((tuple(box), SCAN, SCAN_TEXT))
            continue
        text = code_in(box, codes)
        if text is not None:
            used.append(text)
        items.append((tuple(box), GREEN if text else FAIL,
                      caption(text, sku_of)))
    if pending:
        return items
    for text, c in codes:
        if text in used:
            continue
        items.append(((c[0], c[1], c[0], c[1]), GREEN, caption(text, sku_of)))
    return items


def draw(img, boxes, codes=(), sku_of=None, pending=False):
    """Copy of `img` with the label boxes and their payloads drawn on.

    Returns a new array; `img` is the frame the detector ran on and must not
    be touched. Nothing here reads a threshold or a decision - the colour
    comes from whether a payload was already in `codes`."""
    out = np.ascontiguousarray(img.copy())
    h, w = out.shape[:2]
    for box, colour, text in overlay_items(boxes, codes, sku_of, pending):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        if x1 > x0 and y1 > y0:
            # Dark keyline OUTSIDE the coloured stroke, so the box keeps its
            # edge against a white plate as well as against cardboard.
            k = BOX_THICK // 2 + EDGE_THICK
            cv2.rectangle(out, (x0 - k, y0 - k), (x1 + k, y1 + k),
                          EDGE, EDGE_THICK)
            cv2.rectangle(out, (x0, y0), (x1, y1), colour, BOX_THICK)
        _caption(out, text, colour, x0, y0, w, h)
    return out


def _caption(out, text, colour, x0, y0, w, h):
    """One caption, on a dark chip so it reads over any carton."""
    (tw, th), base = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    # Above the box by preference; inside its top edge when the box is already
    # against the top of the frame, so a caption is never clipped away and
    # never lands outside the inset.
    ty = y0 - TEXT_PAD
    if ty - th - CHIP_PAD < 0:
        ty = min(h - TEXT_PAD, y0 + th + TEXT_PAD + CHIP_PAD)
    tx = max(CHIP_PAD, min(x0, w - tw - CHIP_PAD))
    cv2.rectangle(out,
                  (tx - CHIP_PAD, ty - th - CHIP_PAD),
                  (tx + tw + CHIP_PAD, ty + base + CHIP_PAD), EDGE, -1)
    cv2.putText(out, text, (tx, ty), FONT, FONT_SCALE, colour,
                FONT_THICK, cv2.LINE_AA)


def push(display, img, fmt):
    """Paste a BGR frame onto a Webots Display.

    `fmt` is the device's BGRA constant (controller.Display.BGRA), passed in
    rather than imported so this module stays free of Webots. The image
    reference is deleted straight away: Webots keeps every one that is not,
    and this runs once per frame for a whole audit."""
    h, w = img.shape[:2]
    bgra = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
    ref = display.imageNew(bgra.tobytes(), fmt, w, h)
    try:
        display.imagePaste(ref, 0, 0, False)
    finally:
        display.imageDelete(ref)


def show(display, fmt, img, boxes, codes=(), sku_of=None, pending=False):
    """draw + push, the single call the controller makes per frame."""
    push(display, draw(img, boxes, codes, sku_of, pending), fmt)
