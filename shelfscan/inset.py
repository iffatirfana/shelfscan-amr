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

from .decode import crop_targets, pad_box
from .hud import sku_label
from .types import Status

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


IDLE_BG = (30, 32, 38)
IDLE_FG = (210, 216, 226)
IDLE_DIM = (140, 148, 160)


def idle_card(lines, width=640, height=480):
    """The panel to paint when the Display exists but nothing will fill it.

    A Webots Display starts black and stays black until something pastes to
    it, so a world built with --display and run WITHOUT HUD_DISPLAY=1 shows a
    permanently black rectangle that looks exactly like a crashed or broken
    overlay. There is nothing in the log to contradict that reading, because
    the controller simply never touched the device. This card is pasted once
    in that case so the panel states its own reason instead."""
    canvas = np.full((height, width, 3), IDLE_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (70, 76, 88), 2)
    y = height // 2 - (len(lines) * 30) // 2
    for i, text in enumerate(lines):
        scale, col, thick = ((0.72, IDLE_FG, 2) if i == 0
                             else (0.48, IDLE_DIM, 1))
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
        cv2.putText(canvas, text, ((width - tw) // 2, y + th), FONT, scale,
                    col, thick, cv2.LINE_AA)
        y += 30 if i == 0 else 24
    return canvas


# ------------------------------------------- 3-window demo pipeline views ----
BOX_COLOURS = {
    "cardboard_box": (0, 220, 255),   # yellow-cyan
    "label":         (0, 255, 0),     # green
    "blocker":       (0, 140, 255),   # orange
    "slot_empty":    (255, 130, 40),  # blue
}

_STATUS_BGR = {
    Status.OK:         (83, 200, 0),
    Status.MISPLACED:  (40, 40, 220),
    Status.EMPTY:      (255, 98, 41),
    Status.UNVERIFIED: (0, 171, 255),
}


def _banner(out, text, colour=(255, 255, 255)):
    """Top-left dark chip banner for a camera/detector frame."""
    if not text:
        return
    (tw, th), base = cv2.getTextSize(text, FONT, 0.50, 1)
    cv2.rectangle(out, (6, 6), (6 + tw + 12, 6 + th + base + 10), EDGE, -1)
    cv2.putText(out, text, (12, 6 + th + 4), FONT, 0.50, colour, 1, cv2.LINE_AA)


def draw_detections(img, boxes, codes=(), sku_of=None, pending=False, banner=None):
    """Window 1: full camera frame with all detector boxes drawn on it."""
    out = np.ascontiguousarray(img.copy())
    h, w = out.shape[:2]
    used = []
    for box, cls, conf in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        if x1 <= x0 or y1 <= y0:
            continue
        if cls == "label":
            if pending:
                col = SCAN
                tag = f"{cls} {conf:.2f} | {SCAN_TEXT}"
            else:
                txt = code_in(box, codes)
                if txt is not None:
                    used.append(txt)
                    col = GREEN
                    tag = f"{cls} {conf:.2f} | {caption(txt, sku_of)}"
                else:
                    col = FAIL
                    tag = f"{cls} {conf:.2f}"
        else:
            col = BOX_COLOURS.get(cls, (200, 200, 200))
            tag = f"{cls} {conf:.2f}"
        k = BOX_THICK // 2 + EDGE_THICK
        cv2.rectangle(out, (x0 - k, y0 - k), (x1 + k, y1 + k), EDGE, EDGE_THICK)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        _caption(out, tag, col, x0, y0, w, h)
    if not pending:
        for txt, c in codes:
            if txt not in used:
                _caption(out, caption(txt, sku_of), GREEN,
                         int(round(c[0])), int(round(c[1])), w, h)
    _banner(out, banner)
    return out


def draw_crop(img, boxes, codes=(), sku_of=None, title="CUT LABEL",
              pending=False, width=320, height=260):
    """Window 2: the cut label crop passed to the decoder."""
    canvas = np.full((height, width, 3), (24, 26, 32), dtype=np.uint8)
    

    targets = crop_targets(list(boxes))
    chosen_box, chosen_text = None, None
    for b, _cls in targets:
        t = code_in(b, codes)
        if t is not None:
            chosen_box, chosen_text = b, t
            break
    if chosen_box is None and targets:
        chosen_box = targets[0][0]
        chosen_text = code_in(chosen_box, codes)

    if pending and chosen_box is not None:
        badge, badge_col = "SCANNING", SCAN
        sku_str, raw_str = SCAN_TEXT, ""
    elif chosen_text is not None:
        badge, badge_col = "DECODED", GREEN
        sku_str = caption(chosen_text, sku_of)
        raw_str = (chosen_text if len(chosen_text) <= 28
                   else chosen_text[:27] + "...")
    elif chosen_box is not None:
        badge, badge_col = "FAIL", FAIL
        sku_str, raw_str = FAIL_TEXT, "No readable code in crop"
    else:
        badge, badge_col = "NONE", (130, 138, 150)
        sku_str, raw_str = "--", "No label region detected"

    # Header bar
    cv2.rectangle(canvas, (0, 0), (width, 34), (36, 40, 48), -1)
    cv2.putText(canvas, title, (10, 23), FONT, 0.55, (235, 240, 245), 1, cv2.LINE_AA)
    (bw, _), _ = cv2.getTextSize(badge, FONT, 0.45, 1)
    cv2.rectangle(canvas, (width - bw - 20, 6), (width - 8, 28), badge_col, -1)
    cv2.putText(canvas, badge, (width - bw - 14, 22), FONT, 0.45,
                (15, 18, 22), 1, cv2.LINE_AA)

    # Cropped region in the middle
    top_y, bot_y = 40, height - 48
    if chosen_box is not None and img is not None:
        ih, iw = img.shape[:2]
        x0, y0, x1, y1 = pad_box(chosen_box, iw, ih)
        if x1 > x0 and y1 > y0:
            crop = img[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            max_w, max_h = width - 24, bot_y - top_y - 8
            scale = min(max_w / max(1, cw), max_h / max(1, ch))
            nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
            ox = (width - nw) // 2
            oy = top_y + (bot_y - top_y - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized
            cv2.rectangle(canvas, (ox - 2, oy - 2), (ox + nw + 2, oy + nh + 2),
                          badge_col, 2)
    else:
        msg = "Waiting for label..."
        (mw, _), _ = cv2.getTextSize(msg, FONT, 0.48, 1)
        cv2.putText(canvas, msg, ((width - mw) // 2, (top_y + bot_y) // 2),
                    FONT, 0.48, (140, 148, 160), 1, cv2.LINE_AA)

    # Footer bar
    cv2.rectangle(canvas, (0, height - 44), (width, height), (32, 36, 44), -1)
    cv2.putText(canvas, sku_str, (10, height - 24), FONT, 0.55,
                badge_col, 2, cv2.LINE_AA)
    if raw_str:
        cv2.putText(canvas, raw_str, (10, height - 7), FONT, 0.38,
                    (190, 198, 210), 1, cv2.LINE_AA)
    return canvas


def draw_dashboard(slots_meta, twin_slots, slot_id=None, decoded_sku=None,
                   status=None, action=None, width=420, height=340):
    """Window 3: live auditing dashboard with current slot, tally, and rack map."""
    canvas = np.full((height, width, 3), (22, 24, 30), dtype=np.uint8)

    counts = {s.value: 0 for s in Status}
    visited = 0
    for rec in twin_slots.values():
        counts[rec.status.value] = counts.get(rec.status.value, 0) + 1
        if rec.n_observations > 0:
            visited += 1
    total = len(twin_slots)

    # Header
    cv2.rectangle(canvas, (0, 0), (width, 36), (34, 38, 46), -1)
    cv2.putText(canvas, "AUDIT DASHBOARD", (12, 24), FONT, 0.58,
                (240, 244, 248), 2, cv2.LINE_AA)
    prog = f"{visited}/{total} slots"
    (pw, _), _ = cv2.getTextSize(prog, FONT, 0.48, 1)
    cv2.putText(canvas, prog, (width - pw - 12, 23), FONT, 0.48,
                (170, 210, 255), 1, cv2.LINE_AA)

    # Current slot card
    cv2.rectangle(canvas, (10, 44), (width - 10, 146), (30, 34, 42), -1)
    cv2.rectangle(canvas, (10, 44), (width - 10, 146), (55, 62, 74), 1)
    s = slots_meta.get(slot_id) if slot_id and slots_meta else None
    slot_title = (f"Side {s['rack']} | L{s['level']} | Bay {s['bay']}"
                  if s else "Initializing audit...")
    cv2.putText(canvas, slot_title, (20, 68), FONT, 0.56,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, action or "Ready", (20, 89), FONT, 0.42,
                (255, 213, 79), 1, cv2.LINE_AA)

    exp_str = ("(empty)" if (s and s.get("expected") is None)
               else (sku_label(s["expected"]) if s else "--"))
    dec_str = sku_label(decoded_sku) if decoded_sku else "-"
    cv2.putText(canvas, f"Expected: {exp_str}", (20, 114), FONT, 0.44,
                (200, 208, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Decoded:  {dec_str}", (20, 134), FONT, 0.44,
                (200, 208, 220), 1, cv2.LINE_AA)

    # Status pill
    if status is not None:
        st = Status(status)
        st_txt, st_col = st.value, _STATUS_BGR[st]
    else:
        st_txt, st_col = "PENDING", (110, 118, 130)
    (sw, _), _ = cv2.getTextSize(st_txt, FONT, 0.50, 2)
    px0 = max(250, width - sw - 38)
    cv2.rectangle(canvas, (px0, 96), (width - 20, 134), st_col, -1)
    cv2.putText(canvas, st_txt, (px0 + (width - 20 - px0 - sw) // 2, 121),
                FONT, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    # Tally row
    cv2.rectangle(canvas, (10, 154), (width - 10, 186), (28, 31, 38), -1)
    tally_items = [
        (f"OK {counts['OK']}", _STATUS_BGR[Status.OK]),
        (f"MISP {counts['MISPLACED']}", _STATUS_BGR[Status.MISPLACED]),
        (f"EMPTY {counts['EMPTY']}", _STATUS_BGR[Status.EMPTY]),
        (f"UNVER {counts['UNVERIFIED']}", _STATUS_BGR[Status.UNVERIFIED]),
    ]
    tx = 18
    for lbl, col in tally_items:
        cv2.putText(canvas, lbl, (tx, 175), FONT, 0.44, col, 1, cv2.LINE_AA)
        tx += 98

    # 30-slot Rack Map (Side A & Side B, L2..L0 top-to-bottom, Bay 0..4)
    cv2.putText(canvas, "RACK A", (14, 206), FONT, 0.40, (170, 178, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "RACK B", (width // 2 + 10, 206), FONT, 0.40,
                (170, 178, 190), 1, cv2.LINE_AA)
    tile_w, tile_h, gap = 34, 34, 4
    for side_idx in (0, 1):
        ox = 14 if side_idx == 0 else width // 2 + 10
        for lvl in (2, 1, 0):
            row = 2 - lvl
            for bay in range(5):
                sid = f"S{side_idx}-{lvl}-{bay}"
                rx = ox + bay * (tile_w + gap)
                ry = 214 + row * (tile_h + gap)
                rec = twin_slots.get(sid)
                if rec is None or (rec.n_observations == 0
                                   and rec.status is Status.UNVERIFIED):
                    tcol = (52, 58, 68)
                else:
                    tcol = _STATUS_BGR[rec.status]
                cv2.rectangle(canvas, (rx, ry), (rx + tile_w, ry + tile_h),
                              tcol, -1)
                if sid == slot_id:
                    cv2.rectangle(canvas, (rx - 1, ry - 1),
                                  (rx + tile_w + 1, ry + tile_h + 1),
                                  (255, 255, 255), 2)
                cv2.putText(canvas, f"{lvl}.{bay}", (rx + 5, ry + 21), FONT,
                            0.34, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas
