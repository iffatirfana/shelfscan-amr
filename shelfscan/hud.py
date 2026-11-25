"""On-screen HUD for the demo recording, drawn with Robot.setLabel.

Pure text/colour logic plus a thin renderer that takes ANY object exposing
Webots' `setLabel(id, text, x, y, size, colour, transparency, font)` - so this
module imports no Webots and the whole of it is unit-testable.

The HUD is a READ-ONLY view of the audit: it is handed what the controller has
already decided and never feeds anything back. It is created only when HUD=1,
so with the variable unset not one line of this file runs.

SKU naming: the slot map calls the four SKUs sku_A..sku_D, and the Code128
payload already numbers them 1..4 (make_world.SKU_DIGIT). The HUD shows that
number - sku_C is drawn as "SKU_3" - because it reads better at video size.
"""
from .types import EMPTY_CLASS, Status

# ---------------------------------------------------------------- colours ----
WHITE = 0xFFFFFF
GREY = 0xB0BEC5
ACTION = 0xFFD54F
STATUS_COLOUR = {
    Status.OK: 0x00C853,            # green
    Status.MISPLACED: 0xD50000,     # red
    Status.EMPTY: 0x2962FF,         # blue
    Status.UNVERIFIED: 0xFFAB00,    # amber
}

# Where each line sits in the 3D view: (label id, x, y, size, colour).
# Ids are fixed so a redraw replaces a line instead of stacking a new one.
SLOT, DET, DECODE, TWIN, STATE, NEXT, TALLY = range(7)
LAYOUT = {
    SLOT:   (0.02, 0.015, 0.055, WHITE),
    DET:    (0.02, 0.060, 0.045, GREY),
    DECODE: (0.02, 0.098, 0.045, GREY),
    TWIN:   (0.02, 0.136, 0.045, GREY),
    STATE:  (0.02, 0.180, 0.070, WHITE),
    NEXT:   (0.02, 0.870, 0.095, ACTION),
    TALLY:  (0.02, 0.950, 0.048, WHITE),
}

# Order the detector classes are listed in, so the line does not reshuffle
# between frames. Anything not named here is appended, sorted, after them.
CLASS_ORDER = ("cardboard_box", "label", "slot_empty", "blocker",
               "sku_A", "sku_B", "sku_C", "sku_D")

LABEL_NAME = {"qr": "QR", "barcode": "Code128"}


# ------------------------------------------------------------- text pieces ---
def sku_label(sku):
    """sku_C -> 'SKU_3'. Anything else is passed through unchanged."""
    if not sku:
        return "-"
    if sku == EMPTY_CLASS:
        return "EMPTY"
    if sku.startswith("sku_") and len(sku) == 5:
        i = "ABCD".find(sku[4])
        if i >= 0:
            return f"SKU_{i + 1}"
    return sku


def slot_line(slot, index, total):
    """'Side A | L1 | Bay 3        slot 7/30'."""
    return (f"Side {slot['rack']} | L{slot['level']} | Bay {slot['bay']}"
            f"        slot {index}/{total}")


def detector_line(boxes):
    """'Detector  cardboard_box x2 0.91 | label x2 0.88 | blocker x1 0.74'.

    `boxes` is the controller's [(xyxy, class, confidence)] for this frame, so
    what the HUD shows is literally what the policy was given."""
    best, n = {}, {}
    for _, cls, conf in boxes:
        best[cls] = max(best.get(cls, 0.0), conf)
        n[cls] = n.get(cls, 0) + 1
    if not best:
        return "Detector  no boxes"
    known = [c for c in CLASS_ORDER if c in best]
    rest = sorted(c for c in best if c not in CLASS_ORDER)
    parts = [f"{c} x{n[c]} {best[c]:.2f}" for c in known + rest]
    return "Detector  " + " | ".join(parts)


def decode_line(label_type, text=None, sku=None):
    """'QR -> SKU_3', 'Code128 -> SKU_2' or 'DECODE FAIL'.

    With no label mode the world carries printed patterns and there is nothing
    to decode, so the line says so rather than claiming a failure."""
    name = LABEL_NAME.get(label_type)
    if name is None:
        return "Decode  -  (printed pattern world)"
    if sku:
        return f"{name} -> {sku_label(sku)}"
    if text:
        return f"{name} -> DECODE FAIL  ({text})"
    return f"{name} -> DECODE FAIL"


def twin_line(expected, decoded):
    """'Twin  expected SKU_1  |  read SKU_3'."""
    exp = "(empty)" if expected is None else sku_label(expected)
    return f"Twin  expected {exp}  |  read {sku_label(decoded)}"


def status_entry(status):
    """(text, colour) for the state line."""
    st = Status(status)
    return st.value, STATUS_COLOUR[st]


def action_line(action):
    return f"NEXT  {action}"


def tally_line(counts):
    """'OK 12   MISPLACED 2   EMPTY 1   UNVERIFIED 3'."""
    return "   ".join(f"{s.value} {counts.get(s.value, 0)}" for s in Status)


def next_action(prev, nxt):
    """What the robot is about to do, from the slot it just left to the next.

    `prev` and `nxt` are slot dicts (rack, level, bay); `nxt` is None once the
    scan is over. Checked side first, then level, then bay, because that is the
    order the scan itself moves in: a side change always implies the others."""
    if nxt is None:
        return "Scan complete"
    if prev is None:
        return f"Moving to Bay {nxt['bay']}"
    if prev["rack"] != nxt["rack"]:
        return f"Switching to Side {nxt['rack']}"
    if prev["level"] != nxt["level"]:
        return f"Lifting to L{nxt['level']}"
    if prev["bay"] != nxt["bay"]:
        return f"Moving to Bay {nxt['bay']}"
    return "Re-look from new viewpoint"


# ------------------------------------------------------------------ render ---
class Hud:
    """Draws the HUD and holds the simulation briefly so it can be read.

    `robot` only has to provide setLabel(). `hold` is called once per decided
    slot and is the ONLY thing here that touches the simulation clock - which
    is why the HUD is never enabled for an evaluation run.
    """

    def __init__(self, robot, total, label_type=None, hold_steps=20,
                 step_fn=None, font="Arial"):
        self.robot = robot
        self.total = total
        self.label_type = label_type
        self.hold_steps = hold_steps
        self.step_fn = step_fn
        self.font = font
        self.seen = {}            # slot_id -> 1-based scan index
        self.prev = None          # slot dict of the previous observation
        self.slot = None
        self.slot_id = None
        self.counts = {}

    # -- drawing ----------------------------------------------------------
    def _put(self, lid, text, colour=None):
        x, y, size, default = LAYOUT[lid]
        self.robot.setLabel(lid, text, x, y, size,
                            default if colour is None else colour, 0.0,
                            self.font)

    def begin(self, slot_id, slot):
        """Called as the robot sets off for a slot, before it moves."""
        if slot_id not in self.seen:
            self.seen[slot_id] = len(self.seen) + 1
        self.slot_id, self.slot = slot_id, slot
        self._put(NEXT, action_line(next_action(self.prev, slot)))
        self._put(SLOT, slot_line(slot, self.seen[slot_id], self.total))
        self.prev = slot

    def detection(self, boxes, text=None, sku=None):
        """Called with the detector output and the decode for this frame."""
        self._put(DET, detector_line(boxes))
        self._put(DECODE, decode_line(self.label_type, text, sku))
        self._put(TWIN, twin_line(self.slot["expected"] if self.slot else None,
                                  sku))

    def status(self, status, counts):
        """Called once the twin has classified the slot; then holds."""
        text, colour = status_entry(status)
        self.counts = counts
        self._put(STATE, text, colour)
        self._put(TALLY, tally_line(counts))
        self.hold()

    def finish(self, counts):
        self._put(NEXT, action_line(next_action(self.prev, None)))
        self._put(TALLY, tally_line(counts))
        self.hold()

    # -- the only call that costs simulation time -------------------------
    def hold(self):
        for _ in range(self.hold_steps):
            if self.step_fn is not None:
                self.step_fn()


def attach(twin, hud):
    """Make `twin` redraw the HUD whenever it classifies the observed slot.

    Wrapping the twin keeps shelfscan.policies - which the offline harness
    shares - free of any HUD code. The wrapper forwards every call and its
    return value untouched, so the audit cannot tell it is there; it only
    reads the status afterwards. One frame can also update OTHER slots from
    codes it happened to read, and those must not redraw the panel or spend a
    hold, so it redraws for the observed slot alone."""
    inner = twin.update

    def update(obs, evidence_path=None):
        status = inner(obs, evidence_path)
        if obs.slot_id == hud.slot_id:
            hud.status(status, twin.summary())
        return status

    twin.update = update
    return twin
