"""Run one audit policy in Webots with the trained YOLO detector and write results.

Env: SHELFSCAN_POLICY (V1_fixed | V3_rescan | V4_active), SHELFSCAN_WEIGHTS,
     SHELFSCAN_LAYOUT, SHELFSCAN_RESULTS (output dir, default results_v2),
     SHELFSCAN_CLASSES (v2 | v3).

DETECTOR v3 (SHELFSCAN_CLASSES=v3, with models_v3 weights): the 4-class model
of the QR world - cardboard_box, label, slot_empty, blocker. It has no SKU
class at all, so it CANNOT name a SKU even by accident; only a decoded payload
can. Having a class for the plate turns code reading from a search into a
crop: each detected `label` box is padded, upscaled 3x and sharpened before
decoding, and a cardboard_box with no label box of its own is decoded as the
fallback (shelfscan/decode.py). Everything downstream is unchanged - a payload
is still matched to a slot by projecting the map, slot_empty still needs
conf >= 0.60 and visibility >= 0.70, a blocker still drops visibility and
sends the slot to V4, and a slot with no readable code still ends UNVERIFIED.
With SHELFSCAN_CLASSES unset none of that runs and v2 reproduces exactly.

LABELS (MVP, SHELFSCAN_LABEL=qr|barcode, world built with --label): every
frame is also read for machine-readable codes - QR with
cv2.QRCodeDetector().detectAndDecodeMulti, Code128 with zxing-cpp, because
pyzbar's bundled libzbar DLL will not load on this machine and OpenCV's
BarcodeDetector decodes EAN/UPC only, not Code128. Each decoded code is matched to
a slot by projecting that slot's nominal content region into the image and
testing which region holds the code's centre, so the match uses the map and the
robot's pose, never ground truth. A matched code decides the slot outright (OK
if the SKU is the expected one, MISPLACED if not). A slot with no readable code
stays UNVERIFIED: in label mode the detector's SKU classes are ignored, because
it was trained on printed patterns and only guesses once the cartons carry
codes. Its slot_empty class is still used, so the EMPTY rule and both
thresholds (0.60 / 0.70) are untouched, and a code can never produce EMPTY.
Plain printed-pattern mode is completely unchanged.

Scan order: one shelf side is scanned COMPLETELY - every level from the top
down, the whole length of each level, including any V4 re-verification detours
- before the robot crosses the aisle to the other side. The sides are never
interleaved. The camera is raised to the level being scanned by the lift.

Slot locations in the image come from the known map projected with the robot's
pose (as a real robot would with localisation), using a nominal content region
derived from the bay geometry only - never from the true carton size, which is
ground truth. Slot CONTENT comes only from the detector. Ground truth in
layout.json is used only to score the result.

HUD=1 (demo recording only) draws an on-screen overlay with Robot.setLabel -
slot, detector boxes, decode, twin expectation, state, next action and a
running tally - and holds the simulation for a few steps at each decided slot
so the text is readable in a recording. It is a read-only view: it is handed
what has already been decided and feeds nothing back, and the hold happens
AFTER the frame has been captured and classified, so it cannot change what the
detector sees either. With HUD unset nothing here runs and the run is
unchanged; experiments/run_batch.py clears HUD for every eval job so an
exported HUD=1 cannot leak into a measured run.

HUD_DISPLAY=1 (demo recording only) additionally paints the camera frame onto
an "inset" Display device, with a box round each label region the decoder
tried - green where a payload came back, amber where none did - and the
decoded SKU drawn beside it (shelfscan/inset.py). The device exists only in a
world built by make_world.py --display, so with the flag off the world is
byte-identical and the controller never looks for it. A Display is a sink:
nothing reads it back, and the picture is drawn from the frame, boxes and
codes observe() has already computed, so it cannot change detection, decoding
or the audit.

SHELFSCAN_NBV_LOG=<path> (dev only, MVP) additionally writes one CSV row per
CANDIDATE viewpoint at every next-best-view decision, with the features a
learned picker could use and whether V4 chose that candidate. The hook runs
after choose_viewpoint has already returned, so logging cannot change what V4
does - verified by re-running seed 101 against results_v2.

Targets Webots R2025a (Windows). Writes <policy>_seed<N>.json and
<policy>_seed<N>_slots.csv to SHELFSCAN_RESULTS.
"""
import csv
import json
import math
import os
import sys
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2                                                      # noqa: E402
import numpy as np                                               # noqa: E402
from controller import Display, Supervisor                       # noqa: E402
from ultralytics import YOLO                                     # noqa: E402

from shelfscan.geometry import project_box, visibility_and_side  # noqa: E402
from shelfscan.hud import Hud, attach as hud_attach              # noqa: E402
from shelfscan import inset as INS                                # noqa: E402
from shelfscan.policies import POLICIES                          # noqa: E402
from shelfscan import decode as DEC                              # noqa: E402
from shelfscan.robot_utils import (Driver, Lift, class_set,       # noqa: E402
                                   content_box, load_layout,
                                   results_dir)
from shelfscan.scissor import ScissorLinks                       # noqa: E402
from shelfscan.twin import DigitalTwin                           # noqa: E402
from shelfscan.types import EMPTY_CLASS, Observation, Status     # noqa: E402

CONF_MIN = 0.25

# Which detector is loaded. SHELFSCAN_CLASSES=v3 selects the 4-class model of
# the QR world (cardboard_box, label, slot_empty, blocker) and with it the
# crop-guided decode path below. Unset means the original 6-class v2 model and
# the original whole-frame decode - not one line of the v3 path runs, so the
# committed v2 results reproduce exactly.
CLASSES = class_set()
V3 = CLASSES[0] == "cardboard_box"
# The SKU vocabulary of the WORLD, which is not the detector's class set: v3
# has no SKU classes at all. code_sku() reads payloads, so it must name SKUs,
# never detector classes - indexing the v3 list here would turn payload digit
# 1 into "cardboard_box" and reject every QR.
SKUS = class_set("v2")[:4]
# Inference size must be the size the loaded model was trained at: both are
# 320 (train/train_yolo.py --imgsz 320), kept per-model so they can diverge.
TRAIN_IMGSZ = {False: 320, True: 320}
IMGSZ = int(os.environ.get("SHELFSCAN_IMGSZ") or TRAIN_IMGSZ[V3])

# Demo overlay. Off unless HUD is exactly "1", so an empty HUD= (which is what
# run_batch exports for eval jobs) can never switch it on.
# Diagnostic: log EVERY raw box the detector returned, before `own` is
# filtered. Without it the log shows only the classes that survived the
# filter, so in v3 a slot the detector called cardboard_box and a slot it
# saw nothing in both print as own=[] and look like the same failure.
# Off by default, and it only prints - no decision reads it.
DEBUG_BOXES = os.environ.get("SHELFSCAN_DEBUG_BOXES") == "1"
HUD = os.environ.get("HUD") == "1"
# HUD_DISPLAY=1 paints the annotated camera frame onto the "inset" Display
# device, which make_world.py --display adds to the robot. Independent of
# HUD because the two are decided at different times: the Display has to be
# IN the world, so it is a world-generation choice, while the text overlay is
# a controller one. A world built without the device simply has no inset, and
# the run says so once instead of failing.
HUD_DISPLAY = os.environ.get("HUD_DISPLAY") == "1"
# HUD_SCAN_STEPS>0 paints a cyan "SCANNING..." frame to the inset after the
# detector has boxed the plates and BEFORE the readers run, then holds that
# many steps so a viewer can see it. There is otherwise no such moment to
# film: detection and decoding both finish inside one observe() call, on one
# captured frame, and the Display only reaches the screen on a step - so a
# pending frame painted without a hold is overwritten before it is ever
# drawn. Holding costs SIMULATED TIME, exactly as the HUD hold does, which
# is why it defaults to 0: at 0 nothing is painted, nothing is stepped and
# audit_time_s is unchanged. Decoding reads the frame captured earlier, so
# even with a hold no decision can differ - only the clock.
HUD_SCAN_STEPS = int(os.environ.get("HUD_SCAN_STEPS", "0"))
HUD_HOLD_STEPS = int(os.environ.get("HUD_HOLD_STEPS", "20"))
# HUD_SHOTS=<dir> saves the 3D view, overlay and all, at each decided slot.
# Needs rendering on (run_webots.py --no-minimize, no --no-rendering).
HUD_SHOTS = os.environ.get("HUD_SHOTS") or ""
HUD_MAX_SHOTS = int(os.environ.get("HUD_MAX_SHOTS", "14"))

LABEL = os.environ.get("SHELFSCAN_LABEL") or (
    "qr" if os.environ.get("SHELFSCAN_QR") == "1" else "")
# Both readers miss codes on the native 640x480 frame that they read fine when
# the frame is resampled 2x first (QR on seed 101: 8/11 -> 9/11 frames), for
# about 55 ms a frame. The upscale adds no information; it just gives the
# decoders more samples per module to work with.
CODE_UPSCALE = 2


def code_sku(text):
    """SKU id out of a payload, or None.

    Two payload forms carry the same pair: the QR spells it out as
    "SKU=<id>;INV=<invoice>", while Code128 has to fit in a tenth of the
    modules so it carries "<sku digit><seed><slot>" as six digits."""
    text = (text or "").strip()
    if text.isdigit() and len(text) == 6:
        i = int(text[0]) - 1
        return SKUS[i] if 0 <= i < 4 else None
    for part in text.split(";"):
        k, _, v = part.partition("=")
        if k.strip().upper() == "SKU" and v.strip() in SKUS:
            return v.strip()
    return None


def _scaled(img, k):
    if k == 1:
        return img
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)


def _read_qr(img):
    """QR at a fixed 2x: detectAndDecodeMulti misses codes on the native frame
    that detectAndDecode reads fine, and 2x recovers them (seed 101 sample:
    8/11 -> 9/11 frames) for about 55 ms."""
    big = _scaled(img, CODE_UPSCALE)
    ok, infos, pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(big)
    if not ok or pts is None:
        return []
    out = []
    for text, quad in zip(infos, pts):
        if text:
            c = np.mean(np.asarray(quad, dtype=float), axis=0) / CODE_UPSCALE
            out.append((text, (float(c[0]), float(c[1]))))
    return out


def _read_barcode(img):
    """Code128 through zxing-cpp, as a short cascade.

    A rendered barcode that a crop decodes perfectly is often not FOUND in the
    full frame: the default local-average binarizer loses the thin bars against
    the cardboard. Measured on the seed 101 sample frames, locating it costs a
    cascade: default 0/11, formats=Code128 0/11, FixedThreshold 7/11, and this
    cascade 9/11 at about 13 ms a frame - the same hit rate as the QR path."""
    import zxingcpp

    fmt = zxingcpp.BarcodeFormat.Code128
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for k in (1, 2, 3):
        g = _scaled(gray, k)
        for binz in (zxingcpp.Binarizer.FixedThreshold,
                     zxingcpp.Binarizer.LocalAverage):
            found = zxingcpp.read_barcodes(g, formats=fmt, binarizer=binz)
            if not found:
                continue
            out = []
            for b in found:
                if not b.text:
                    continue
                q = b.position
                xs = [q.top_left.x, q.top_right.x, q.bottom_right.x,
                      q.bottom_left.x]
                ys = [q.top_left.y, q.top_right.y, q.bottom_right.y,
                      q.bottom_left.y]
                out.append((b.text, (sum(xs) / 4.0 / k, sum(ys) / 4.0 / k)))
            if out:
                return out
    return []


def read_codes(img, boxes=None):
    """[(payload, (cx, cy))] for every code in the frame, in frame pixels.

    v2 searches the whole frame. v3 has a detector class for the plate, so it
    decodes the detected boxes instead - every `label` box, then any
    cardboard_box with no label box of its own - each padded, sharpened and
    swept over the same 3 scales x 2 binarizers v2 uses, but over a crop that
    is mostly code. Either way the payload is placed by WHERE IT SITS in the
    image and matched to a slot by the projected map, never by ground truth,
    and either way the detector supplies no SKU: only a decoded payload can.
    """
    if not LABEL:
        return []
    if V3:
        # ... and when no crop reads, the v2 whole-frame sweep is the fallback,
        # so v3 can only lose a code to a frame where BOTH approaches fail.
        return DEC.read_codes(img, boxes or [], LABEL, fallback=_read_frame)
    return _read_frame(img)


def _read_frame(img):
    """The v2 whole-frame reader. Unchanged, and still the only path in v2."""
    return _read_qr(img) if LABEL == "qr" else _read_barcode(img)


def true_status(exp, act):
    if act == exp:
        return Status.OK
    return Status.EMPTY if act is None else Status.MISPLACED


def main():
    # ---- what did this process ACTUALLY get? ------------------------------
    # HUD and HUD_DISPLAY are read the same way (os.environ.get(...) == "1"),
    # so when one works and the other does not the difference is in what
    # reached the process, not in how it is read. Printing the raw values -
    # repr, so "1 " or "" is visible - plus the script actually executing and
    # its mtime, settles both "is the env arriving" and "is this the patched
    # file or a stale copy" from the log alone, without another guess.
    import datetime as _dt
    _me = pathlib.Path(__file__).resolve()
    _mt = _dt.datetime.fromtimestamp(_me.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"env: script={_me} mtime={_mt}", flush=True)
    print("env: " + "  ".join(
        f"{k}={os.environ.get(k)!r}"
        for k in ("HUD", "HUD_DISPLAY", "SHELFSCAN_TAG", "SHELFSCAN_RESULTS",
                  "SHELFSCAN_LABEL")), flush=True)

    policy = os.environ.get("SHELFSCAN_POLICY", "V4_active")
    weights = os.environ.get("SHELFSCAN_WEIGHTS", str(ROOT / "models/shelfscan_yolov8n.pt"))
    res_dir = results_dir(os.environ.get("SHELFSCAN_RESULTS"), ROOT)
    res_dir.mkdir(parents=True, exist_ok=True)

    layout = load_layout()
    slots = layout["slots"]
    bay = layout["bay_size"]
    model = YOLO(weights)

    robot = Supervisor()
    import controller as _c; _c._shelfscan_robot = robot
    step = int(robot.getBasicTimeStep())
    cam = robot.getDevice("camera")
    cam.enable(step)
    W, H, fov = cam.getWidth(), cam.getHeight(), cam.getFov()
    cam_node = robot.getFromDef("CAM")
    drv = Driver(robot, step)
    links = ScissorLinks(robot)
    lift = Lift(robot, step, links=links)
    drv.on_step = lift.tick          # the scissor folds while the base drives
    twin = DigitalTwin({sid: s["expected"] for sid, s in slots.items()})
    n_frames = 0
    qr_read = {}          # slot_id -> payload, for the report only
    qr_frames = 0

    order = layout["order"]
    sides = list(dict.fromkeys(slots[sid]["rack"] for sid in order))

    # ---- demo overlay (HUD=1) ---------------------------------------------
    # hud.attach wraps twin.update rather than editing shelfscan.policies, so
    # the policy code the offline harness shares stays untouched.
    hud = None
    if HUD:
        def hud_step():
            robot.step(step)
            lift.tick()

        shot_fn = None
        if HUD_SHOTS:
            shot_dir = Path(HUD_SHOTS)
            shot_dir.mkdir(parents=True, exist_ok=True)
            n_shot = {"n": 0}

            def shot_fn(tag):
                if n_shot["n"] >= HUD_MAX_SHOTS:
                    return
                n_shot["n"] += 1
                f = shot_dir / f"{n_shot['n']:02d}_{tag}.png"
                robot.exportImage(str(f), 100)
                print(f"hud: shot {f}", flush=True)

        hud = Hud(robot, total=len(order), label_type=LABEL or None,
                  hold_steps=HUD_HOLD_STEPS, step_fn=hud_step, shot_fn=shot_fn)
        hud_attach(twin, hud)

    # ---- camera inset (HUD_DISPLAY=1) -------------------------------------
    # A Display is a sink: the controller writes to it and nothing reads it
    # back, so painting it cannot reach the detector, the decoders or the
    # twin. getDevice returns None when the world was built without
    # --display, which is the common case and not an error.
    inset = robot.getDevice("inset") if HUD_DISPLAY else None
    crop_disp = robot.getDevice("crop") if HUD_DISPLAY else None
    dash_disp = robot.getDevice("dashboard") if HUD_DISPLAY else None
    if HUD_DISPLAY and inset is None:
        print("hud: HUD_DISPLAY=1 but this world has no 'inset' Display - "
              "rebuild it with make_world.py --display", flush=True)

    # The opposite mistake, and the one that LOOKS like a broken overlay: the
    # world was built with --display, so Webots draws an inset panel, but
    # HUD_DISPLAY is unset, so nothing ever pastes to it and it stays the
    # black a Display is born as. Nothing in the log contradicted that, which
    # is how it reads as "the inset is painting black" rather than "the inset
    # was never switched on". Say so, and paint the reason onto the panel.
    # Devices are listed rather than looked up by name so a world WITHOUT the
    # device - every measured run - triggers no Webots "device not found"
    # warning on a path that did not exist before.
    if not HUD_DISPLAY and any(
            robot.getDeviceByIndex(i).getName() == "inset"
            for i in range(robot.getNumberOfDevices())):
        _idle = robot.getDevice("inset")
        print("hud: this world has an 'inset' Display but HUD_DISPLAY is not "
              "set, so the panel will stay blank. Relaunch with HUD_DISPLAY=1 "
              "to paint the annotated camera frame onto it.", flush=True)
        try:
            INS.push(_idle, INS.idle_card([
                "INSET IDLE",
                "HUD_DISPLAY is not set for this run,",
                "so no camera frame is being painted here.",
                "Relaunch with HUD_DISPLAY=1 to enable it.",
            ]), Display.BGRA)
        except Exception as exc:                       # recording only
            print(f"hud: idle card paste failed: {exc!r}", flush=True)

    # ---- is the inset actually being written to, and with what? -----------
    # A permanently black overlay looks identical from outside to one that is
    # never painted, so say so in the log the first time each sink is pasted,
    # with the statistics that tell a black CAMERA FRAME from a black OVERLAY:
    # mean/min/max over the image that was handed to the Display. A paste that
    # raises is reported too rather than swallowed - this is a recording-only
    # sink, so it must never take the audit down, but a silent failure here is
    # exactly the bug this reports.
    _painted = {}

    def paint(display, img, what):
        """INS.push with a one-time report per sink. Returns True on success."""
        try:
            INS.push(display, img, Display.BGRA)
        except Exception as exc:                       # recording only
            if _painted.get(what) != "failed":
                _painted[what] = "failed"
                import traceback
                print(f"hud: {what} paste FAILED: {exc!r}", flush=True)
                traceback.print_exc()
            return False
        if what not in _painted:
            _painted[what] = "ok"
            print(f"hud: {what} first paste OK - {img.shape[1]}x{img.shape[0]} "
                  f"mean={img.mean():.1f} min={img.min()} max={img.max()}"
                  f"{' <<< ALL-BLACK IMAGE' if int(img.max()) == 0 else ''}",
                  flush=True)
        return True

    drive_banner = {"text": "Starting patrol..."}
    if HUD_DISPLAY and (inset is not None or dash_disp is not None or crop_disp is not None):
        step_counter = {"n": 0}
        _prev_on_step = drv.on_step

        def _display_on_step():
            _prev_on_step()
            step_counter["n"] += 1
            if inset is not None and step_counter["n"] % 8 == 1:
                raw = cam.getImage()
                if raw is not None:
                    frame = np.ascontiguousarray(
                        np.frombuffer(raw, np.uint8).reshape((H, W, 4))[:, :, :3])
                    paint(inset, INS.draw_detections(
                        frame, [], banner=drive_banner["text"]), "inset/drive")

        drv.on_step = _display_on_step
        if dash_disp is not None:
            paint(dash_disp, INS.draw_dashboard(
                slots, twin.slots, action="Initializing..."), "dashboard")
            _orig_twin_update = twin.update

            def _dash_twin_update(obs):
                st = _orig_twin_update(obs)
                # The twin's answer is what the audit runs on, so it is
                # computed and kept BEFORE the dashboard is touched and
                # returned whatever the drawing does. A recording-only sink
                # must not be able to end a six-minute run: redrawing used
                # to raise AttributeError here on the first decided slot
                # (obs.sku - Observation carries `label`), which killed the
                # audit at slot 1 of 30 with the overlay half painted.
                try:
                    paint(dash_disp, INS.draw_dashboard(
                        slots, twin.slots, slot_id=obs.slot_id,
                        decoded_sku=obs.label, status=st,
                        action="Audited"), "dashboard")
                except Exception as exc:
                    if _painted.get("dashboard/draw") != "failed":
                        _painted["dashboard/draw"] = "failed"
                        import traceback
                        print(f"hud: dashboard redraw FAILED: {exc!r} - the "
                              f"audit continues without it", flush=True)
                        traceback.print_exc()
                return st

            twin.update = _dash_twin_update
        if crop_disp is not None:
            paint(crop_disp, INS.draw_crop(
                None, [], (), code_sku, title=f"CUT {('CODE128' if LABEL == 'barcode' else LABEL.upper()) if LABEL else 'LABEL'}"), "crop")

    # ---- next-best-view decision logging (dev only) -----------------------
    nbv_path = os.environ.get("SHELFSCAN_NBV_LOG")
    nbv_rows = []
    last_level = {"v": None}

    def cand_rows(rec, tried, cands):
        """One dict per candidate viewpoint: the features of this decision.

        The single place candidate features are built, so what V5 scores at
        run time is exactly what the table was trained on."""
        s = slots[rec.slot_id]
        px, py, _ = drv.pose()
        out = []
        for vp in cands:
            vx, vy, _ = s["viewpoints"][vp]
            out.append(dict(
                seed=layout["seed"], slot_id=rec.slot_id, candidate=vp,
                n_tried=len(tried), try_index=rec.n_observations,
                uncertainty=round(1.0 - rec.confidence, 4),
                blocker_present=int(rec.last_occluder_side is not None),
                occluder_side=rec.last_occluder_side or "none",
                visibility=round(rec.last_visibility, 4),
                travel_m=round(math.hypot(vx - px, vy - py), 4),
                level=s["level"],
                level_change=int(last_level["v"] is not None
                                 and last_level["v"] != s["level"]),
                side=s["rack"], bay=s["bay"]))
        return out

    def log_decision(rec, tried, cands, chosen):
        for r in cand_rows(rec, tried, cands):
            r["chosen"] = int(r["candidate"] == chosen)
            nbv_rows.append(r)

    # ---- learned picker (V5) ---------------------------------------------
    nbv_model = os.environ.get("SHELFSCAN_NBV_MODEL")
    picker = None
    if nbv_model and policy == "V5_learned":
        from shelfscan.nbv_model import NBVPicker
        picker = NBVPicker(nbv_model)
        print(f"V5: loaded next-best-view model {nbv_model}", flush=True)

    def pick_viewpoint(rec, tried, cands):
        return picker.pick(cand_rows(rec, tried, cands)) if picker else None

    def ids_of(side, level=None):
        return [sid for sid in order if slots[sid]["rack"] == side
                and (level is None or slots[sid]["level"] == level)]

    def n_decided(ids):
        return sum(1 for sid in ids if twin.slots[sid].status is not Status.UNVERIFIED)

    levels_done, current = set(), {"key": None}

    def finish_level(key):
        """Progress line, printed once, when the scan leaves a level."""
        if key is None or key in levels_done:
            return
        levels_done.add(key)
        side, level = key
        ids = ids_of(side, level)
        print(f"Side {side} level {level} done: {n_decided(ids)}/{len(ids)} "
              f"slots decided", flush=True)

    def observe(sid, vp):
        nonlocal n_frames, qr_frames
        s = slots[sid]
        key = (s["rack"], s["level"])
        if current["key"] != key:
            finish_level(current["key"])
            current["key"] = key
        if hud:                          # drawn BEFORE the robot sets off
            hud.begin(sid, s)
        if HUD_DISPLAY:
            drive_banner["text"] = f"MOVING -> Side {s['rack']} L{s['level']} Bay {s['bay']}"
            if dash_disp is not None:
                paint(dash_disp, INS.draw_dashboard(
                    slots, twin.slots, slot_id=sid,
                    action=f"Moving to Side {s['rack']} L{s['level']} Bay {s['bay']}"),
                    "dashboard")
        x, y, yaw = s["viewpoints"][vp]
        last_level["v"] = s["level"]
        lift.request(s["lift"])          # lift travels while the base drives
        drv.go_to(x, y, yaw)
        lift.settle(s["cam_z"])
        robot.step(step)
        lift.tick()
        img = np.ascontiguousarray(np.frombuffer(cam.getImage(), np.uint8).reshape((H, W, 4))[:, :, :3])  # BGR
        det = model.predict(img, conf=CONF_MIN, imgsz=IMGSZ, verbose=False)[0]
        n_frames += 1
        boxes = [(tuple(b.xyxy[0].tolist()), CLASSES[int(b.cls)], float(b.conf))
                 for b in det.boxes]

        if inset is not None and HUD_SCAN_STEPS:
            # Recording only, and never during an eval: run_batch clears
            # HUD_DISPLAY, and this needs it as well as its own step count.
            paint(inset, INS.draw_detections(
                img, boxes, (), code_sku, pending=True,
                banner=f"YOLO DETECTING | Side {s['rack']} L{s['level']} Bay {s['bay']}"),
                "inset/pending")
            if crop_disp is not None:
                paint(crop_disp, INS.draw_crop(
                    img, boxes, (), code_sku, title=f"CUT {('CODE128' if LABEL == 'barcode' else LABEL.upper()) if LABEL else 'LABEL'}", pending=True),
                    "crop")
            for _ in range(HUD_SCAN_STEPS):
                robot.step(step)
                lift.tick()

        o9 = cam_node.getOrientation()
        rot = [o9[0:3], o9[3:6], o9[6:9]]
        cam_pos = cam_node.getPosition()

        def slot_box(other):
            c, sz = content_box(other, bay)
            return project_box(cam_pos, rot, c, sz, fov, W, H)

        # ---- QR pass: map every code in this frame onto a slot ------------
        qr_here = {}
        codes = []          # also read by the HUD_DISPLAY inset, below
        if LABEL:
            level_ids = [i for i in order if slots[i]["rack"] == s["rack"]
                         and slots[i]["level"] == s["level"]]
            slot_boxes = {i: slot_box(slots[i]) for i in level_ids}
            codes = read_codes(img, boxes)
            if codes:
                qr_frames += 1
            for text, (cx, cy) in codes:
                sku = code_sku(text)
                if sku is None:
                    continue
                for i in level_ids:
                    b = slot_boxes[i]
                    if b and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                        qr_here[i] = (text, sku)
                        qr_read[i] = text
                        break
            # codes for OTHER slots on this level are free information: a code
            # that decodes is unambiguous, so record it rather than discard it
            for i, (text, sku) in qr_here.items():
                if i != sid:
                    twin.update(Observation(i, sku, 1.0, 1.0, None, vp,
                                            robot.getTime()))

        if hud:
            _txt, _sku = qr_here.get(sid, (None, None))
            hud.detection(boxes, _txt, _sku)

        if inset is not None:
            paint(inset, INS.draw_detections(
                img, boxes, codes, code_sku,
                banner=f"YOLO | Side {s['rack']} L{s['level']} Bay {s['bay']}"),
                "inset/decode")
        if crop_disp is not None:
            paint(crop_disp, INS.draw_crop(
                img, boxes, codes, code_sku, title=f"CUT {('CODE128' if LABEL == 'barcode' else LABEL.upper()) if LABEL else 'LABEL'}"), "crop")

        sb = slot_boxes[sid] if LABEL else slot_box(s)
        if sb is None:
            print(f"obs {sid} vp={vp} slot not in view", flush=True)
            return Observation(sid, None, 0.0, 0.0, None, vp, robot.getTime())

        def inside(b):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            return sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]

        if sid in qr_here:
            text, sku = qr_here[sid]
            px, py, pyaw = drv.pose()
            print(f"obs {sid} vp={vp} {LABEL.upper()}={text} -> {sku} "
                  f"pose=({px:.2f},{py:.2f},{pyaw:.2f})", flush=True)
            # a code you can read is a code you can see: full confidence and
            # full visibility, so classify() gives OK or MISPLACED, never EMPTY
            return Observation(sid, sku, 1.0, 1.0, None, vp, robot.getTime())

        own = [(b, c, p) for b, c, p in boxes if c != "blocker" and inside(b)]
        if LABEL or V3:
            # In label mode the SKU comes from a decoded code or not at all.
            # models_v2 was trained on printed SKU patterns; once the cartons
            # carry QR or Code128 instead, its SKU head is guessing, and a
            # guess over 0.60 turns a blocked slot into a false MISPLACED
            # (seed 101, Code128: S1-2-2 called sku_D at 0.614 against a true
            # sku_A). slot_empty is a judgement about shape, not about print,
            # so it is kept and the EMPTY rule and both thresholds are
            # untouched. A slot with no readable code now stays UNVERIFIED.
            #
            # v3 is filtered whether or not label mode is on, because its class
            # set holds no SKU at all: cardboard_box says a carton is there,
            # never which one. Letting it through would have classify() compare
            # "cardboard_box" against the expected SKU and call every slot
            # MISPLACED. The gate is therefore structural in v3, not a policy.
            own = [(b, c, p) for b, c, p in own if c == EMPTY_CLASS]
        label, conf = (None, 0.0)
        if own:
            _, label, conf = max(own, key=lambda t: t[2])
        others = [b for b, c, p in boxes if c == "blocker" or (c != "blocker" and not inside(b))]
        vis, side = visibility_and_side(sb, others)
        px, py, pyaw = drv.pose()
        print(f"obs {sid} vp={vp} pose=({px:.2f},{py:.2f},{pyaw:.2f},z={lift.height():.2f}) "
              f"slot_box={[round(v) for v in sb]} "
              f"own={[(c, round(p, 2)) for _, c, p in own]} n_det={len(boxes)} vis={vis:.2f} side={side}", flush=True)
        if DEBUG_BOXES:
            inner = " ".join(f"{c}:{p:.3f}{'*' if inside(b) else ''}"
                             for b, c, p in sorted(boxes, key=lambda t: -t[2]))
            print(f"raw {sid} vp={vp} conf_min={CONF_MIN} "
                  f"boxes=[{inner}] (* = centre inside the slot box)",
                  flush=True)
        return Observation(sid, label, conf, vis, side, vp, robot.getTime())

    # ---- one shelf side at a time, top level down, never interleaved -------
    t0 = robot.getTime()
    per_side = {}
    for side in sides:
        ids = ids_of(side)
        lv = sorted({slots[sid]["level"] for sid in ids}, reverse=True)
        t_side = robot.getTime()
        print(f"Side {side} started: {len(ids)} slots, levels {lv} top down",
              flush=True)
        if policy == "V5_learned":
            POLICIES[policy](twin, observe, ids, picker=pick_viewpoint,
                             on_decision=log_decision if nbv_path else None)
        elif nbv_path and policy == "V4_active":
            POLICIES[policy](twin, observe, ids, on_decision=log_decision)
        else:
            POLICIES[policy](twin, observe, ids)
        for level in lv:
            finish_level((side, level))
        dec = n_decided(ids)
        print(f"Side {side} complete: {dec} slots decided "
              f"({len(ids) - dec} UNVERIFIED) in "
              f"{robot.getTime() - t_side:.1f} s", flush=True)
        per_side[side] = dict(n_slots=len(ids), decided=dec,
                              time_s=round(robot.getTime() - t_side, 2))
    elapsed = robot.getTime() - t0
    if hud:                              # after `elapsed`: the hold it spends
        hud.finish(twin.summary())       # is not charged to the audit time
    if dash_disp is not None:
        paint(dash_disp, INS.draw_dashboard(
            slots, twin.slots, action="Scan complete"), "dashboard")

    # ---- scoring against ground truth (layout.json is used ONLY here) ----
    truth = {sid: true_status(slots[sid]["expected"], slots[sid]["actual"]) for sid in order}
    final = {sid: twin.slots[sid].status for sid in order}
    real = [sid for sid in order if truth[sid] != Status.OK]
    caught = sum(1 for sid in real if final[sid] == truth[sid])
    false_alarms = sum(1 for sid in order if final[sid] in (Status.MISPLACED, Status.EMPTY)
                       and truth[sid] == Status.OK)
    correct = sum(1 for sid in order if final[sid] == truth[sid])
    unverified = sum(1 for sid in order if final[sid] == Status.UNVERIFIED)
    decided = len(order) - unverified
    not_empty = [sid for sid in order if truth[sid] != Status.EMPTY]
    false_empty = sum(1 for sid in not_empty if final[sid] == Status.EMPTY)
    true_empty = [sid for sid in order if truth[sid] == Status.EMPTY]
    # QR stats. Ground truth is read here, in the scoring section, only - the
    # decisions above were taken from the decoded payload alone.
    qr_with_carton = [sid for sid in order if slots[sid]["actual"]]
    qr_wrong = sorted(sid for sid, txt in qr_read.items()
                      if code_sku(txt) != slots[sid]["actual"])
    out = dict(policy=policy, seed=layout["seed"], n_slots=len(order),
               correct=correct, slot_accuracy=correct / len(order),
               decided=decided,
               decided_accuracy=(correct / decided) if decided else None,
               unverified=unverified, unverified_rate=unverified / len(order),
               n_truly_not_empty=len(not_empty), false_empty=false_empty,
               false_empty_rate=false_empty / len(not_empty) if not_empty else 0.0,
               n_truly_empty=len(true_empty),
               empty_caught=sum(1 for sid in true_empty if final[sid] == Status.EMPTY),
               true_discrepancies=len(real), caught=caught,
               detection_rate=caught / len(real) if real else None,
               false_alarms=false_alarms,
               audit_time_s=round(elapsed, 2), distance_m=round(drv.distance, 2),
               frames=n_frames, weights=str(weights),
               nbv_model=(str(nbv_model) if picker else None),
               nbv_calls=(picker.n_calls if picker else 0),
               nbv_fallbacks=(picker.n_fallback if picker else 0),
               qr=bool(LABEL), label_type=LABEL or None,
               qr_slots_decoded=len(qr_read),
               qr_slots_with_carton=len(qr_with_carton),
               qr_decode_rate=(len(qr_read) / len(qr_with_carton)
                               if qr_with_carton else None),
               qr_frames_with_a_code=qr_frames,
               qr_wrong_payload=len(qr_wrong), qr_wrong_slots=qr_wrong,
               n_sides=layout.get("n_sides"), n_levels=layout.get("n_levels"),
               n_bays=layout.get("n_bays"), scan_order="side_by_side_top_down",
               per_side=per_side)
    tag = f"{policy}_seed{layout['seed']}"
    (res_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
    with open(res_dir / f"{tag}_slots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scan_index", "slot_id", "side", "level", "bay", "size_class",
                    "expected", "actual", "blocker", "occluder", "true_status",
                    "final_status", "observed_label", "confidence", "observations",
                    "last_viewpoint", "qr_text"])
        for i, sid in enumerate(order):
            r, s = twin.slots[sid], slots[sid]
            w.writerow([i, sid, s["rack"], s["level"], s["bay"], s["size_class"],
                        s["expected"] or "", s["actual"] or "",
                        s["blocker_kind"] or "", s["occluder"] or "",
                        truth[sid].value, r.status.value,
                        r.observed_label or "", round(r.confidence, 3),
                        r.n_observations, r.last_viewpoint,
                        qr_read.get(sid, "")])
    if nbv_path and nbv_rows:
        f = Path(nbv_path)
        f.parent.mkdir(parents=True, exist_ok=True)
        new = not f.exists()
        with open(f, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(nbv_rows[0]))
            if new:
                w.writeheader()
            w.writerows(nbv_rows)
        print(f"nbv: wrote {len(nbv_rows)} candidate rows to {f}", flush=True)

    print(json.dumps(out))
    robot.simulationQuit(0)


if __name__ == "__main__":
    from shelfscan.robot_utils import run_logged
    run_logged(main, "shelfscan_audit")
