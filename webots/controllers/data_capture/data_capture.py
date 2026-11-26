"""Drive to viewpoints around every slot, save frames and YOLO labels.

Labels come from the simulator and are used ONLY to build the training set:
projection of the known 3D boxes (cartons, blockers, empty bay volumes) of the
rack level being scanned. Cartons are kept if >= 35% unoccluded, empty bays if
>= 70%. The deployed detector (shelfscan_audit) sees camera pixels only.

Labels use each carton's TRUE size and pose so the boxes are tight. That is
ground truth, which is legitimate here (this builds the training set) but is
never available to shelfscan_audit, which projects a nominal bay region.

In a QR/barcode world (layout["label"] set) and with SHELFSCAN_CLASSES=v3, the
class set becomes [cardboard_box, label, slot_empty, blocker]: every SKU maps to
cardboard_box, and the code plate on the carton face gets its own `label` box,
projected from the same known geometry as the carton itself (make_world's
code_plate is the one definition both sides use).

Run after generating a world: python webots/tools/make_world.py --seed N
Env: SHELFSCAN_OUT (dataset dir), SHELFSCAN_LAYOUT, SHELFSCAN_CLASSES (v2|v3),
SHELFSCAN_SPLIT (force every frame into train or val), SHELFSCAN_VP_KEEP (keep
only this many of the four viewpoints per slot, drawn at random) and
SHELFSCAN_VP_KEEP_EMPTY (... but this many for a slot with no carton).
"""
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webots/tools"))

from controller import Supervisor                               # noqa: E402
from make_world import PLATE_T, code_plate                      # noqa: E402
from shelfscan.geometry import project_box, to_yolo, visibility_and_side  # noqa: E402
from shelfscan.robot_utils import (Driver, Lift,                 # noqa: E402
                                   class_set, content_box, load_layout,
                                   slot_matches, under_root)
from shelfscan.scissor import ScissorLinks                       # noqa: E402

OUT = under_root(os.environ.get("SHELFSCAN_OUT"), ROOT, "dataset")
VIEWPOINTS = ["center", "left", "right", "close"]
# How much of an EMPTY slot must be visible before it is labelled slot_empty.
# A carton is labelled down to 0.35 and a plate down to 0.5, but an empty slot
# used to need 0.7 - so a bay half hidden behind a pallet had its blocker box
# written and its slot_empty box dropped, which is a frame that says the right
# answer for a blocked empty bay is "blocker". That is what v3 learned: on
# seeds 101-110 it got 35/35 unobstructed empty slots right and 1/6 blocked
# ones, calling them blocker at 0.93-0.97. The audit applies its OWN vis>=0.70
# gate before believing an EMPTY, so the detector emitting the box lower down
# costs nothing - the gate still decides. Default unchanged.
EMPTY_VIS = float(os.environ.get("SHELFSCAN_EMPTY_VIS") or 0.7)
JITTER = 0.08
VAL_FRAC = 0.15
CLASSES = class_set()
V3 = CLASSES[0] == "cardboard_box"
BOX_CLS = "cardboard_box" if V3 else None   # None: keep the per-SKU class
LABEL_MIN_PX = 6          # a code plate smaller than this carries no signal
LABEL_VIS = 0.5           # ... and one less than half unoccluded is noise


def carton_boxes(s):
    """True world boxes of a slot's carton(s): (centre, size) each."""
    if not s["actual"]:
        return []
    sx, sy, sz = s["carton_size"]
    cx, cy, cz = s["carton_pos"]
    if s["n_cartons"] == 2:
        off = sx / 2 + 0.02
        return [((cx - off, cy, cz), (sx, sy, sz)),
                ((cx + off, cy, cz), (sx, sy, sz))]
    return [((cx, cy, cz), (sx, sy, sz))]


def label_boxes(s, label):
    """World boxes of the code plates on a slot's carton(s), one per carton and
    in the SAME order as carton_boxes, so a plate can be paired with its box.

    The plate is a child of the carton solid at local (0, dy, dz); the solid is
    yawed by base_yaw + carton_yaw about z, and a z-rotation by `a` sends local
    (0, dy, dz) to (-dy*sin a, dy*cos a, dz). Extent is taken axis-aligned,
    exactly as the carton's own box is."""
    if not label:
        return []
    dy, dz, w, h = code_plate(s["carton_size"], label)
    a = (0.0 if s["facing"] > 0 else math.pi) + s["carton_yaw"]
    ox, oy = -dy * math.sin(a), dy * math.cos(a)
    return [((cx + ox, cy + oy, cz + dz), (w, PLATE_T, h))
            for (cx, cy, cz), _ in carton_boxes(s)]


ONLY = os.environ.get("SHELFSCAN_CAPTURE_ONLY") or ""


def main():
    rng = random.Random(int(os.environ.get("SHELFSCAN_SEED", "0")))
    layout = load_layout()
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
    bay = layout["bay_size"]
    tag = os.environ.get("SHELFSCAN_TAG", f"w{layout['seed']}")
    label = layout.get("label") if V3 else None
    force_split = os.environ.get("SHELFSCAN_SPLIT") or None
    assert force_split in (None, "train", "val"), force_split
    keep = int(os.environ.get("SHELFSCAN_VP_KEEP") or len(VIEWPOINTS))
    # slot_empty is the rare class by a long way - a seed's 30 slots hold only
    # 3-6 empty bays, against ~27 cartons - so subsampling viewpoints hits it
    # hardest. Every viewpoint of an EMPTY slot is therefore kept while filled
    # slots are thinned: it costs a few hundred frames and triples the
    # slot_empty boxes. Those frames are ordinary frames (they still show the
    # neighbouring cartons and blockers), so nothing else is distorted.
    keep_empty = int(os.environ.get("SHELFSCAN_VP_KEEP_EMPTY") or len(VIEWPOINTS))
    n = 0

    order = [sid for sid in layout["order"]
             if slot_matches(layout["slots"][sid], ONLY)]
    for sid in order:
        cur = layout["slots"][sid]
        # A fresh draw per slot, so over a seed all four viewpoints - and so
        # both camera distances and the two lateral angles - stay represented.
        k = keep_empty if not cur["actual"] else keep
        vps = (VIEWPOINTS if k >= len(VIEWPOINTS)
               else sorted(rng.sample(VIEWPOINTS, k), key=VIEWPOINTS.index))
        for vp in vps:
            x, y, yaw = cur["viewpoints"][vp]
            lift.request(cur["lift"])
            drv.go_to(x + rng.uniform(-JITTER, JITTER),
                      y + rng.uniform(-JITTER, JITTER), yaw)
            lift.settle(cur["cam_z"])
            for _ in range(2):
                robot.step(step)
                lift.tick()

            # Labels from the known 3D geometry of the rack level in view only
            # (other levels fall outside the camera's vertical field, and the
            # far rack is behind the robot). Webots Recognition boxes were
            # checked with overlay_check.py and found loose, so are not used.
            pos = cam_node.getPosition()
            o9 = cam_node.getOrientation()
            rot = [o9[0:3], o9[3:6], o9[6:9]]
            level = [s for s in layout["slots"].values()
                     if s["rack"] == cur["rack"] and s["level"] == cur["level"]]

            def proj(c, sz, min_px=8):
                bb = project_box(pos, rot, c, sz, fov, W, H)
                if bb is None or bb[2] - bb[0] < min_px or bb[3] - bb[1] < min_px:
                    return None
                return bb

            def in_frame(bb):
                """Drop boxes cut by the left/right image edge.

                Applied to CARTONS and empty bays only, on purpose. A
                truncated carton may have its SKU print cut off, so its class
                is ambiguous and a label would be noise. A blocker is
                identified by coarse shape, so a truncated one is still
                unambiguously a blocker - and blockers sit 0.20 m off the bay
                centre only 0.70 m from the camera, so any blocker 0.45 m or
                wider ALWAYS overruns the frame at the centre viewpoint.
                Filtering them too starves the class: it cost 12x the blocker
                boxes (64 per seed down to 5) when tried, and leaving large
                unlabelled objects in frame teaches the detector to ignore
                exactly what the occlusion reasoning depends on."""
                return bb is not None and bb[0] > 1 and bb[2] < W - 1

            boxes, blockers = [], []
            for s in level:
                if s["occluder"]:
                    bb = proj(tuple(s["blocker_pos"]), tuple(s["blocker_size"]))
                    if bb:
                        boxes.append(("blocker", bb)); blockers.append(bb)
            for s in level:
                if s["actual"]:
                    plates = label_boxes(s, label)
                    for i, (c, sz) in enumerate(carton_boxes(s)):
                        bb = proj(c, sz)
                        if not in_frame(bb):
                            continue
                        vis, _ = visibility_and_side(bb, blockers)
                        if vis < 0.35:
                            continue
                        boxes.append((BOX_CLS or s["actual"], bb))
                        # The plate rides with its carton: it is only labelled
                        # when the carton it is printed on was kept, so the two
                        # boxes never disagree about what is in view.
                        if not plates:
                            continue
                        lb = proj(*plates[i], min_px=LABEL_MIN_PX)
                        if not in_frame(lb):
                            continue
                        lvis, _ = visibility_and_side(lb, blockers)
                        if lvis >= LABEL_VIS:
                            boxes.append(("label", lb))
                else:
                    c, sz = content_box(s, bay)
                    bb = proj(c, sz)
                    if not in_frame(bb):
                        continue
                    vis, _ = visibility_and_side(bb, blockers)
                    if vis >= EMPTY_VIS:
                        boxes.append(("slot_empty", bb))

            if not boxes:
                continue
            split = force_split or ("val" if rng.random() < VAL_FRAC else "train")
            (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
            name = f"{tag}_{n:05d}"
            cam.saveImage(str(OUT / "images" / split / f"{name}.jpg"), 92)
            with open(OUT / "labels" / split / f"{name}.txt", "w") as f:
                for cls, bb in boxes:
                    cx, cy, w, h = to_yolo(bb, W, H)
                    f.write(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            n += 1
    print(f"saved {n} frames to {OUT}")
    robot.simulationQuit(0)


if __name__ == "__main__":
    from shelfscan.robot_utils import run_logged
    run_logged(main, "data_capture")
