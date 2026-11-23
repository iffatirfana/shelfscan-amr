"""Drive to viewpoints around every slot, save frames and YOLO labels.

Labels come from the simulator and are used ONLY to build the training set:
projection of the known 3D boxes (items, blockers, empty slot volumes) of the
row being scanned. Items are kept if >= 35% unoccluded, empty slots if >= 70%.
The deployed detector (shelfscan_audit) sees camera pixels only.

STATUS: written without access to Webots; first run may need small fixes.
Run after generating a world: python webots/tools/make_world.py --seed N
Env: SHELFSCAN_OUT (dataset dir), SHELFSCAN_LAYOUT.
"""
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controller import Supervisor                               # noqa: E402
from shelfscan.geometry import project_box, to_yolo, visibility_and_side  # noqa: E402
from shelfscan.robot_utils import CLASSES, Driver, load_layout   # noqa: E402

OUT = Path(os.environ.get("SHELFSCAN_OUT", ROOT / "dataset"))
VIEWPOINTS = ["center", "left", "right", "close"]
JITTER = 0.08
OCC_Z = 0.30 + 0.04 / 2 + 0.175     # must match make_world.py
OCC_SIZE = (0.25, 0.25, 0.35)
VAL_FRAC = 0.15


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
    size = layout["slot_size"]
    tag = os.environ.get("SHELFSCAN_TAG", f"w{layout['seed']}")
    n = 0

    for sid in layout["order"]:
        for vp in VIEWPOINTS:
            x, y, yaw = layout["slots"][sid]["viewpoints"][vp]
            drv.go_to(x + rng.uniform(-JITTER, JITTER), y + rng.uniform(-JITTER, JITTER), yaw)
            for _ in range(2):
                robot.step(step)

            # Labels from the known 3D geometry of the CURRENT row only (other
            # rows are hidden behind it). Webots Recognition boxes were checked
            # with overlay_check.py and found to be loose, so they are not used.
            pos = cam_node.getPosition()
            o9 = cam_node.getOrientation()
            rot = [o9[0:3], o9[3:6], o9[6:9]]
            row_y = layout["slots"][sid]["y"]
            row = [s for s in layout["slots"].values() if abs(s["y"] - row_y) < 1e-6]

            def proj(c, sz):
                bb = project_box(pos, rot, c, sz, fov, W, H)
                if bb is None or bb[2] - bb[0] < 8 or bb[3] - bb[1] < 8:
                    return None
                return bb

            boxes, blockers = [], []
            for s in row:
                if s["occluder"]:
                    sign = 1 if s["occluder"] == "left" else -1
                    bb = proj((s["x"] + sign * 0.14, s["face_y"] + 0.30, OCC_Z), OCC_SIZE)
                    if bb:
                        boxes.append(("blocker", bb)); blockers.append(bb)
            for s in row:
                bb = proj((s["x"], s["face_y"] - size[1] / 2 - 0.02, s["z"]), size)
                if bb is None or bb[0] <= 1 or bb[2] >= W - 1:
                    continue                                   # cut by image edge
                vis, _ = visibility_and_side(bb, blockers)
                if s["actual"] is not None and vis >= 0.35:
                    boxes.append((s["actual"], bb))
                elif s["actual"] is None and vis >= 0.7:
                    boxes.append(("slot_empty", bb))

            if not boxes:
                continue
            split = "val" if rng.random() < VAL_FRAC else "train"
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
