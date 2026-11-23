"""Drive to viewpoints around every slot, save frames and YOLO labels.

Labels come from the simulator and are used ONLY to build the training set:
  - visible items / blockers: Webots Recognition (id, model, bbox in pixels)
  - empty slots: projection of the known slot volume, kept only if it is not
    mostly covered by a blocker.
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
VAL_FRAC = 0.15


def main():
    rng = random.Random(int(os.environ.get("SHELFSCAN_SEED", "0")))
    layout = load_layout()
    robot = Supervisor()
    step = int(robot.getBasicTimeStep())
    cam = robot.getDevice("camera")
    cam.enable(step)
    cam.recognitionEnable(step)
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

            objs = cam.getRecognitionObjects()
            boxes, blockers = [], []
            for o in objs:
                px, py = o.getPositionOnImage()
                sw, sh = o.getSizeOnImage()
                b = (px - sw / 2, py - sh / 2, px + sw / 2, py + sh / 2)
                b = (max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3]))
                if b[2] - b[0] < 3 or b[3] - b[1] < 3:
                    continue
                m = o.getModel()
                if m == "occluder":
                    boxes.append(("blocker", b)); blockers.append(b)
                elif m in CLASSES:
                    boxes.append((m, b))

            pos = cam_node.getPosition()
            o9 = cam_node.getOrientation()
            rot = [o9[0:3], o9[3:6], o9[6:9]]
            for s in layout["slots"].values():
                if s["actual"] is not None:
                    continue
                bb = project_box(pos, rot, (s["x"], s["face_y"] - size[1] / 2 - 0.02, s["z"]),
                                 size, fov, W, H)
                if bb is None or bb[0] <= 1 or bb[2] >= W - 1:
                    continue                                   # cut by image edge
                vis, _ = visibility_and_side(bb, blockers)
                if vis >= 0.7:
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
    main()
