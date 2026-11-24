"""Drive to viewpoints around every slot, save frames and YOLO labels.

Labels come from the simulator and are used ONLY to build the training set:
projection of the known 3D boxes (cartons, blockers, empty bay volumes) of the
rack level being scanned. Cartons are kept if >= 35% unoccluded, empty bays if
>= 70%. The deployed detector (shelfscan_audit) sees camera pixels only.

Labels use each carton's TRUE size and pose so the boxes are tight. That is
ground truth, which is legitimate here (this builds the training set) but is
never available to shelfscan_audit, which projects a nominal bay region.

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
from shelfscan.robot_utils import (CLASSES, Driver, Lift,        # noqa: E402
                                   content_box, load_layout)
from shelfscan.scissor import ScissorLinks                       # noqa: E402

OUT = Path(os.environ.get("SHELFSCAN_OUT", ROOT / "dataset"))
VIEWPOINTS = ["center", "left", "right", "close"]
JITTER = 0.08
VAL_FRAC = 0.15


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
    n = 0

    for sid in layout["order"]:
        cur = layout["slots"][sid]
        for vp in VIEWPOINTS:
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

            def proj(c, sz):
                bb = project_box(pos, rot, c, sz, fov, W, H)
                if bb is None or bb[2] - bb[0] < 8 or bb[3] - bb[1] < 8:
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
                    for c, sz in carton_boxes(s):
                        bb = proj(c, sz)
                        if not in_frame(bb):
                            continue
                        vis, _ = visibility_and_side(bb, blockers)
                        if vis >= 0.35:
                            boxes.append((s["actual"], bb))
                else:
                    c, sz = content_box(s, bay)
                    bb = proj(c, sz)
                    if not in_frame(bb):
                        continue
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
    from shelfscan.robot_utils import run_logged
    run_logged(main, "data_capture")
