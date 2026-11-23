"""Run one audit policy in Webots with the trained YOLO detector and write results.

Env: SHELFSCAN_POLICY (V1_fixed | V3_rescan | V4_active), SHELFSCAN_WEIGHTS,
     SHELFSCAN_LAYOUT, SHELFSCAN_RESULTS (output dir).

Slot locations in the image come from the known map projected with the robot's
pose (as a real robot would with localisation). Slot CONTENT comes only from the
detector. Ground truth in layout.json is used only to score the result.

STATUS: written without access to Webots; first run may need small fixes.
"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np                                               # noqa: E402
from controller import Supervisor                                # noqa: E402
from ultralytics import YOLO                                     # noqa: E402

from shelfscan.geometry import project_box, visibility_and_side  # noqa: E402
from shelfscan.policies import POLICIES                          # noqa: E402
from shelfscan.robot_utils import CLASSES, Driver, load_layout   # noqa: E402
from shelfscan.twin import DigitalTwin                           # noqa: E402
from shelfscan.types import Observation, Status                  # noqa: E402

CONF_MIN = 0.25


def true_status(exp, act):
    if act == exp:
        return Status.OK
    return Status.EMPTY if act is None else Status.MISPLACED


def main():
    policy = os.environ.get("SHELFSCAN_POLICY", "V4_active")
    weights = os.environ.get("SHELFSCAN_WEIGHTS", str(ROOT / "runs/shelfscan/weights/best.pt"))
    res_dir = Path(os.environ.get("SHELFSCAN_RESULTS", ROOT / "results"))
    res_dir.mkdir(parents=True, exist_ok=True)

    layout = load_layout()
    slots = layout["slots"]
    size = layout["slot_size"]
    model = YOLO(weights)

    robot = Supervisor()
    step = int(robot.getBasicTimeStep())
    cam = robot.getDevice("camera")
    cam.enable(step)
    W, H, fov = cam.getWidth(), cam.getHeight(), cam.getFov()
    cam_node = robot.getFromDef("CAM")
    drv = Driver(robot, step)
    twin = DigitalTwin({sid: s["expected"] for sid, s in slots.items()})
    n_frames = 0

    def observe(sid, vp):
        nonlocal n_frames
        s = slots[sid]
        x, y, yaw = s["viewpoints"][vp]
        drv.go_to(x, y, yaw)
        robot.step(step)
        img = np.frombuffer(cam.getImage(), np.uint8).reshape((H, W, 4))[:, :, :3]  # BGR
        det = model.predict(img, conf=CONF_MIN, verbose=False)[0]
        n_frames += 1
        boxes = [(tuple(b.xyxy[0].tolist()), CLASSES[int(b.cls)], float(b.conf))
                 for b in det.boxes]

        o9 = cam_node.getOrientation()
        rot = [o9[0:3], o9[3:6], o9[6:9]]
        sb = project_box(cam_node.getPosition(), rot,
                         (s["x"], s["face_y"] - size[1] / 2 - 0.02, s["z"]), size, fov, W, H)
        if sb is None:
            return Observation(sid, None, 0.0, 0.0, None, vp, robot.getTime())

        def inside(b):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            return sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]

        own = [(b, c, p) for b, c, p in boxes if c != "blocker" and inside(b)]
        label, conf = (None, 0.0)
        if own:
            _, label, conf = max(own, key=lambda t: t[2])
        others = [b for b, c, p in boxes if c == "blocker" or (c != "blocker" and not inside(b))]
        vis, side = visibility_and_side(sb, others)
        return Observation(sid, label, conf, vis, side, vp, robot.getTime())

    t0 = robot.getTime()
    POLICIES[policy](twin, observe, layout["order"])
    elapsed = robot.getTime() - t0

    # ---- scoring against ground truth ----
    real = [sid for sid in layout["order"]
            if true_status(slots[sid]["expected"], slots[sid]["actual"]) != Status.OK]
    caught = sum(1 for sid in real
                 if twin.slots[sid].status == true_status(slots[sid]["expected"], slots[sid]["actual"]))
    false_alarms = sum(1 for sid in layout["order"]
                       if twin.slots[sid].status in (Status.MISPLACED, Status.EMPTY)
                       and true_status(slots[sid]["expected"], slots[sid]["actual"]) == Status.OK)
    out = dict(policy=policy, seed=layout["seed"], n_slots=len(slots),
               true_discrepancies=len(real), caught=caught,
               detection_rate=caught / len(real) if real else 1.0,
               false_alarms=false_alarms,
               unresolved=len(twin.unresolved()), unresolved_rate=len(twin.unresolved()) / len(slots),
               audit_time_s=elapsed, distance_m=drv.distance, frames=n_frames)
    tag = f"{policy}_seed{layout['seed']}"
    (res_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
    twin.write_report(str(res_dir / f"{tag}_report.csv"))
    print(json.dumps(out))
    robot.simulationQuit(0)


if __name__ == "__main__":
    main()
