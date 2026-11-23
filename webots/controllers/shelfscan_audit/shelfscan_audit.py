"""Run one audit policy in Webots with the trained YOLO detector and write results.

Env: SHELFSCAN_POLICY (V1_fixed | V3_rescan | V4_active), SHELFSCAN_WEIGHTS,
     SHELFSCAN_LAYOUT, SHELFSCAN_RESULTS (output dir).

Slot locations in the image come from the known map projected with the robot's
pose (as a real robot would with localisation). Slot CONTENT comes only from the
detector. Ground truth in layout.json is used only to score the result.

Verified to run in Webots R2025a (Windows). Writes <policy>_seed<N>.json and
<policy>_seed<N>_slots.csv to SHELFSCAN_RESULTS.
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
IMGSZ = int(os.environ.get("SHELFSCAN_IMGSZ", "320"))   # must match training


def true_status(exp, act):
    if act == exp:
        return Status.OK
    return Status.EMPTY if act is None else Status.MISPLACED


def main():
    policy = os.environ.get("SHELFSCAN_POLICY", "V4_active")
    weights = os.environ.get("SHELFSCAN_WEIGHTS", str(ROOT / "models/shelfscan_yolov8n.pt"))
    res_dir = Path(os.environ.get("SHELFSCAN_RESULTS", ROOT / "results"))
    res_dir.mkdir(parents=True, exist_ok=True)

    layout = load_layout()
    slots = layout["slots"]
    size = layout["slot_size"]
    model = YOLO(weights)

    robot = Supervisor()
    import controller as _c; _c._shelfscan_robot = robot
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
        img = np.ascontiguousarray(np.frombuffer(cam.getImage(), np.uint8).reshape((H, W, 4))[:, :, :3])  # BGR
        det = model.predict(img, conf=CONF_MIN, imgsz=IMGSZ, verbose=False)[0]
        n_frames += 1
        boxes = [(tuple(b.xyxy[0].tolist()), CLASSES[int(b.cls)], float(b.conf))
                 for b in det.boxes]

        o9 = cam_node.getOrientation()
        rot = [o9[0:3], o9[3:6], o9[6:9]]
        sb = project_box(cam_node.getPosition(), rot,
                         (s["x"], s["face_y"] - size[1] / 2 - 0.02, s["z"]), size, fov, W, H)
        if sb is None:
            print(f"obs {sid} vp={vp} slot not in view", flush=True)
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
        px, py, pyaw = drv.pose()
        print(f"obs {sid} vp={vp} pose=({px:.2f},{py:.2f},{pyaw:.2f}) slot_box={[round(v) for v in sb]} "
              f"own={[(c, round(p, 2)) for _, c, p in own]} n_det={len(boxes)} vis={vis:.2f} side={side}", flush=True)
        return Observation(sid, label, conf, vis, side, vp, robot.getTime())

    t0 = robot.getTime()
    POLICIES[policy](twin, observe, layout["order"])
    elapsed = robot.getTime() - t0

    # ---- scoring against ground truth (layout.json is used ONLY here) ----
    order = layout["order"]
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
               frames=n_frames, weights=str(weights))
    tag = f"{policy}_seed{layout['seed']}"
    (res_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
    with open(res_dir / f"{tag}_slots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot_id", "expected", "actual", "occluder", "true_status", "final_status",
                    "observed_label", "confidence", "observations", "last_viewpoint"])
        for sid in order:
            r = twin.slots[sid]
            w.writerow([sid, slots[sid]["expected"] or "", slots[sid]["actual"] or "",
                        slots[sid]["occluder"] or "", truth[sid].value, r.status.value,
                        r.observed_label or "", round(r.confidence, 3), r.n_observations,
                        r.last_viewpoint])
    print(json.dumps(out))
    robot.simulationQuit(0)


if __name__ == "__main__":
    from shelfscan.robot_utils import run_logged
    run_logged(main, "shelfscan_audit")
