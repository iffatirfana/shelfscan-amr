"""Stage-1 world check: load the world, prove the lift really moves the camera,
and save a few frames so the look can be inspected. Writes nothing to results/.

Env: SHELFSCAN_CHECK_DIR (output dir for the PNGs), SHELFSCAN_LAYOUT.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controller import Supervisor                      # noqa: E402

from shelfscan.robot_utils import Driver, Lift, load_layout  # noqa: E402
from shelfscan.scissor import ScissorLinks             # noqa: E402


def main():
    out = Path(os.environ.get("SHELFSCAN_CHECK_DIR", ROOT / "logs/world_check"))
    out.mkdir(parents=True, exist_ok=True)
    layout = load_layout()
    slots = layout["slots"]

    robot = Supervisor()
    import controller as _c
    _c._shelfscan_robot = robot
    step = int(robot.getBasicTimeStep())
    cam = robot.getDevice("camera")
    cam.enable(step)
    drv = Driver(robot, step)
    links = ScissorLinks(robot)
    hoist = Lift(robot, step, links=links)
    drv.on_step = hoist.tick
    lift = robot.getDevice("lift")
    carriage = robot.getFromDef("LIFT")
    print(f"devices ok: camera {cam.getWidth()}x{cam.getHeight()} "
          f"fov={cam.getFov():.4f}  lift={lift is not None} "
          f"scissor_links={len(links)}", flush=True)
    print(f"layout: {layout['n_slots']} slots, {layout['n_sides']} sides, "
          f"{layout['n_levels']} levels, {layout['n_bays']} bays", flush=True)

    def settle_lift(target, n=400):
        lift.setPosition(target)
        for _ in range(n):
            if robot.step(step) == -1:
                raise SystemExit
            hoist.tick()
        return carriage.getPosition()[2]

    # 1. the lift must really change the camera height
    for i, pos in enumerate(layout["lift_positions"]):
        z = settle_lift(pos)
        want = layout["cam_level_z"][i]
        print(f"lift L{i}: target={pos:.2f} carriage_z={z:.3f} "
              f"want_cam_z={want:.3f} err={abs(z - want):.3f}", flush=True)

    # 2. a frame from the centre viewpoint of a few representative bays
    picks = [sid for sid in layout["order"]
             if slots[sid]["rack"] == "A" and slots[sid]["level"] == 2][:2]
    picks += [sid for sid in layout["order"]
              if slots[sid]["rack"] == "A" and slots[sid]["level"] == 0][:2]
    picks += [sid for sid in layout["order"]
              if slots[sid]["rack"] == "B" and slots[sid]["level"] == 1][:1]
    picks += [sid for sid in layout["order"] if slots[sid]["occluder"]][:2]
    # worst case for label readability: the smallest carton class, and a bay
    # holding two small cartons
    picks += [sid for sid in layout["order"]
              if slots[sid]["size_class"] == "small" and slots[sid]["actual"]][:2]
    picks += [sid for sid in layout["order"] if slots[sid]["n_cartons"] == 2][:2]
    # one bay per SKU, so every print pattern gets eyeballed
    for sku in ("sku_A", "sku_B", "sku_C", "sku_D"):
        picks += [sid for sid in layout["order"]
                  if slots[sid]["actual"] == sku and not slots[sid]["occluder"]][:1]
    for sid in dict.fromkeys(picks):
        s = slots[sid]
        settle_lift(s["lift"], 260)
        x, y, yaw = s["viewpoints"]["center"]
        drv.go_to(x, y, yaw)
        robot.step(step)
        hoist.tick()
        px, py, pyaw = drv.pose()
        f = out / f"{sid}_{s['size_class']}_{s['actual'] or 'empty'}.png"
        cam.saveImage(str(f), 95)
        print(f"frame {sid} L{s['level']} {s['size_class']} "
              f"act={s['actual']} blocker={s['blocker_kind']} "
              f"pose=({px:.2f},{py:.2f},{pyaw:.2f}) cam_z={carriage.getPosition()[2]:.2f} "
              f"-> {f.name}", flush=True)

    print(f"world_check done, distance={drv.distance:.2f} m", flush=True)
    robot.simulationQuit(0)


if __name__ == "__main__":
    from shelfscan.robot_utils import run_logged
    run_logged(main, "world_check")
