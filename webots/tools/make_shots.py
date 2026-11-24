"""Render the documentation figures from the CURRENT world.

    python webots/tools/make_shots.py --seed 101

Writes docs/img/sim_overview.png and docs/img/sim_aisle.png. It generates the
world with the robot's own controller disabled and appends a second, static
Robot carrying a wide-angle camera named "overview", driven by the snapshot
controller. Nothing here touches the audit pipeline or any result.

    python webots/tools/make_shots.py --seed 101 --lift

--lift instead writes docs/figures/scissor_L0|L1|L2.png: the same fixed camera
on the robot while the lift_shots controller holds each shelf level, so the
three frames differ only in the lift height.
"""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webots/tools"))

from make_world import make, wbt                                   # noqa: E402

SHOTS = {
    # name: (camera position, look-at target, horizontal FOV)
    "sim_overview": ((-3.6, -1.9, 3.2), (1.6, 1.5, 0.9), 1.15),
    # from the far end looking back, so the robot (which starts at the near
    # end of the aisle) is framed down the aisle instead of filling the lens
    "sim_aisle":    ((5.0, 1.50, 1.30), (-1.2, 1.5, 0.95), 1.25),
}


# one fixed three-quarter view of the robot, used for all three lift heights
# A tall robot wants a tall frame: 720x960 keeps the 2 m of lift travel
# in shot without dragging in three metres of racking either side.
LIFT_SHOT = ((-2.269, 1.550, 1.22), (-0.80, 1.15, 1.00), 1.0, 720, 960)
# a low close view of the base, to check the stack really is seated on it
BASE_SHOT = ((-1.330, 1.520, 0.330), (-0.80, 1.14, 0.165), 0.9, 1100, 760)
LIFT_HOLD = 250              # steps held per level by the lift_shots controller


def shot_robot(pos, target, fov, controller="snapshot", w=1280, h=720):
    dx, dy, dz = (t - p for t, p in zip(target, pos))
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    yaw = math.atan2(dy, dx)
    pitch = math.asin(-dz / n)
    return (f'Robot {{ translation {pos[0]} {pos[1]} {pos[2]} '
            f'rotation 0 0 1 {yaw:.6f} name "shot" controller "{controller}" '
            f'supervisor TRUE '
            f'children [ Pose {{ rotation 0 1 0 {pitch:.6f} children [ '
            f'Camera {{ name "overview" width {w} height {h} '
            f'fieldOfView {fov} }} ] }} ] }}')


def lift_shots(a):
    """Three frames of the robot from one camera, at the three shelf levels."""
    out = ROOT / "docs/figures"
    out.mkdir(parents=True, exist_ok=True)
    worlds = ROOT / "webots/worlds"
    lay = make(a.seed)
    pos, target, fov, w, h = BASE_SHOT if a.base else LIFT_SHOT
    text = (wbt(lay, a.version, "lift_shots")
            + shot_robot(pos, target, fov, w=w, h=h))
    wf = worlds / "_shot_scissor.wbt"
    wf.write_text(text)
    # the snapshot robot shoots just before each hold ends, by which time the
    # lift has stopped and the links have been posed for that height
    n = 1 if a.base else len(lay["lift_positions"])
    at = ",".join(str(LIFT_HOLD * (i + 1) - 10) for i in range(n))
    env = dict(os.environ,
               SHELFSCAN_SNAPSHOT=str(out / ("scissor_base.png" if a.base
                                             else "scissor_L{}.png")),
               SHELFSCAN_SNAPSHOT_AT=at,
               SHELFSCAN_LIFT_HOLD=str(LIFT_HOLD))
    rc = subprocess.run(
        [sys.executable, str(ROOT / "webots/tools/run_webots.py"), str(wf),
         "--timeout", "300"], env=env).returncode
    print(f"scissor shots: webots exit={rc} -> {out}")
    for p in (wf, worlds / "._shot_scissor.wbproj"):
        p.unlink(missing_ok=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--version", default="R2025a")
    ap.add_argument("--lift", action="store_true",
                    help="robot-at-three-lift-heights figures into docs/figures")
    ap.add_argument("--base", action="store_true",
                    help="close view of the chassis and the stack's feet")
    a = ap.parse_args()
    if a.lift or a.base:
        return lift_shots(a)
    out = ROOT / "docs/img"
    out.mkdir(parents=True, exist_ok=True)
    worlds = ROOT / "webots/worlds"
    lay = make(a.seed)

    for name, (pos, target, fov) in SHOTS.items():
        # the audit robot must not drive: give it no controller
        text = wbt(lay, a.version, "void") + shot_robot(pos, target, fov) + "\n"
        w = worlds / f"_shot_{name}.wbt"
        w.write_text(text)
        env = dict(os.environ, SHELFSCAN_SNAPSHOT=str(out / f"{name}.png"))
        rc = subprocess.run(
            [sys.executable, str(ROOT / "webots/tools/run_webots.py"), str(w),
             "--timeout", "180"], env=env).returncode
        print(f"{name}: webots exit={rc} -> {out / f'{name}.png'}")
        for p in (w, worlds / f"._shot_{name}.wbproj"):
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
