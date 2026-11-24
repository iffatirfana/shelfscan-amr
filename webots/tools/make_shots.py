"""Render the documentation figures from the CURRENT world.

    python webots/tools/make_shots.py --seed 101

Writes docs/img/sim_overview.png and docs/img/sim_aisle.png. It generates the
world with the robot's own controller disabled and appends a second, static
Robot carrying a wide-angle camera named "overview", driven by the snapshot
controller. Nothing here touches the audit pipeline or any result.
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


def shot_robot(pos, target, fov, controller="snapshot"):
    dx, dy, dz = (t - p for t, p in zip(target, pos))
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    yaw = math.atan2(dy, dx)
    pitch = math.asin(-dz / n)
    return (f'Robot {{ translation {pos[0]} {pos[1]} {pos[2]} '
            f'rotation 0 0 1 {yaw:.6f} name "shot" controller "{controller}" '
            f'supervisor TRUE '
            f'children [ Pose {{ rotation 0 1 0 {pitch:.6f} children [ '
            f'Camera {{ name "overview" width 1280 height 720 '
            f'fieldOfView {fov} }} ] }} ] }}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--version", default="R2025a")
    a = ap.parse_args()
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
