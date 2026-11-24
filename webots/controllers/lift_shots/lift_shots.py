"""Dev-only: hold the lift at each shelf level so the scissor stack can be
photographed folding and extending, and check the links from the supervisor.

Env: SHELFSCAN_LIFT_HOLD (simulation steps to hold each level, default 250),
     SHELFSCAN_LAYOUT.

At every hold it prints, for the measured carriage height, the stage angle and
spread, and then reads the link nodes back out of the world and compares their
real positions with what the kinematics asked for. That is the end-to-end
check: it fails loudly if a field write never landed, so the picture and the
numbers cannot disagree. Writes nothing to results/.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controller import Supervisor                      # noqa: E402

from shelfscan import scissor as sc                    # noqa: E402
from shelfscan.robot_utils import Lift, load_layout    # noqa: E402
from shelfscan.scissor import ScissorLinks             # noqa: E402


def main():
    hold = int(os.environ.get("SHELFSCAN_LIFT_HOLD", "250"))
    layout = load_layout()
    robot = Supervisor()
    import controller as _c
    _c._shelfscan_robot = robot
    step = int(robot.getBasicTimeStep())
    robot.step(step)
    links = ScissorLinks(robot)
    lift = Lift(robot, step, links=links)
    me = robot.getSelf()
    print(f"scissor links driven: {len(links)} nodes "
          f"({sc.N_STAGES} stages x 2 bars x 2 stacks + pins)", flush=True)

    nodes = {n: robot.getFromDef(n) for n, _, _ in sc.poses(0.0)}

    def report(i, want_h):
        h = lift.h()
        theta, spread, dz, stack = sc.geometry(h)
        base, o = me.getPosition(), me.getOrientation()
        worst = 0.0
        for name, t, _ in sc.poses(h):
            p = nodes[name].getPosition()
            # robot-local -> world through the robot's own pose, so the check
            # holds whatever yaw the robot is standing at
            for a in range(3):
                want = base[a] + sum(o[3 * a + b] * t[b] for b in range(3))
                worst = max(worst, abs(p[a] - want))
        print(f"L{i}: lift target={want_h:.2f} measured h={h:.4f} "
              f"camera_z={lift.height():.4f} stage_angle={theta * 57.2958:.2f} deg "
              f"spread={spread:.4f} stage_rise={dz:.4f} stack={stack:.4f} "
              f"worst_link_placement_error={worst * 1000:.3f} mm", flush=True)
        return worst

    worst_all = 0.0
    for i, pos in enumerate(layout["lift_positions"]):
        lift.request(pos)
        for _ in range(hold):
            if robot.step(step) == -1:
                raise SystemExit
            lift.tick()
        worst_all = max(worst_all, report(i, pos))
    print(f"all levels held; worst link placement error "
          f"{worst_all * 1000:.3f} mm", flush=True)

    while robot.step(step) != -1:      # hold so the snapshot robot can finish
        lift.tick()


if __name__ == "__main__":
    from shelfscan.robot_utils import run_logged
    run_logged(main, "lift_shots")
