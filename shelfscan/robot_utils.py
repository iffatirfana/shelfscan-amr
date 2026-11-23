"""Webots helpers shared by the controllers. Imports `controller` lazily so the
rest of the package (and the unit tests) run without Webots installed."""
import json
import math
import os
from pathlib import Path

CLASSES = ["sku_A", "sku_B", "sku_C", "sku_D", "slot_empty", "blocker"]

# TurtleBot3 Burger
WHEEL_R, AXLE, MAX_W = 0.033, 0.160, 6.67


def load_layout(path=None):
    path = path or os.environ.get("SHELFSCAN_LAYOUT",
                                  str(Path(__file__).resolve().parents[1] / "webots/worlds/layout.json"))
    return json.loads(Path(path).read_text())


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Driver:
    """Go-to-pose for a differential-drive robot using GPS + IMU feedback.
    No obstacle avoidance: viewpoint poses are on the open aisle side of the shelf.
    Tune the gains on first run."""

    def __init__(self, robot, step):
        self.robot, self.step = robot, step
        self.l = robot.getDevice("left wheel motor")
        self.r = robot.getDevice("right wheel motor")
        for m in (self.l, self.r):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.gps = robot.getDevice("gps")
        self.imu = robot.getDevice("imu")
        self.gps.enable(step)
        self.imu.enable(step)
        self.distance = 0.0
        self._last = None
        robot.step(step)  # prime sensors

    def pose(self):
        x, y, _ = self.gps.getValues()
        return x, y, self.imu.getRollPitchYaw()[2]

    def _drive(self, v, w):
        wl = (v - w * AXLE / 2) / WHEEL_R
        wr = (v + w * AXLE / 2) / WHEEL_R
        s = max(1.0, abs(wl) / MAX_W, abs(wr) / MAX_W)
        self.l.setVelocity(wl / s)
        self.r.setVelocity(wr / s)

    def _tick(self):
        if self.robot.step(self.step) == -1:
            raise SystemExit
        x, y, _ = self.pose()
        if self._last:
            self.distance += math.hypot(x - self._last[0], y - self._last[1])
        self._last = (x, y)

    def go_to(self, x, y, yaw, tol=0.04, yaw_tol=0.04, timeout_s=60.0):
        t_end = self.robot.getTime() + timeout_s
        while self.robot.getTime() < t_end:
            px, py, pyaw = self.pose()
            dx, dy = x - px, y - py
            dist = math.hypot(dx, dy)
            if dist > tol:
                err = wrap(math.atan2(dy, dx) - pyaw)
                if abs(err) > 0.25:
                    self._drive(0.0, 1.8 * err)
                else:
                    self._drive(min(0.18, 0.8 * dist), 2.5 * err)
            else:
                err = wrap(yaw - pyaw)
                if abs(err) < yaw_tol:
                    break
                self._drive(0.0, 1.8 * err)
            self._tick()
        self._drive(0.0, 0.0)
        for _ in range(3):
            self._tick()          # settle
        return True
