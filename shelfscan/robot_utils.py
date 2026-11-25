"""Webots helpers shared by the controllers. Imports `controller` lazily so the
rest of the package (and the unit tests) run without Webots installed."""
import json
import math
import os
from pathlib import Path

CLASSES = ["sku_A", "sku_B", "sku_C", "sku_D", "slot_empty", "blocker"]

# v3 class set, for the QR/barcode world: the four SKUs collapse into ONE
# cardboard_box class (in that world the SKU is carried by the code, not by the
# carton's look, so asking the detector to tell SKUs apart is asking it to read
# the code - which is zxing's job), and the code plate itself becomes a class so
# the audit can crop it before decoding.
CLASSES_V3 = ["cardboard_box", "label", "slot_empty", "blocker"]
CLASS_SETS = {"v2": CLASSES, "v3": CLASSES_V3}


CAPTURE_KINDS = ("empty_blocked",)


def slot_matches(slot, only):
    """Does this slot belong to the capture subset named by `only`?

    `only` empty (the default) means every slot. "empty_blocked" selects the
    empty bays that have a blocker in front of them - roughly one slot in a
    hundred, and the case the v3 detector got wrong on 5 of the 6 that
    appeared in seeds 101-110. Collecting it by driving to those slots alone
    costs one stop per seed instead of thirty, and leaves the scenes
    themselves untouched: the frames still show whatever is on the level.

    Shared so the capture controller and the batch runner cannot disagree
    about which seeds are worth starting Webots for."""
    if not only:
        return True
    if only == "empty_blocked":
        return not slot["actual"] and bool(slot.get("blocker_kind"))
    raise ValueError(f"unknown capture subset {only!r}")


def under_root(value, root, default):
    """Resolve an output path from the environment against the REPO ROOT.

    Webots starts a controller with its working directory set to the
    controller's OWN folder, so a relative path from the caller lands under
    webots/controllers/<name>/ instead of where the caller meant. It has
    bitten twice: --results results_v3/qr wrote the eval JSON to
    webots/controllers/shelfscan_audit/ and the batch runner reported all
    twenty seeds FAILED, and --out datasets_v4 wrote 364 captured frames to
    webots/controllers/data_capture/. Both runs had actually succeeded, and
    both looked like total failures. An absolute path and an unset value are
    untouched, so no existing run changes."""
    if not value:
        return root / default
    path = Path(value)
    return path if path.is_absolute() else root / path


def results_dir(value, root):
    """Where an audit run writes its JSON and CSV."""
    return under_root(value, root, "results_v2")


def class_set(name=None):
    """The class list to write labels against. SHELFSCAN_CLASSES selects it;
    the default is the original v2 set, so every existing run is unchanged."""
    return CLASS_SETS[name or os.environ.get("SHELFSCAN_CLASSES") or "v2"]

# TurtleBot3 Burger
WHEEL_R, AXLE, MAX_W = 0.033, 0.160, 6.67


def load_layout(path=None):
    path = path or os.environ.get("SHELFSCAN_LAYOUT",
                                  str(Path(__file__).resolve().parents[1] / "webots/worlds/layout.json"))
    return json.loads(Path(path).read_text())


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


CONTENT_DEPTH = 0.30     # nominal depth of the region where a bay's content sits


def content_box(slot, bay_size):
    """Centre and size of the region where a slot's content is expected to be.

    Deliberately the SAME shape for every slot: it is derived only from the bay
    geometry the robot knows from its map, never from the true carton size,
    which is ground truth the robot does not have."""
    cy = slot["front_y"] - slot["facing"] * CONTENT_DEPTH / 2
    return ((slot["x"], cy, slot["deck_top"] + bay_size[2] / 2),
            (bay_size[0], CONTENT_DEPTH, bay_size[2]))


class Lift:
    """The camera lift: a SliderJoint driven by the LinearMotor named "lift".

    `request` starts the move and returns at once, so the lift travels while
    the base drives; `settle` then waits for the carriage to reach the level.
    Height is read from the supervisor rather than a position sensor, so what
    is checked is the real camera height in the world."""

    def __init__(self, robot, step, defname="LIFT", links=None):
        self.robot, self.step = robot, step
        self.motor = robot.getDevice("lift")
        self.node = robot.getFromDef(defname)
        self.target = None
        # Carriage height with the lift fully down, read from the world rather
        # than hardcoded, so h() is the real travel whatever CAM_Z is.
        self.z0 = self.node.getPosition()[2]
        self.links = links

    def h(self):
        """Lift travel: carriage height above its fully-down position."""
        return self.node.getPosition()[2] - self.z0

    def tick(self):
        """Pose the visual scissor links for the current travel. Called after
        every control step, by Driver as well, so the stack folds and extends
        while the base drives."""
        if self.links is not None:
            self.links.update(self.h())

    def request(self, pos):
        if pos != self.target:
            self.motor.setPosition(pos)
            self.target = pos

    def height(self):
        return self.node.getPosition()[2]

    def settle(self, cam_z, tol=0.005, timeout_s=15.0):
        t_end = self.robot.getTime() + timeout_s
        while abs(self.height() - cam_z) > tol and self.robot.getTime() < t_end:
            if self.robot.step(self.step) == -1:
                raise SystemExit
            self.tick()
        return self.height()


class Driver:
    """Go-to-pose for a differential-drive robot using GPS + IMU feedback.
    No obstacle avoidance: viewpoint poses are on the open aisle side of the shelf.
    Tune the gains on first run."""

    def __init__(self, robot, step, on_step=None):
        self.robot, self.step = robot, step
        self.on_step = on_step        # called after every simulation step
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
        if self.on_step is not None:
            self.on_step()
        x, y, _ = self.pose()
        if self._last:
            self.distance += math.hypot(x - self._last[0], y - self._last[1])
        self._last = (x, y)

    AISLE_X = -0.8          # x of the free aisle end, where the robot starts

    def go_to(self, x, y, yaw, tol=0.04, yaw_tol=0.04, timeout_s=60.0):
        """Drive straight to (x, y), then turn to yaw.

        Every viewpoint lies in the single open aisle between the two racking
        rows, so a straight line between any two of them is clear. The earlier
        reroute via the aisle end existed only for the old multi-row world,
        where the robot had to avoid driving through a rack."""
        return self._go_pose(x, y, yaw, tol, yaw_tol, timeout_s)

    def _go_pose(self, x, y, yaw, tol, yaw_tol, timeout_s):
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


def run_logged(main, name):
    """Run a controller main() with stdout/stderr teed to logs/<name>.log.

    On Windows, Webots does not forward controller output to the launching
    terminal, so batch runs need a file log. On an exception the simulation is
    quit with status 1 so unattended runs never hang."""
    import sys
    import traceback
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(exist_ok=True)
    tag = os.environ.get("SHELFSCAN_TAG")
    f = open(log_dir / (f"{name}_{tag}.log" if tag else f"{name}.log"), "a", buffering=1)

    class Tee:
        def __init__(self, *s):
            self.s = s

        def write(self, x):
            for s in self.s:
                try:
                    s.write(x)
                except Exception:
                    pass

        def flush(self):
            for s in self.s:
                try:
                    s.flush()
                except Exception:
                    pass

    sys.stdout, sys.stderr = Tee(sys.__stdout__, f), Tee(sys.__stderr__, f)
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        try:
            import controller
            r = getattr(controller, "_shelfscan_robot", None) or controller.Supervisor()
            r.simulationQuit(1)
        except Exception:
            pass
        sys.exit(1)
