"""Scissor-lift link kinematics: pure geometry, no Webots and no numpy.

The lift itself is a real SliderJoint + LinearMotor that carries the platform
(and therefore the camera). The scissor links are NOT a physics joint chain -
they are visual-only `Pose` nodes whose translation and rotation are recomputed
from the lift height every control step. That keeps the dynamics of the robot
exactly what they were before the links existed, while the stack really folds
and extends.

Geometry of one stack (all in robot-local coordinates, x forward, z up):

  * N_STAGES stages, each a pair of equal bars of length LINK_L crossing at
    their midpoints.
  * Pivot rows i = 0 .. N_STAGES sit at z = BASE_Z + i * dz.
  * Every row carries two pins: a FRONT pin at the fixed x = X_FRONT and a
    BACK pin at X_FRONT - spread, which travels as the stack folds. Stage i is
    the bar (back_i -> front_i+1) plus the bar (front_i -> back_i+1), so
    consecutive stages are pin-jointed at both ends and only row 0 (on the
    chassis) and row N (under the platform) have one end that slides along a
    rail. The FRONT column is the pinned one, so at full extension the stack
    stands directly under the camera bar instead of trailing behind it.
  * With theta the bar angle to horizontal: dz = LINK_L * sin(theta),
    spread = LINK_L * cos(theta), stack height = N_STAGES * LINK_L * sin(theta).

So theta = asin(stack_height / (N_STAGES * LINK_L)). The stack height needed is
fixed by where the camera has to be:

    stack_height(h) = H_DOWN + h      (h = lift travel, 0 .. LIFT_MAX)

with H_DOWN = CAM_Z - BASE_Z - TOP_DROP. At h = 0 the X's are nearly flat
(theta ~ 11 deg) and at full lift they are steep (theta ~ 70 deg).

X_BACK is chosen so the widest spread still leaves every part of the lift at
robot-local x <= X_MAX = 0.032, i.e. behind the camera at x = 0.05, so no part
of the robot can ever enter its own field of view.
"""
import math

N_STAGES = 6
LINK_L = 0.29           # bar length
BASE_Z = 0.126          # z of the bottom pivot row (top of the mounting rails)
TOP_DROP = 0.103        # camera axis above the top pivot row
H_DOWN = 0.331          # stack height with the lift fully down
X_FRONT = 0.0082        # x of the fixed (front) pivot column
X_MAX = 0.032           # nothing on the lift may reach further forward
STACK_DY = 0.062        # |y| of each of the two stacks
BAR_T = 0.020           # bar cross-section
BAR_GAP = 0.011         # |y| offset of each bar of a stage from its stack line
PIN_R = 0.011
PIN_H = 0.046       # must span both bars of a stage (+-0.021)
CROSS_R = 0.013
CROSS_H = 0.050
CHEEK_T = 0.008     # rail/bracket cheek either side of a pin
CHEEK_DY = 0.028    # |y| of a cheek from its stack line

SPAN = N_STAGES * LINK_L


def geometry(h):
    """(theta, spread, dz, stack_height) for lift travel h."""
    hs = H_DOWN + h
    theta = math.asin(max(0.0, min(1.0, hs / SPAN)))
    return theta, LINK_L * math.cos(theta), hs / N_STAGES, hs


def row_z(h, i):
    _, _, dz, _ = geometry(h)
    return BASE_Z + i * dz


def back_x(h):
    """x of the sliding (back) pivot column at lift travel h."""
    return X_FRONT - geometry(h)[1]


def poses(h):
    """Every driven link part at lift travel h.

    Yields (def_name, (x, y, z), angle) where `angle` is the rotation about the
    robot's +y axis for a bar, or None for a pin (a Cylinder's axis is already
    +y, so pins never rotate)."""
    theta, spread, dz, _ = geometry(h)
    xm = X_FRONT - spread / 2
    for s, ys in enumerate((STACK_DY, -STACK_DY)):
        for i in range(N_STAGES):
            zm = BASE_Z + (i + 0.5) * dz
            # bar P: back_i -> front_i+1 ; bar Q: front_i -> back_i+1.
            # A Box's long axis is +x; rotating by `a` about +y points it along
            # (cos a, 0, -sin a), so P takes -theta and Q takes +theta.
            yield f"SCB{s}{i}P", (xm, ys - BAR_GAP, zm), -theta
            yield f"SCB{s}{i}Q", (xm, ys + BAR_GAP, zm), theta
            yield f"SCX{s}{i}", (xm, ys, zm), None
        for i in range(N_STAGES + 1):
            z = BASE_Z + i * dz
            yield f"SCP{s}{i}B", (X_FRONT - spread, ys, z), None
            yield f"SCP{s}{i}F", (X_FRONT, ys, z), None


def bar_ends(h, i):
    """Endpoints ((x, z), (x, z)) of the two bars of stage i, for checking."""
    theta, spread, dz, _ = geometry(h)
    z0, z1 = BASE_Z + i * dz, BASE_Z + (i + 1) * dz
    xb, xf = X_FRONT - spread, X_FRONT
    return ((xb, z0), (xf, z1)), ((xf, z0), (xb, z1))


class ScissorLinks:
    """Drive the visual link poses from the lift height.

    Built from whatever DEF names the loaded world actually has, so a world
    generated before the links existed simply gives an inert object. Nothing
    here touches physics: the nodes are plain Pose nodes with no bounding
    object, so updating them cannot change the simulation's dynamics."""

    def __init__(self, robot):
        self.fields = {}
        for name, _, angle in poses(0.0):
            node = robot.getFromDef(name)
            if node is None:
                continue
            self.fields[name] = (node.getField("translation"),
                                 node.getField("rotation") if angle is not None
                                 else None)
        self._h = None

    def __len__(self):
        return len(self.fields)

    def update(self, h):
        """Recompute every link pose for lift travel h. Cheap no-op while the
        lift is not moving, which is most control steps."""
        if self._h is not None and abs(h - self._h) < 1e-5:
            return False
        self._h = h
        for name, t, angle in poses(h):
            f = self.fields.get(name)
            if f is None:
                continue
            f[0].setSFVec3f([t[0], t[1], t[2]])
            if f[1] is not None:
                f[1].setSFRotation([0.0, 1.0, 0.0, angle])
        return True
