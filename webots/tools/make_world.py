"""Generate a randomised two-row warehouse aisle (.wbt) plus layout.json.

    python webots/tools/make_world.py --seed 101 --controller shelfscan_audit

Layout (v2):
  * TWO tall racking rows facing each other across ONE narrow aisle.
  * LEVELS levels per row, BAYS bays per level  ->  2 * LEVELS * BAYS slots.
  * Slot contents are corrugated CARDBOARD CARTONS. There is no colour cue:
    every carton is brown/tan. The SKU is carried only by a large black/white
    pattern printed on a white shipping label on the aisle-facing face
    (sku_A square, sku_B horizontal bars, sku_C dot grid, sku_D vertical bars).
  * Carton SIZE is drawn per slot from SIZE_CLASSES, independently of the SKU
    and of the brown shade, so neither size nor shade is a shortcut.
  * A blocker is an UNLABELLED carton or a pallet stood at the bay mouth; an
    empty slot is a bare bay. Both are told apart by shape/texture, not colour.

layout.json is the contract with the controllers: bay positions, the camera
height and lift setpoint per level, expected vs. true contents, carton size
class and pose, blockers, and per-slot viewpoint poses [x, y, yaw].

The world uses only Webots base nodes (no webots:// PROTOs), so it loads
offline. Targets Webots R2025a on Windows.
"""
import argparse
import json
import math
import random
from pathlib import Path

# ---------------------------------------------------------------- SKUs --------
# The 4 SKUs differ ONLY by the black/white print on the shipping label.
SKUS = ("sku_A", "sku_B", "sku_C", "sku_D")

# ------------------------------------------------------------ carton sizes ----
# (x = width along the aisle, y = depth into the rack, z = height)
SIZE_CLASSES = {
    "small":     (0.25, 0.20, 0.20),
    "medium":    (0.35, 0.28, 0.25),
    "large":     (0.45, 0.35, 0.30),
    "tall":      (0.30, 0.30, 0.45),
    "wide_flat": (0.50, 0.35, 0.18),
}
ITEM = SIZE_CLASSES["medium"]      # nominal carton, kept for the unit tests

# ------------------------------------------------------------ SKU printing ---
# The label is sized so that the thinnest printed feature clears 8 px at the
# 320 px inference size from the centre viewpoint, for EVERY size class and
# every SKU - the flattest class (wide_flat, 0.18 m high) is the binding case.
# Patterns use a 5-unit scheme: 3 bars/dots and 2 gaps, each 1/5 of the printed
# area, so bar thickness and gap width are equal and both as large as possible.
LABEL_W_FRAC = 0.90       # label width  / carton face width
LABEL_H_FRAC = 0.80       # label height / carton face height
LABEL_Z_FRAC = -0.06      # label centre / carton height, below the seam tape
PRINT_FRAC = 0.90         # printed area / label
PRINT_UNITS = 5           # bar thickness = gap = printed extent / 5
SEAM_T = 0.016            # front-seam tape height

# --------------------------------------------------------------- racking ------
SIDES, LEVELS, BAYS = 2, 3, 5      # -> 2 * 3 * 5 = 30 slots per seed
BAY_W = 0.60                       # bay pitch along x
BAY_INNER = 0.56                   # clear width inside a bay
SHELF_DEPTH = 0.80                 # rack depth along y
DECK_T = 0.05                      # shelf deck thickness
LEVEL_Z = (0.30, 0.95, 1.60)       # deck TOP surface height of each level
BAY_H = 0.60                       # clear height of a bay
RACK_H = 2.35                      # top of the uprights
AISLE_W = 2.20                     # clear aisle between the two rack faces
CEIL_Z = 6.0

# ------------------------------------------------------------- viewpoints -----
SETBACK = 0.20     # carton front face, measured back from the bay face plane
STAND_D = 0.90     # camera -> carton front distance, centre viewpoint
CLOSE_D = 0.65     # ... close viewpoint
LATERAL = 0.45     # left/right viewpoint lateral shift
CAM_AIM = 0.26     # camera height above the deck top of the level it scans

CAM_LEVEL_Z = tuple(round(z + CAM_AIM, 4) for z in LEVEL_Z)          # .56 1.21 1.86
LIFT_POS = tuple(round(z - CAM_LEVEL_Z[0], 4) for z in CAM_LEVEL_Z)  # 0 .65 1.30
CAM_Z = CAM_LEVEL_Z[0]             # camera height with the lift fully down
ROW_Y = (0.0, round(SHELF_DEPTH + AISLE_W, 4))   # rack centre y of each row
RACK_LABEL = ("A", "B")
AISLE_MID_Y = round((SHELF_DEPTH / 2 + ROW_Y[1] - SHELF_DEPTH / 2) / 2, 4)


def bay_x(k):
    return round(k * BAY_W, 4)


def slot_id(r, l, k):
    return f"S{r}-{l}-{k}"


def make(seed, p_empty=0.08, p_mis=0.08, p_occ=0.20, p_exp_empty=0.05):
    """Deterministic per seed: the RNG is consumed in a fixed (row, level, bay)
    order, independent of the scan order written to layout["order"]."""
    rng = random.Random(seed)
    slots = {}
    for r in range(SIDES):
        f = 1 if r == 0 else -1                 # +1: robot stands on the +y side
        y_rack = ROW_Y[r]
        face_y = round(y_rack + f * SHELF_DEPTH / 2, 4)
        front_y = round(face_y - f * SETBACK, 4)        # carton front plane
        for l in range(LEVELS):
            deck_top = LEVEL_Z[l]
            for k in range(BAYS):
                x = bay_x(k)
                # ---- contents ------------------------------------------------
                exp = None if rng.random() < p_exp_empty else rng.choice(SKUS)
                q = rng.random()
                if exp is None:
                    act = rng.choice(SKUS) if q < p_mis else None
                elif q < p_empty:
                    act = None
                elif q < p_empty + p_mis:
                    act = rng.choice([s for s in SKUS if s != exp])
                else:
                    act = exp
                # ---- carton shape (drawn whatever the content, so the RNG
                #      stream does not depend on emptiness) --------------------
                cls = rng.choice(list(SIZE_CLASSES))
                sx, sy, sz = SIZE_CLASSES[cls]
                shade = round(rng.uniform(-0.06, 0.06), 4)
                on_pallet = rng.random() < 0.30
                two = cls == "small" and rng.random() < 0.35
                dx_max = max(0.0, (BAY_INNER - sx) / 2 - 0.01)
                dx = 0.0 if two else round(rng.uniform(-1, 1) * min(0.05, dx_max), 4)
                cyaw = round(rng.uniform(-0.09, 0.09), 4)
                pallet_h = 0.06 if on_pallet else 0.0
                cz = round(deck_top + pallet_h + sz / 2, 4)
                cy = round(front_y - f * sy / 2, 4)
                # ---- blocker -------------------------------------------------
                occ = rng.choice(["left", "right"]) if rng.random() < p_occ else None
                b_kind = b_cls = b_size = b_pos = b_yaw = None
                if occ:
                    b_kind = "pallet" if rng.random() < 0.35 else "carton"
                    if b_kind == "pallet":
                        b_cls, b_size = "pallet", [0.45, 0.09, 0.55]
                    else:
                        b_cls = rng.choice(list(SIZE_CLASSES))
                        b_size = list(SIZE_CLASSES[b_cls])
                    sgn = 1 if occ == "left" else -1      # image-left side
                    b_pos = [round(x + f * sgn * 0.20, 4), face_y,
                             round(deck_top + b_size[2] / 2, 4)]
                    b_yaw = round(rng.uniform(-0.07, 0.07), 4)
                # ---- viewpoints ----------------------------------------------
                yaw0 = round(-math.pi / 2 if f > 0 else math.pi / 2, 6)
                vps = {
                    "center": [x, round(front_y + f * STAND_D, 4), yaw0],
                    "close":  [x, round(front_y + f * CLOSE_D, 4), yaw0],
                    "left":   _aimed(x + f * LATERAL, front_y + f * STAND_D, x, front_y),
                    "right":  _aimed(x - f * LATERAL, front_y + f * STAND_D, x, front_y),
                }
                slots[slot_id(r, l, k)] = dict(
                    x=x, y=y_rack, z=round(deck_top + BAY_H / 2, 4),
                    face_y=face_y, front_y=front_y,
                    rack=RACK_LABEL[r], facing=f, level=l, bay=k,
                    deck_top=deck_top, cam_z=CAM_LEVEL_Z[l], lift=LIFT_POS[l],
                    expected=exp, actual=act, occluder=occ,
                    size_class=cls, carton_size=[sx, sy, sz],
                    carton_pos=[round(x + dx, 4), cy, cz], carton_yaw=cyaw,
                    carton_shade=shade, on_pallet=on_pallet,
                    n_cartons=(2 if two else 1) if act else 0,
                    blocker_kind=b_kind, blocker_class=b_cls,
                    blocker_size=b_size, blocker_pos=b_pos, blocker_yaw=b_yaw,
                    viewpoints=vps)
    # ---- scan order: one side at a time, top level down, serpentine ----------
    order = []
    for r in range(SIDES):
        for i, l in enumerate(range(LEVELS - 1, -1, -1)):
            ks = range(BAYS) if i % 2 == 0 else range(BAYS - 1, -1, -1)
            order += [slot_id(r, l, k) for k in ks]
    return dict(seed=seed, n_slots=len(slots),
                n_sides=SIDES, n_levels=LEVELS, n_bays=BAYS,
                level_z=list(LEVEL_Z), cam_level_z=list(CAM_LEVEL_Z),
                lift_positions=list(LIFT_POS),
                bay_size=[BAY_INNER, SHELF_DEPTH, BAY_H],
                slot_size=[BAY_INNER, SHELF_DEPTH, BAY_H],   # legacy alias
                size_classes={k: list(v) for k, v in SIZE_CLASSES.items()},
                slots=slots, order=order,
                start_pose=[-0.8, slots[order[0]]["viewpoints"]["center"][1],
                            round(-math.pi / 2, 6)])


def _aimed(px, py, tx, ty):
    """Viewpoint [x, y, yaw] at (px, py) turned to look at (tx, ty)."""
    return [round(px, 4), round(py, 4), round(math.atan2(ty - py, tx - px), 6)]


# ============================================================ VRML helpers ====
# name -> (r, g, b, roughness, metalness, emissive)
PALETTE = {
    "CONCRETE":     (0.72, 0.72, 0.70, 0.95, 0.0, None),
    "LANE_YELLOW":  (0.90, 0.74, 0.10, 0.70, 0.0, None),
    "CEILING":      (0.80, 0.80, 0.82, 0.90, 0.0, None),
    "TRUSS_STEEL":  (0.52, 0.54, 0.57, 0.45, 0.7, None),
    "RACK_BLUE":    (0.09, 0.24, 0.60, 0.55, 0.25, None),
    "RACK_ORANGE":  (0.93, 0.42, 0.08, 0.50, 0.25, None),
    "DECK_STEEL":   (0.58, 0.58, 0.60, 0.60, 0.30, None),
    "RACK_MESH":    (0.28, 0.29, 0.32, 0.80, 0.20, None),
    "CARD_TAPE":    (0.45, 0.33, 0.21, 0.75, 0.0, None),
    "LABEL_WHITE":  (0.95, 0.95, 0.93, 0.50, 0.0, None),
    "PRINT_BLACK":  (0.03, 0.03, 0.03, 0.60, 0.0, None),
    "WOOD_PALLET":  (0.56, 0.42, 0.26, 0.90, 0.0, None),
    "BOT_GREY":     (0.34, 0.35, 0.37, 0.55, 0.35, None),
    "BOT_YELLOW":   (0.95, 0.74, 0.08, 0.45, 0.20, None),
    "BOT_DARK":     (0.10, 0.10, 0.11, 0.60, 0.0, None),
    "BOT_LED":      (1.00, 1.00, 1.00, 0.25, 0.0, (1.0, 1.0, 1.0)),
}
_SEEN = set()


def _app(name):
    """DEF an appearance on first use, USE it afterwards. Must be called at
    every use site - caching the returned string would emit duplicate DEFs."""
    if name in _SEEN:
        return f"USE {name}"
    _SEEN.add(name)
    r, g, b, rough, metal, emis = PALETTE[name]
    e = f" emissiveColor {emis[0]} {emis[1]} {emis[2]}" if emis else ""
    return (f"DEF {name} PBRAppearance {{ baseColor {r:.4f} {g:.4f} {b:.4f} "
            f"roughness {rough} metalness {metal}{e} }}")


def _card_app(shade):
    """Per-carton brown shade, DEF'd once per distinct shade."""
    name = f"CARD{int(round((shade + 0.10) * 1000)):04d}"
    if name in _SEEN:
        return f"USE {name}"
    _SEEN.add(name)
    return (f"DEF {name} PBRAppearance {{ baseColor {0.62 + shade:.4f} "
            f"{0.47 + shade * 0.85:.4f} {0.32 + shade * 0.7:.4f} "
            f"roughness 0.95 metalness 0 }}")


def _box(app, sx, sy, sz, x=0.0, y=0.0, z=0.0):
    return (f'Pose {{ translation {x:.4f} {y:.4f} {z:.4f} children [ '
            f'Shape {{ appearance {app} '
            f'geometry Box {{ size {sx:.4f} {sy:.4f} {sz:.4f} }} }} ] }}')


def _solid(name, children, x=0.0, y=0.0, z=0.0, yaw=0.0, model=None, rec=None,
           defname=None):
    d = f"DEF {defname} " if defname else ""
    m = f'model "{model}" ' if model else ""
    rc = f"recognitionColors [ {rec[0]} {rec[1]} {rec[2]} ] " if rec else ""
    return (f'{d}Solid {{ name "{name}" {m}translation {x:.4f} {y:.4f} {z:.4f} '
            f'rotation 0 0 1 {yaw:.6f} {rc}children [ {children} ] }}')


def _norm(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _look_at(pos, target):
    """Webots Viewpoint (axis, angle) for a camera at `pos` aimed at `target`.
    A Webots viewpoint looks along its own -z with +y up."""
    fwd = _norm([t - p for t, p in zip(target, pos)])
    zc = [-v for v in fwd]
    xc = _norm(_cross([0.0, 0.0, 1.0], zc))
    yc = _cross(zc, xc)
    m = [[xc[0], yc[0], zc[0]], [xc[1], yc[1], zc[1]], [xc[2], yc[2], zc[2]]]
    ang = math.acos(max(-1.0, min(1.0, (m[0][0] + m[1][1] + m[2][2] - 1.0) / 2.0)))
    if abs(ang) < 1e-9:
        return (0.0, 0.0, 1.0, 0.0)
    s = 2.0 * math.sin(ang)
    ax = _norm([(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s])
    return (ax[0], ax[1], ax[2], ang)


# ============================================================ world content ===
def thinnest_feature(sx, sz):
    """Size (m) of the smallest printed black feature on a carton face sx x sz.
    Equal for every SKU by construction, so one number bounds them all."""
    pw = PRINT_FRAC * LABEL_W_FRAC * sx
    ph = PRINT_FRAC * LABEL_H_FRAC * sz
    return min(pw, ph) / PRINT_UNITS


def _label_print(sku, pw, ph, y, lz):
    """Large, high-contrast black print on the white label. The pattern TYPE is
    the whole signal: no colour, and it carries no size or shade information.

    3 bars/dots with 2 gaps fill the printed area, each PRINT_UNITS-th of it, so
    the black features and the white gaps are the same width and as thick as the
    label allows. Bar/dot centres sit at +-2/5 of the printed extent."""
    u = 1.0 / PRINT_UNITS
    span_x, span_z = 2 * u * pw, 2 * u * ph
    out = []
    if sku == "sku_A":                                   # one big black square
        s = 0.80 * min(pw, ph)
        out.append(_box(_app("PRINT_BLACK"), s, 0.002, s, 0, y, lz))
    elif sku == "sku_B":                                 # thick horizontal bars
        for dz in (-span_z, 0.0, span_z):
            out.append(_box(_app("PRINT_BLACK"), pw, 0.002, u * ph, 0, y, lz + dz))
    elif sku == "sku_C":                                 # 3x3 square dot grid
        d = u * min(pw, ph)
        for dx in (-span_x, 0.0, span_x):
            for dz in (-span_z, 0.0, span_z):
                out.append(_box(_app("PRINT_BLACK"), d, 0.002, d, dx, y, lz + dz))
    else:                                                # sku_D vertical bars
        for dx in (-span_x, 0.0, span_x):
            out.append(_box(_app("PRINT_BLACK"), u * pw, 0.002, ph, dx, y, lz))
    return out


def _carton_children(size, shade, sku):
    """Corrugated carton: brown body, darker tape across the top and along the
    front seam, and - for a real SKU carton - a white shipping label carrying
    the black print. Local +y is the aisle-facing direction."""
    sx, sy, sz = size
    out = [_box(_card_app(shade), sx, sy, sz),
           _box(_app("CARD_TAPE"), 0.055, sy + 0.004, 0.004, 0, 0, sz / 2 + 0.002),
           _box(_app("CARD_TAPE"), sx + 0.004, 0.004, SEAM_T,
                0, sy / 2 + 0.002, sz / 2 - SEAM_T / 2)]
    if sku:
        lw, lh = LABEL_W_FRAC * sx, LABEL_H_FRAC * sz
        lz = LABEL_Z_FRAC * sz
        out.append(_box(_app("LABEL_WHITE"), lw, 0.003, lh,
                        0, sy / 2 + 0.0025, lz))
        out += _label_print(sku, PRINT_FRAC * lw, PRINT_FRAC * lh,
                            sy / 2 + 0.005, lz)
    return " ".join(out)


def _pallet_children(px, py):
    out = [_box(_app("WOOD_PALLET"), px, py, 0.016, 0, 0, 0.022)]
    for dy in (0.0, py / 2 - 0.035, -(py / 2 - 0.035)):
        out.append(_box(_app("WOOD_PALLET"), px, 0.07, 0.044, 0, dy, -0.008))
    return " ".join(out)


def _standing_pallet_children():
    """A pallet stood on edge at the bay mouth. Local +y faces the aisle."""
    out = []
    for dx in (0.0, 0.19, -0.19):
        out.append(_box(_app("WOOD_PALLET"), 0.05, 0.08, 0.55, dx, 0, 0))
    for dz in (-0.22, -0.11, 0.0, 0.11, 0.22):
        out.append(_box(_app("WOOD_PALLET"), 0.45, 0.03, 0.07, 0, 0.035, dz))
    return " ".join(out)


def _racking(r):
    """Uprights, beams, decks and back panels of one racking row."""
    f = 1 if r == 0 else -1
    kids, run = [], BAYS * BAY_W + 0.10
    xc = (BAYS - 1) * BAY_W / 2
    for k in range(BAYS + 1):                                     # uprights
        xp = -BAY_W / 2 + k * BAY_W
        for dy in (SHELF_DEPTH / 2 - 0.035, -(SHELF_DEPTH / 2 - 0.035)):
            kids.append(_box(_app("RACK_BLUE"), 0.07, 0.07, RACK_H,
                             xp, dy, RACK_H / 2))
    for z in list(LEVEL_Z) + [RACK_H - 0.05]:                     # beams
        for dy in (SHELF_DEPTH / 2 - 0.04, -(SHELF_DEPTH / 2 - 0.04)):
            kids.append(_box(_app("RACK_ORANGE"), run, 0.08, 0.10,
                             xc, dy, z - 0.05))
    for z in LEVEL_Z:                                             # decks, backs
        for k in range(BAYS):
            kids.append(_box(_app("DECK_STEEL"), BAY_INNER, SHELF_DEPTH - 0.08,
                             DECK_T, bay_x(k), 0, z - DECK_T / 2))
            # full-height back panel: closes the bay so the camera cannot see
            # through the rack to the background
            kids.append(_box(_app("RACK_MESH"), BAY_INNER, 0.02, BAY_H,
                             bay_x(k), -f * (SHELF_DEPTH / 2 - 0.03),
                             z + BAY_H / 2))
    return _solid(f"rack_{RACK_LABEL[r]}", " ".join(kids), 0.0, ROW_Y[r], 0.0)


def _building():
    out = [f'Solid {{ name "floor" children [ Shape {{ appearance {_app("CONCRETE")} '
           f'geometry Plane {{ size 40 40 }} }} ] '
           f'boundingObject Plane {{ size 40 40 }} }}',
           _solid("lane_line", _box(_app("LANE_YELLOW"), 12.0, 0.12, 0.004),
                  2.0, AISLE_MID_Y, 0.002),
           _solid("ceiling", _box(_app("CEILING"), 40.0, 30.0, 0.10),
                  0.0, AISLE_MID_Y, CEIL_Z)]
    kids = []
    for i in range(9):                                      # cross trusses
        kids.append(_box(_app("TRUSS_STEEL"), 0.12, 8.0, 0.28,
                         -4.0 + i * 1.8, 0.0, 0.0))
    for dy in (-3.2, 0.0, 3.2):                             # chords
        kids.append(_box(_app("TRUSS_STEEL"), 16.0, 0.10, 0.18, 2.0, dy, -0.16))
    out.append(_solid("trusses", " ".join(kids), 0.0, AISLE_MID_Y, CEIL_Z - 0.45))
    return out


def robot_node(x0, y0, yaw, controller):
    """Scissor-lift look-alike built from base nodes only.

    The drive train is unchanged from v1 (TurtleBot3-Burger geometry: wheel
    r = 0.033 m, track 0.160 m) so shelfscan.robot_utils.Driver stays valid,
    and the collision/physics body is still the small low cylinder, so the tall
    mast cannot destabilise the chassis.

    The lift is real: a SliderJoint driven by a LinearMotor named "lift" raises
    the top bar, and the 640x480 camera rides on that bar, so the camera height
    really changes with the shelf level. The yellow crossed bars are a static,
    simplified scissor mast. Every piece of mast and top-bar geometry is kept
    at robot-local x <= 0.032, i.e. behind the camera at x = 0.05, so none of
    it can enter the camera's field of view.
    """
    def wheel(side, y):
        return (f'HingeJoint {{ jointParameters HingeJointParameters {{ '
                f'axis 0 1 0 anchor 0 {y} 0.033 }} '
                f'device [ RotationalMotor {{ name "{side} wheel motor" '
                f'maxVelocity 6.67 maxTorque 10 }} ] '
                f'endPoint Solid {{ name "{side} wheel" translation 0 {y} 0.033 '
                f'rotation 1 0 0 1.5708 children [ Shape {{ '
                f'appearance {_app("BOT_DARK")} '
                f'geometry Cylinder {{ height 0.018 radius 0.033 }} }} ] '
                f'boundingObject Cylinder {{ height 0.018 radius 0.033 }} '
                f'physics Physics {{ density -1 mass 0.05 }} }} }}')

    def caster(name, x):
        return (f'Solid {{ name "{name}" translation {x} 0 0.016 '
                f'contactMaterial "caster" children [ Shape {{ '
                f'appearance {_app("BOT_GREY")} '
                f'geometry Sphere {{ radius 0.015 }} }} ] '
                f'boundingObject Sphere {{ radius 0.015 }} '
                f'physics Physics {{ density -1 mass 0.01 }} }}')

    body = [_box(_app("BOT_GREY"), 0.24, 0.17, 0.055, 0, 0, 0.055),
            _box(_app("BOT_YELLOW"), 0.24, 0.012, 0.050, 0, 0.088, 0.055),
            _box(_app("BOT_YELLOW"), 0.24, 0.012, 0.050, 0, -0.088, 0.055),
            _box(_app("BOT_DARK"), 0.25, 0.14, 0.018, 0, 0, 0.026)]
    mast_h = CAM_Z + LIFT_POS[-1] + 0.12
    for dy in (0.06, -0.06):
        body.append(_box(_app("BOT_YELLOW"), 0.030, 0.030, mast_h - 0.10,
                         -0.055, dy, 0.10 + (mast_h - 0.10) / 2))
        for i in range(7):                                  # scissor lattice
            zc = 0.14 + i * (mast_h - 0.24) / 7
            for sgn in (1, -1):
                body.append(
                    f'Pose {{ translation -0.055 {dy} {zc:.4f} '
                    f'rotation 0 1 0 {sgn * 0.62:.4f} children [ Shape {{ '
                    f'appearance {_app("BOT_YELLOW")} '
                    f'geometry Box {{ size 0.020 0.020 0.200 }} }} ] }}')
    carriage = [_box(_app("BOT_GREY"), 0.12, 0.26, 0.035, -0.04, 0, -0.030),
                _box(_app("BOT_LED"), 0.11, 0.020, 0.008, -0.04, 0.115, -0.008),
                _box(_app("BOT_LED"), 0.11, 0.020, 0.008, -0.04, -0.115, -0.008),
                _box(_app("BOT_DARK"), 0.05, 0.055, 0.045, 0, 0, 0),
                _box(_app("BOT_DARK"), 0.05, 0.050, 0.045, 0, 0.105, 0),
                _box(_app("BOT_DARK"), 0.05, 0.050, 0.045, 0, -0.105, 0)]
    for dy in (0.105, -0.105, 0.0):                         # lens discs
        carriage.append(
            f'Pose {{ translation 0.027 {dy} 0 rotation 0 0 1 -1.5708 '
            f'children [ Shape {{ appearance {_app("BOT_DARK")} '
            f'geometry Cylinder {{ height 0.008 radius 0.016 }} }} ] }}')
    carriage.append(
        'DEF CAM Camera { name "camera" translation 0.05 0 0 width 640 '
        'height 480 fieldOfView 1.0 '
        'recognition Recognition { maxRange 4 frameThickness 0 } }')
    lift = (f'SliderJoint {{ jointParameters JointParameters {{ axis 0 0 1 }} '
            f'device [ LinearMotor {{ name "lift" minPosition 0 '
            f'maxPosition {LIFT_POS[-1]} maxForce 300 maxVelocity 0.35 }} ] '
            f'endPoint DEF LIFT Solid {{ name "lift carriage" '
            f'translation 0 0 {CAM_Z} children [ {" ".join(carriage)} ] '
            f'boundingObject Box {{ size 0.02 0.02 0.02 }} '
            f'physics Physics {{ density -1 mass 0.05 }} }} }}')
    return (f'DEF SHELFSCAN Robot {{ translation {x0} {y0} 0 '
            f'rotation 0 0 1 {yaw} name "shelfscan" controller "{controller}" '
            f'supervisor TRUE children [ {" ".join(body)} '
            f'{wheel("left", 0.08)} {wheel("right", -0.08)} '
            f'{caster("front caster", 0.07)} {caster("rear caster", -0.07)} '
            f'GPS {{ name "gps" }} InertialUnit {{ name "imu" }} {lift} ] '
            f'boundingObject Pose {{ translation 0 0 0.055 children [ '
            f'Cylinder {{ height 0.08 radius 0.09 }} ] }} '
            f'physics Physics {{ density -1 mass 1.0 }} }}')


def wbt(layout, version, controller="shelfscan_audit"):
    _SEEN.clear()
    L = []
    a = L.append
    a(f"#VRML_SIM {version} utf8")
    a('WorldInfo { basicTimeStep 16 contactProperties [ '
      'ContactProperties { material2 "caster" coulombFriction [ 0 ] } ] }')
    vp = (-2.8, AISLE_MID_Y, 1.75)
    ax, ay, az, ang = _look_at(vp, (2.4, AISLE_MID_Y, 0.95))
    a(f'Viewpoint {{ orientation {ax:.5f} {ay:.5f} {az:.5f} {ang:.5f} '
      f'position {vp[0]} {vp[1]} {vp[2]} }}')
    a('Background { skyColor [ 0.62 0.66 0.70 ] luminosity 1.6 }')
    a('DirectionalLight { direction -0.3 -0.6 -1 intensity 2.2 castShadows FALSE }')
    a('DirectionalLight { direction 0.3 0.7 -1 intensity 1.6 castShadows FALSE }')
    a('DirectionalLight { direction 0 0 -1 intensity 1.2 castShadows FALSE }')
    for n in _building():
        a(n)
    for r in range(SIDES):
        a(_racking(r))
    for sid, s in layout["slots"].items():
        n = sid.replace("-", "_")
        f, (sx, sy, sz) = s["facing"], s["carton_size"]
        base_yaw = 0.0 if f > 0 else math.pi
        cx, cy, cz = s["carton_pos"]
        if s["actual"]:
            if s["on_pallet"]:
                pw = 2 * sx + 0.10 if s["n_cartons"] == 2 else sx + 0.06
                a(_solid(f"pallet_{n}",
                         _pallet_children(min(BAY_INNER, pw),
                                          min(SHELF_DEPTH - 0.10, sy + 0.06)),
                         cx, cy, s["deck_top"], base_yaw + s["carton_yaw"]))
            if s["n_cartons"] == 2:
                for i, dx in enumerate((-(sx / 2 + 0.02), sx / 2 + 0.02)):
                    a(_solid(f"item_{n}_{i}",
                             _carton_children(s["carton_size"],
                                              s["carton_shade"], s["actual"]),
                             cx + dx, cy, cz, base_yaw + s["carton_yaw"],
                             model=s["actual"], rec=(0.62, 0.47, 0.32),
                             defname=f"ITEM_{n}_{i}"))
            else:
                a(_solid(f"item_{n}",
                         _carton_children(s["carton_size"],
                                          s["carton_shade"], s["actual"]),
                         cx, cy, cz, base_yaw + s["carton_yaw"],
                         model=s["actual"], rec=(0.62, 0.47, 0.32),
                         defname=f"ITEM_{n}"))
        if s["occluder"]:
            bx, by, bz = s["blocker_pos"]
            kids = (_standing_pallet_children() if s["blocker_kind"] == "pallet"
                    else _carton_children(s["blocker_size"],
                                          -s["carton_shade"], None))
            a(_solid(f"blocker_{n}", kids, bx, by, bz,
                     base_yaw + s["blocker_yaw"], model="blocker",
                     rec=(0.56, 0.42, 0.26)))
        # marker node used only for ground-truth lookup by the offline tools
        a(f'DEF SLOT_{n} Pose {{ translation {s["x"]} {s["y"]} {s["z"]} }}')
    x0, y0, yaw = layout["start_pose"]
    a(robot_node(x0, y0, yaw, controller))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="webots/worlds")
    ap.add_argument("--version", default="R2025a")
    ap.add_argument("--controller", default="shelfscan_audit")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lay = make(a.seed)
    (out / "layout.json").write_text(json.dumps(lay, indent=1))
    (out / "warehouse.wbt").write_text(wbt(lay, a.version, a.controller))
    cls, sku, blk = {}, {}, {}
    for s in lay["slots"].values():
        cls[s["size_class"]] = cls.get(s["size_class"], 0) + 1
        sku[s["actual"] or "(empty)"] = sku.get(s["actual"] or "(empty)", 0) + 1
        blk[s["blocker_kind"] or "(none)"] = blk.get(s["blocker_kind"] or "(none)", 0) + 1
    print(f"wrote {out/'warehouse.wbt'} and {out/'layout.json'} (seed {a.seed}, "
          f"{lay['n_slots']} slots = {SIDES} sides x {LEVELS} levels x {BAYS} bays)")
    print("  size classes: " + ", ".join(f"{k}={v}" for k, v in sorted(cls.items())))
    print("  true content: " + ", ".join(f"{k}={v}" for k, v in sorted(sku.items())))
    print("  blockers:     " + ", ".join(f"{k}={v}" for k, v in sorted(blk.items())))
    print("  levels (deck top / camera z / lift): " + ", ".join(
        f"L{i}: {LEVEL_Z[i]} / {CAM_LEVEL_Z[i]} / {LIFT_POS[i]}" for i in range(LEVELS)))


if __name__ == "__main__":
    main()
