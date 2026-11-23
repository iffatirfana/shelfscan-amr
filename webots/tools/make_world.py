"""Generate a randomised warehouse world (.wbt) plus layout.json.

    python webots/tools/make_world.py --seed 3 --out webots/worlds/ --version R2025a

layout.json is the contract with the controllers: slot positions, expected SKUs,
the true contents, occluders, and per-slot viewpoint poses [x, y, yaw].

The world uses only Webots base nodes (no webots:// PROTOs), so it loads
offline. Verified to load and run in Webots R2025a on Windows.
"""
import argparse
import json
import math
import random
from pathlib import Path

SKUS = {"sku_A": (0.90, 0.20, 0.20), "sku_B": (0.20, 0.70, 0.25),
        "sku_C": (0.20, 0.35, 0.90), "sku_D": (0.95, 0.80, 0.15)}

SHELVES, SLOTS = 3, 6          # shelf rows, slots per row
SLOT_W = 0.45                  # slot pitch along x
ITEM = (0.30, 0.30, 0.30)      # item box size (x, y, z)
SHELF_DEPTH = 0.50
PLATFORM_Z, PLATFORM_T = 0.30, 0.04   # shelf platform centre height / thickness
PANEL_H = 0.60                       # back panel height
ROW_PITCH = 3.2                # distance between rows along y
STAND_OFF = 1.0                # robot distance from shelf face (centre view)
CLOSE_OFF = 0.65
LATERAL = 0.55                 # left/right viewpoint lateral shift


def make(seed, p_empty=0.08, p_mis=0.08, p_occ=0.20, p_exp_empty=0.05):
    rng = random.Random(seed)
    slots, order = {}, []
    for r in range(SHELVES):
        face_y = ROW_PITCH * r + SHELF_DEPTH / 2
        for k in range(SLOTS):
            sid = f"S{r}-{k}"
            x = k * SLOT_W
            z = PLATFORM_Z + PLATFORM_T / 2 + ITEM[2] / 2
            exp = None if rng.random() < p_exp_empty else rng.choice(list(SKUS))
            q = rng.random()
            if exp is None:
                act = rng.choice(list(SKUS)) if q < p_mis else None
            elif q < p_empty:
                act = None
            elif q < p_empty + p_mis:
                act = rng.choice([s for s in SKUS if s != exp])
            else:
                act = exp
            occ = None
            if rng.random() < p_occ:
                occ = rng.choice(["left", "right"])
            slots[sid] = dict(
                x=x, y=ROW_PITCH * r, z=z, face_y=face_y, expected=exp, actual=act,
                occluder=occ,
                viewpoints={
                    "center": [x, face_y + STAND_OFF, -math.pi / 2],
                    # side viewpoints are turned to look AT the slot (first version kept
                    # yaw = -pi/2, which put the slot at the image edge)
                    "left":   [x + LATERAL, face_y + STAND_OFF, math.atan2(-(STAND_OFF + ITEM[1] / 2 + 0.02), -LATERAL)],
                    "right":  [x - LATERAL, face_y + STAND_OFF, math.atan2(-(STAND_OFF + ITEM[1] / 2 + 0.02), LATERAL)],
                    "close":  [x, face_y + CLOSE_OFF, -math.pi / 2],
                })
            order.append(sid)
    return dict(seed=seed, slot_size=list(ITEM), slots=slots, order=order,
                start_pose=[-0.8, slots["S0-0"]["face_y"] + STAND_OFF, -math.pi / 2])


CAM_Z = 0.47                   # camera mast height = item centre height


def _mat(r, g, b):
    return f"Appearance {{ material Material {{ diffuseColor {r} {g} {b} }} }}"


def robot_node(x0, y0, yaw, controller):
    """Self-contained differential-drive robot (TurtleBot3-Burger wheel geometry:
    r=0.033 m, track 0.160 m) built from base nodes only, so the world needs no
    network PROTO download. Camera sits on a mast at item height."""
    def wheel(side, y):
        return (f'HingeJoint {{ jointParameters HingeJointParameters {{ axis 0 1 0 anchor 0 {y} 0.033 }} '
                f'device [ RotationalMotor {{ name "{side} wheel motor" maxVelocity 6.67 maxTorque 10 }} ] '
                f'endPoint Solid {{ name "{side} wheel" translation 0 {y} 0.033 rotation 1 0 0 1.5708 '
                f'children [ Shape {{ appearance {_mat(0.1, 0.1, 0.1)} geometry Cylinder {{ height 0.018 radius 0.033 }} }} ] '
                f'boundingObject Cylinder {{ height 0.018 radius 0.033 }} '
                f'physics Physics {{ density -1 mass 0.05 }} }} }}')

    def caster(name, x):
        return (f'Solid {{ name "{name}" translation {x} 0 0.016 contactMaterial "caster" '
                f'children [ Shape {{ appearance {_mat(0.3, 0.3, 0.3)} geometry Sphere {{ radius 0.015 }} }} ] '
                f'boundingObject Sphere {{ radius 0.015 }} physics Physics {{ density -1 mass 0.01 }} }}')

    return (f'DEF SHELFSCAN Robot {{ translation {x0} {y0} 0 rotation 0 0 1 {yaw} name "shelfscan" '
            f'controller "{controller}" supervisor TRUE children [ '
            f'Pose {{ translation 0 0 0.06 children [ Shape {{ appearance {_mat(0.15, 0.3, 0.6)} '
            f'geometry Cylinder {{ height 0.08 radius 0.09 }} }} ] }} '
            f'Pose {{ translation 0 0 {(0.10 + CAM_Z) / 2} children [ Shape {{ appearance {_mat(0.6, 0.6, 0.6)} '
            f'geometry Box {{ size 0.03 0.03 {CAM_Z - 0.10} }} }} ] }} '
            f'{wheel("left", 0.08)} {wheel("right", -0.08)} {caster("front caster", 0.07)} {caster("rear caster", -0.07)} '
            f'GPS {{ name "gps" }} InertialUnit {{ name "imu" }} '
            f'DEF CAM Camera {{ name "camera" translation 0.05 0 {CAM_Z} width 640 height 480 '
            f'fieldOfView 1.0 recognition Recognition {{ maxRange 4 frameThickness 0 }} }} ] '
            f'boundingObject Pose {{ translation 0 0 0.06 children [ Cylinder {{ height 0.08 radius 0.09 }} ] }} '
            f'physics Physics {{ density -1 mass 1.0 }} }}')


def wbt(layout, version, controller="shelfscan_audit"):
    L = []
    a = L.append
    a(f"#VRML_SIM {version} utf8")
    a('WorldInfo { basicTimeStep 16 contactProperties [ ContactProperties { material2 "caster" coulombFriction [ 0 ] } ] }')
    a('Viewpoint { orientation -0.4 0.4 0.82 1.75 position 1.2 -4 6 }')
    a('Background { skyColor [ 0.75 0.8 0.85 ] }')
    a('DirectionalLight { direction -0.3 -0.6 -1 intensity 1.2 castShadows FALSE }')
    a('DirectionalLight { direction 0.2 0.8 -0.6 intensity 0.6 castShadows FALSE }')
    a(f'Solid {{ name "floor" children [ Shape {{ appearance {_mat(0.8, 0.8, 0.78)} '
      f'geometry Plane {{ size 30 30 }} }} ] boundingObject Plane {{ size 30 30 }} }}')
    for sid, s in layout["slots"].items():
        n = sid.replace("-", "_")
        # shelf platform + back panel (static); empty slots show bare platform/panel
        a(f'Solid {{ name "shelf_{n}" translation {s["x"]} {s["y"]} {PLATFORM_Z} '
          f'children [ Shape {{ appearance {_mat(0.55, 0.55, 0.6)} '
          f'geometry Box {{ size {SLOT_W-0.02} {SHELF_DEPTH} {PLATFORM_T} }} }} ] '
          f'boundingObject Box {{ size {SLOT_W-0.02} {SHELF_DEPTH} {PLATFORM_T} }} }}')
        a(f'Solid {{ name "panel_{n}" translation {s["x"]} {s["y"] - SHELF_DEPTH/2 + 0.01} {PLATFORM_Z + PANEL_H/2} '
          f'children [ Shape {{ appearance {_mat(0.35, 0.35, 0.4)} '
          f'geometry Box {{ size {SLOT_W-0.02} 0.02 {PANEL_H} }} }} ] '
          f'boundingObject Box {{ size {SLOT_W-0.02} 0.02 {PANEL_H} }} }}')
        if s["actual"]:
            r, g, b = SKUS[s["actual"]]
            a(f'DEF ITEM_{n} Solid {{ name "item_{n}" model "{s["actual"]}" '
              f'translation {s["x"]} {s["face_y"] - ITEM[1]/2 - 0.02} {s["z"]} '
              f'recognitionColors [ {r} {g} {b} ] children [ Shape {{ appearance {_mat(r, g, b)} '
              f'geometry Box {{ size {ITEM[0]} {ITEM[1]} {ITEM[2]} }} }} ] }}')
        if s["occluder"]:
            sign = 1 if s["occluder"] == "left" else -1      # image-left = world +x
            a(f'Solid {{ name "occluder_{n}" model "occluder" '
              f'translation {s["x"] + sign*0.14} {s["face_y"] + 0.30} {PLATFORM_Z + PLATFORM_T/2 + 0.175} '
              f'recognitionColors [ 0.5 0.5 0.5 ] children [ Shape {{ appearance {_mat(0.5, 0.5, 0.5)} '
              f'geometry Box {{ size 0.25 0.25 0.35 }} }} ] }}')
        # marker node used only for ground-truth lookup
        a(f'DEF SLOT_{n} Pose {{ translation {s["x"]} {s["face_y"] - ITEM[1]/2 - 0.02} {s["z"]} }}')
    x0, y0, yaw = layout["start_pose"]
    a(robot_node(x0, y0, yaw, controller))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="webots/worlds")
    ap.add_argument("--version", default="R2025a")
    ap.add_argument("--controller", default="shelfscan_audit",
                    help="shelfscan_audit or data_capture")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lay = make(a.seed)
    (out / "layout.json").write_text(json.dumps(lay, indent=1))
    (out / "warehouse.wbt").write_text(wbt(lay, a.version, a.controller))
    print(f"wrote {out/'warehouse.wbt'} and {out/'layout.json'} (seed {a.seed}, "
          f"{len(lay['slots'])} slots)")


if __name__ == "__main__":
    main()
