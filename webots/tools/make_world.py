"""Generate a randomised warehouse world (.wbt) plus layout.json.

    python webots/tools/make_world.py --seed 3 --out webots/worlds/ --version R2025a

layout.json is the contract with the controllers: slot positions, expected SKUs,
the true contents, occluders, and per-slot viewpoint poses [x, y, yaw].

STATUS: generated text has NOT been opened in Webots by the author. Open it,
fix any version-specific field errors, and run webots/tools/overlay_check.py.
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
                    "left":   [x + LATERAL, face_y + STAND_OFF, -math.pi / 2],
                    "right":  [x - LATERAL, face_y + STAND_OFF, -math.pi / 2],
                    "close":  [x, face_y + CLOSE_OFF, -math.pi / 2],
                })
            order.append(sid)
    return dict(seed=seed, slot_size=list(ITEM), slots=slots, order=order,
                start_pose=[-0.8, slots["S0-0"]["face_y"] + STAND_OFF, -math.pi / 2])


def wbt(layout, version, controller="shelfscan_audit"):
    L = []
    a = L.append
    a(f"#VRML_SIM {version} utf8")
    a('EXTERNPROTO "webots://projects/objects/floors/protos/Floor.proto"')
    a('EXTERNPROTO "webots://projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"')
    a('EXTERNPROTO "webots://projects/robots/robotis/turtlebot/protos/TurtleBot3Burger.proto"')
    a('WorldInfo { coordinateSystem "ENU" basicTimeStep 16 }')
    a('Viewpoint { orientation 0 0.6 -0.8 0 position 1.2 -4 6 }')
    a('TexturedBackgroundLight {}')
    a('Floor { size 30 30 }')
    for sid, s in layout["slots"].items():
        n = sid.replace("-", "_")
        # shelf platform + back panel (static); empty slots show bare platform/panel
        a(f'Solid {{ name "shelf_{n}" translation {s["x"]} {s["y"]} {PLATFORM_Z} '
          f'children [ Shape {{ appearance PBRAppearance {{ baseColor 0.55 0.55 0.6 roughness 1 metalness 0 }} '
          f'geometry Box {{ size {SLOT_W-0.02} {SHELF_DEPTH} {PLATFORM_T} }} }} ] '
          f'boundingObject Box {{ size {SLOT_W-0.02} {SHELF_DEPTH} {PLATFORM_T} }} }}')
        a(f'Solid {{ name "panel_{n}" translation {s["x"]} {s["y"] - SHELF_DEPTH/2 + 0.01} {PLATFORM_Z + PANEL_H/2} '
          f'children [ Shape {{ appearance PBRAppearance {{ baseColor 0.35 0.35 0.4 roughness 1 metalness 0 }} '
          f'geometry Box {{ size {SLOT_W-0.02} 0.02 {PANEL_H} }} }} ] '
          f'boundingObject Box {{ size {SLOT_W-0.02} 0.02 {PANEL_H} }} }}')
        if s["actual"]:
            r, g, b = SKUS[s["actual"]]
            a(f'DEF ITEM_{n} Solid {{ name "item_{n}" model "{s["actual"]}" '
              f'translation {s["x"]} {s["face_y"] - ITEM[1]/2 - 0.02} {s["z"]} '
              f'recognitionColors [ {r} {g} {b} ] children [ Shape {{ appearance PBRAppearance '
              f'{{ baseColor {r} {g} {b} roughness 1 metalness 0 }} geometry Box {{ size {ITEM[0]} {ITEM[1]} {ITEM[2]} }} }} ] }}')
        if s["occluder"]:
            sign = 1 if s["occluder"] == "left" else -1      # image-left = world +x
            a(f'Solid {{ name "occluder_{n}" model "occluder" '
              f'translation {s["x"] + sign*0.14} {s["face_y"] + 0.30} {PLATFORM_Z + PLATFORM_T/2 + 0.175} '
              f'recognitionColors [ 0.5 0.5 0.5 ] children [ Shape {{ appearance PBRAppearance '
              f'{{ baseColor 0.5 0.5 0.5 roughness 1 metalness 0 }} geometry Box {{ size 0.25 0.25 0.35 }} }} ] }}')
        # invisible-to-camera marker node used only for ground-truth lookup
        a(f'DEF SLOT_{n} Pose {{ translation {s["x"]} {s["face_y"] - ITEM[1]/2 - 0.02} {s["z"]} }}')
    x0, y0, yaw = layout["start_pose"]
    a(f'DEF SHELFSCAN TurtleBot3Burger {{ translation {x0} {y0} 0 rotation 0 0 1 {yaw} '
      f'controller "{controller}" supervisor TRUE extensionSlot [ '
      f'GPS {{ name "gps" }} InertialUnit {{ name "imu" }} '
      f'DEF CAM Camera {{ name "camera" translation 0.05 0 0.12 width 640 height 480 '
      f'fieldOfView 1.0 recognition Recognition {{ maxRange 4 frameThickness 0 }} }} ] }}')
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
