"""Launch the recording world with the demo overlays switched on.

    python webots/tools/record.py                 # paused, fresh tag
    python webots/tools/record.py --tag take7 --mode realtime

Why this exists rather than a documented run_webots.py command line: the
recording needs eleven environment variables, and as one pasted line that is
~600 characters. Losing the tail of it is silent and looks exactly like a bug
in the code - on 2026-11-26 a paste that dropped only its FINAL token,
HUD_DISPLAY=1, produced three panels reading "INSET IDLE" while HUD=1 (the
token before it) worked, which reads as "HUD is read but HUD_DISPLAY is not".
Every variable is built here instead, so there is no tail to lose.

It only ever launches a recording: HUD and HUD_DISPLAY are always set, so this
must not be used for a measured run - use experiments/run_batch.py, which
clears both and builds worlds without the Display devices.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORLDS = ROOT / "webots/worlds"
DEFAULT_WEIGHTS = ROOT / "models_v4/yolov8n_v4/weights/best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rec",
                    help="world tag: launches webots/worlds/_run_<name>.wbt "
                         "with _layout_<name>.json (default: rec)")
    ap.add_argument("--tag", default=None,
                    help="SHELFSCAN_TAG - names logs/shelfscan_audit_<tag>.log "
                         "and keeps this run's log off the last one's. "
                         "Defaults to --name.")
    ap.add_argument("--results", default=None,
                    help="SHELFSCAN_RESULTS (default: results_<tag>)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--policy", default="V4_active")
    ap.add_argument("--label", default="qr", choices=["qr", "barcode", ""])
    ap.add_argument("--classes", default="v3")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--mode", default="pause",
                    help="Webots mode: pause (default), realtime, fast")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--minimize", action="store_true",
                    help="let Webots minimise. Off by default: a recording "
                         "needs the window up, and a minimised one does not "
                         "render the 3D view for exportImage either.")
    a = ap.parse_args()

    tag = a.tag or a.name
    world = WORLDS / f"_run_{a.name}.wbt"
    layout = WORLDS / f"_layout_{a.name}.json"
    if not world.exists():
        sys.exit(f"no world at {world}\n"
                 f"build it first:  HUD_DISPLAY=1 python webots/tools/"
                 f"make_world.py --seed {a.seed} --qr --name {a.name} --demo")
    if not layout.exists():
        sys.exit(f"no layout at {layout} (rebuild the world)")
    if not Path(a.weights).exists():
        sys.exit(f"no weights at {a.weights}")

    env = {
        "SHELFSCAN_LAYOUT": str(layout),
        "SHELFSCAN_SEED": str(a.seed),
        "SHELFSCAN_TAG": tag,
        "SHELFSCAN_POLICY": a.policy,
        "SHELFSCAN_RESULTS": a.results or f"results_{tag}",
        "SHELFSCAN_WEIGHTS": str(Path(a.weights).resolve()),
        "SHELFSCAN_CLASSES": a.classes,
        "SHELFSCAN_LABEL": a.label,
        # The two that make it a recording. Set here, together, so neither can
        # go missing without the other - which is the failure this file exists
        # to prevent.
        "HUD": "1",
        "HUD_DISPLAY": "1",
    }
    print(f"world   {world}")
    print(f"log     logs/shelfscan_audit_{tag}.log")
    print("env     " + "  ".join(f"{k}={v}" for k, v in env.items()
                                 if k in ("HUD", "HUD_DISPLAY", "SHELFSCAN_TAG",
                                          "SHELFSCAN_RESULTS")))
    cmd = [sys.executable, str(ROOT / "webots/tools/run_webots.py"),
           str(world), "--mode", a.mode, "--timeout", str(a.timeout),
           "--env"] + [f"{k}={v}" for k, v in env.items()]
    if not a.minimize:
        cmd.append("--no-minimize")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    sys.exit(main())
