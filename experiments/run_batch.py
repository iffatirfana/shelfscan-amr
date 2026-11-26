"""Run many Webots simulations in parallel (data capture or policy evaluation).

    # training data from seeds 1..10
    python experiments/run_batch.py capture --seeds 1-10 --jobs 5
    # v3 (4-class) data from the QR world, seeds 13-14 held out as the dev set
    python experiments/run_batch.py capture --seeds 1-12 --val-seeds 13-14
        --label qr --classes v3 --vp-keep 2 --vp-keep-empty 4 --out datasets_v3
    # evaluate the three policies on unseen seeds 101..110
    python experiments/run_batch.py eval --seeds 101-110 --jobs 6

Each run gets its own world file (webots/worlds/_run_<tag>.wbt) and layout
(webots/worlds/_layout_<tag>.json, passed to the controller via
SHELFSCAN_LAYOUT), so instances never overwrite each other's inputs.
Every run's wall time and Webots exit code go to logs/batch_<mode>.csv.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webots/tools"))
sys.path.insert(0, str(ROOT))
from make_world import make, wbt  # noqa: E402
from shelfscan.robot_utils import slot_matches  # noqa: E402

WORLDS = ROOT / "webots/worlds"
POLICIES = ["V1_fixed", "V3_rescan", "V4_active"]


def seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_one(tag, seed, controller, env_extra, timeout, version, no_render=False,
            label=None):
    lay = make(seed, label=label)
    layout = WORLDS / f"_layout_{tag}.json"
    world = WORLDS / f"_run_{tag}.wbt"
    layout.write_text(json.dumps(lay, indent=1))
    world.write_text(wbt(lay, version, controller))
    env = [f"SHELFSCAN_LAYOUT={layout}", f"SHELFSCAN_SEED={seed}", f"SHELFSCAN_TAG={tag}"]
    env += [f"{k}={v}" for k, v in env_extra.items()]
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "webots/tools/run_webots.py"), str(world),
           "--timeout", str(timeout)]
    if no_render:
        cmd.append("--no-rendering")
    cmd += ["--env", *env]
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    rc = p.returncode
    print(f"[{tag}] exit={rc} wall={wall:.0f}s", flush=True)
    return dict(tag=tag, seed=seed, exit=rc, wall_s=round(wall, 1), **env_extra)


def eval_seedwise(a):
    """Run seeds one at a time, all strategies of a seed together. Each run
    writes its own JSON/CSV the moment it finishes, so stopping early leaves
    every completed seed scored by all three strategies."""
    res = Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    pols = a.policies.split(",")
    t_start = time.time()
    done = []
    for i, seed in enumerate(seeds(a.seeds), 1):
        jobs = [(f"{p}_s{seed}", seed, "shelfscan_audit",
                 {"SHELFSCAN_POLICY": p, "SHELFSCAN_RESULTS": a.results,
                  "SHELFSCAN_WEIGHTS": a.weights,
                  "SHELFSCAN_CLASSES": a.classes,
                  "SHELFSCAN_LABEL": a.label or "",
                  "SHELFSCAN_NBV_LOG": a.nbv_log or "",
                  "SHELFSCAN_NBV_MODEL": a.nbv_model or "",
                  # a measured run is never recorded: clear both overlay
                  # flags explicitly so an exported HUD=1 / HUD_DISPLAY=1 in
                  # the shell cannot reach the controller. The worlds built
                  # here never carry the Display device either, because wbt()
                  # is called without display=True.
                  "HUD": "", "HUD_DISPLAY": ""}) for p in pols]
        t0 = time.time()
        with ThreadPoolExecutor(len(jobs)) as ex:
            list(ex.map(lambda j: run_one(*j, a.timeout, a.version,
                                          a.no_rendering, a.label), jobs))
        for j in jobs:
            for pth in (WORLDS / f"_run_{j[0]}.wbt", WORLDS / f"_layout_{j[0]}.json",
                        WORLDS / f"._run_{j[0]}.wbproj"):
                pth.unlink(missing_ok=True)
        wall = time.time() - t0
        done.append(wall)
        parts = []
        for p in pols:
            f = res / f"{p}_seed{seed}.json"
            if f.exists():
                d = json.loads(f.read_text())
                parts.append(f"{p.split('_')[0]} acc={d['slot_accuracy']*100:5.1f}% "
                             f"unv={d['unverified']:2d} t={d['audit_time_s']:.0f}s")
            else:
                parts.append(f"{p.split('_')[0]} FAILED")
        avg = sum(done) / len(done)
        left = (len(seeds(a.seeds)) - i) * avg
        print(f"[seed {seed}] " + " | ".join(parts)
              + f" | wall={wall:.0f}s avg={avg:.0f}s eta={left/60:.0f}min",
              flush=True)
    print(f"all seeds done in {(time.time() - t_start)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "eval"])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--results", default=str(ROOT / "results_v2/runs"))
    ap.add_argument("--out", default=str(ROOT / "dataset_v2"),
                    help="capture mode: dataset directory")
    ap.add_argument("--version", default="R2025a")
    ap.add_argument("--weights", default=str(ROOT / "models_v2/shelfscan_yolov8n.pt"))
    ap.add_argument("--no-rendering", action="store_true")
    ap.add_argument("--val-seeds", default=None,
                    help="capture mode: these seeds go to images/val, the rest "
                         "to images/train (default: random 15%% per frame)")
    ap.add_argument("--classes", choices=["v2", "v3"], default="v2",
                    help="capture mode: v2 = the 4 SKUs + slot_empty + "
                         "blocker; v3 = cardboard_box + label + slot_empty + "
                         "blocker, for the QR/barcode world")
    ap.add_argument("--vp-keep", type=int, default=None,
                    help="capture mode: keep only this many of the four "
                         "viewpoints per slot, drawn at random")
    ap.add_argument("--capture-only", choices=["empty_blocked"],
                    help="capture mode: drive to only the slots of this kind, "
                         "so a rare case can be collected over many seeds "
                         "without paying for all 30 slots of each")
    ap.add_argument("--empty-vis", type=float,
                    help="capture mode: how much of an empty slot must be "
                         "visible before it is labelled slot_empty "
                         "(default 0.7; a carton is kept down to 0.35)")
    ap.add_argument("--vp-keep-empty", type=int, default=None,
                    help="capture mode: ... but this many for a slot with no "
                         "carton, so the rare slot_empty class is not thinned")
    ap.add_argument("--label", choices=["qr", "barcode"], default=None,
                    help="MVP: build the world with machine-readable carton "
                         "labels and let the audit read them")
    ap.add_argument("--qr", action="store_true", help="shorthand for --label qr")
    ap.add_argument("--nbv-model", default=None,
                    help="V5: joblib next-best-view model to score candidates")
    ap.add_argument("--nbv-log", default=None,
                    help="dev: append next-best-view candidate rows to this CSV")
    ap.add_argument("--seedwise", action="store_true",
                    help="eval mode: finish all strategies of one seed before "
                         "starting the next, so an early stop still leaves a "
                         "fair like-for-like comparison")
    a = ap.parse_args()
    a.label = a.label or ("qr" if a.qr else None)

    if a.mode == "eval" and a.seedwise:
        return eval_seedwise(a)

    jobs = []
    val = set(seeds(a.val_seeds)) if a.val_seeds else set()
    for s in seeds(a.seeds):
        if a.mode == "capture":
            # With --capture-only, most seeds hold no slot of the wanted kind
            # at all (43 empty-and-blocked slots are spread over 29 of seeds
            # 15-74). Starting Webots for those costs a minute and saves no
            # frame, so the layout is generated here and the seed dropped.
            if a.capture_only and not any(
                    slot_matches(sl, a.capture_only)
                    for sl in make(s, label=a.label)["slots"].values()):
                continue
            # Split BY SEED when --val-seeds is given: a seed's 30 slots are
            # one layout, so splitting frames at random would put near-copies
            # of the same carton on both sides of the split.
            env = {"SHELFSCAN_OUT": a.out, "SHELFSCAN_CLASSES": a.classes,
                   "SHELFSCAN_VP_KEEP": str(a.vp_keep) if a.vp_keep else "",
                   "SHELFSCAN_VP_KEEP_EMPTY": (str(a.vp_keep_empty)
                                               if a.vp_keep_empty else ""),
                   "SHELFSCAN_CAPTURE_ONLY": a.capture_only or "",
                   "SHELFSCAN_EMPTY_VIS": (str(a.empty_vis)
                                           if a.empty_vis else ""),
                   "SHELFSCAN_SPLIT": ("val" if s in val else "train") if val else ""}
            jobs.append((f"cap{s}", s, "data_capture", env))
        else:
            for pol in a.policies.split(","):
                jobs.append((f"{pol}_s{s}", s, "shelfscan_audit",
                             {"SHELFSCAN_POLICY": pol,
                              "SHELFSCAN_RESULTS": a.results,
                              "SHELFSCAN_WEIGHTS": a.weights,
                              "SHELFSCAN_CLASSES": a.classes,
                              "SHELFSCAN_LABEL": a.label or "",
                              "HUD": "", "HUD_DISPLAY": ""}))
    print(f"{len(jobs)} runs, {a.jobs} in parallel", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda j: run_one(*j, a.timeout, a.version,
                                             a.no_rendering, a.label), jobs))
    (ROOT / "logs").mkdir(exist_ok=True)
    log = ROOT / f"logs/batch_{a.mode}.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        if new:
            w.writeheader()
        w.writerows(rows)
    for j in jobs:
        for ext in (".wbt", ".json"):
            for p in (WORLDS / f"_run_{j[0]}{ext}", WORLDS / f"_layout_{j[0]}{ext}"):
                p.unlink(missing_ok=True)
        (WORLDS / f"._run_{j[0]}.wbproj").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
