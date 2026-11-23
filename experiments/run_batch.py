"""Run many Webots simulations in parallel (data capture or policy evaluation).

    # training data from seeds 1..10
    python experiments/run_batch.py capture --seeds 1-10 --jobs 5
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
from make_world import make, wbt  # noqa: E402

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


def run_one(tag, seed, controller, env_extra, timeout, version):
    lay = make(seed)
    layout = WORLDS / f"_layout_{tag}.json"
    world = WORLDS / f"_run_{tag}.wbt"
    layout.write_text(json.dumps(lay, indent=1))
    world.write_text(wbt(lay, version, controller))
    env = [f"SHELFSCAN_LAYOUT={layout}", f"SHELFSCAN_SEED={seed}", f"SHELFSCAN_TAG={tag}"]
    env += [f"{k}={v}" for k, v in env_extra.items()]
    t0 = time.time()
    p = subprocess.run([sys.executable, str(ROOT / "webots/tools/run_webots.py"), str(world),
                        "--timeout", str(timeout), "--env", *env],
                       capture_output=True, text=True)
    wall = time.time() - t0
    rc = p.returncode
    print(f"[{tag}] exit={rc} wall={wall:.0f}s", flush=True)
    return dict(tag=tag, seed=seed, exit=rc, wall_s=round(wall, 1), **env_extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "eval"])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--results", default=str(ROOT / "results/runs"))
    ap.add_argument("--version", default="R2025a")
    a = ap.parse_args()

    jobs = []
    for s in seeds(a.seeds):
        if a.mode == "capture":
            jobs.append((f"cap{s}", s, "data_capture", {}))
        else:
            for pol in a.policies.split(","):
                jobs.append((f"{pol}_s{s}", s, "shelfscan_audit",
                             {"SHELFSCAN_POLICY": pol, "SHELFSCAN_RESULTS": a.results}))
    print(f"{len(jobs)} runs, {a.jobs} in parallel", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda j: run_one(*j, a.timeout, a.version), jobs))
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
