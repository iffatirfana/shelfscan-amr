"""Launch Webots on a world, wait for the controller to quit, kill on timeout.

    python webots/tools/run_webots.py webots/worlds/warehouse.wbt --timeout 600 [--env K=V ...]

Returns Webots' exit code (124 on timeout). Uses WEBOTS_HOME or the default
Windows install path. Rendering stays on (the camera needs it)."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def webots_exe():
    home = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")
    for c in (Path(home) / "msys64/mingw64/bin/webots.exe", Path(home) / "webots"):
        if c.exists():
            return str(c)
    w = shutil.which("webots")
    if w:
        return w
    sys.exit("Webots not found: set WEBOTS_HOME")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--mode", default="fast")
    a = ap.parse_args()
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    cmd = [webots_exe(), "--batch", f"--mode={a.mode}", "--minimize", "--stdout", "--stderr", a.world]
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env)
    try:
        rc = p.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "webots-bin.exe"], capture_output=True)
        else:
            p.kill()
        rc = 124
    print(f"webots exit={rc} wall={time.time() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
