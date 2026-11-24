"""Save frames from the overview camera, then quit.

Env: SHELFSCAN_SNAPSHOT  output path. With SHELFSCAN_SNAPSHOT_AT it must
                         contain "{}", which is filled with the shot index.
     SHELFSCAN_SNAPSHOT_AT  comma-separated simulation step numbers to shoot
                         at. Default: a single shot at step 10.

The step numbers are how this static camera stays in step with a controller
running on another robot (e.g. lift_shots raising the lift): simulation time is
deterministic, so a fixed step count is a reliable rendezvous.
"""
import os

from controller import Supervisor

r = Supervisor()
step = int(r.getBasicTimeStep())
cam = r.getDevice("overview")
cam.enable(step)
out = os.environ.get("SHELFSCAN_SNAPSHOT", "overview.png")   # "{}" -> shot index
at = os.environ.get("SHELFSCAN_SNAPSHOT_AT")
shots = [int(v) for v in at.split(",")] if at else [10]

n = 0
for i in range(max(shots) + 40):     # a short tail so the other controller
                                     # can finish its own reporting
    if r.step(step) == -1:
        break
    if i in shots:
        path = out.format(n) if "{}" in out else out
        cam.saveImage(path, 95)
        print(f"shot {n} at step {i} -> {path}", flush=True)
        n += 1
r.simulationQuit(0)
