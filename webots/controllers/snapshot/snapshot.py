"""Save one frame from the overview camera to SHELFSCAN_SNAPSHOT, then quit."""
import os
from controller import Supervisor

r = Supervisor()
step = int(r.getBasicTimeStep())
cam = r.getDevice("overview")
cam.enable(step)
for _ in range(10):
    r.step(step)
cam.saveImage(os.environ.get("SHELFSCAN_SNAPSHOT", "overview.png"), 95)
r.simulationQuit(0)
