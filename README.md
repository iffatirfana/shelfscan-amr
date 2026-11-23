# ShelfScan AMR

Uncertainty-aware autonomous inventory-auditing robot with active visual
verification and a self-updating digital twin. Simulated in Webots.

Each shelf slot ends in one of four states: **OK, MISPLACED, EMPTY, UNVERIFIED**.
"Nothing detected" is never reported as EMPTY. Unresolved slots are re-observed
from a better viewpoint (active verification).

## Layout

```
shelfscan/            core logic (no Webots dependency)
  reasoning.py        observation + expected SKU -> slot status
  twin.py             digital twin (expected vs observed, report export)
  viewpoint.py        next-best-view rule for unresolved slots
  policies.py         V1 fixed patrol / V3 re-scan / V4 active verification
  geometry.py         camera projection, slot visibility and blocker side
  robot_utils.py      Webots helpers (go-to-pose driver, class list)
webots/tools/make_world.py          randomised world + layout.json
webots/tools/overlay_check.py       draw labels on images to verify them
webots/controllers/data_capture/    builds the YOLO dataset
webots/controllers/shelfscan_audit/ runs one policy with the trained detector
train/train_yolo.py   YOLOv8n fine-tune (+ optional INT8 export)
experiments/policy_harness.py       abstract logic check (NOT a Webots result)
experiments/aggregate_results.py    results/*.json -> comparison table
tests/                unit tests (python -m unittest discover -s tests)
```

## Status

| Part | State |
|---|---|
| Core logic, geometry, policies | Unit tested (22 tests pass) |
| `policy_harness.py` | Runs; numbers depend on stated assumptions, not evidence |
| `make_world.py`, controllers | Written without Webots available; **not yet run** |

## Run order (Webots R2023b or newer, Python 3 controller)

```bash
pip install -r requirements.txt
python -m unittest discover -s tests            # sanity check

# 1. Data: several seeds, data_capture controller
for s in 1 2 3 4 5 6; do
  python webots/tools/make_world.py --seed $s --out webots/worlds --controller data_capture
  SHELFSCAN_SEED=$s webots --batch --mode=fast webots/worlds/warehouse.wbt
done
python webots/tools/overlay_check.py dataset/images/train dataset/labels/train overlay 20
#   look at overlay/: boxes must sit on the items. If not, fix geometry.py first.

# 2. Train
python train/train_yolo.py --data dataset --epochs 60

# 3. Evaluate on UNSEEN seeds (e.g. 101..120), each policy
for s in $(seq 101 120); do
  python webots/tools/make_world.py --seed $s --out webots/worlds --controller shelfscan_audit
  for p in V1_fixed V3_rescan V4_active; do
    SHELFSCAN_POLICY=$p webots --batch --mode=fast webots/worlds/warehouse.wbt
  done
done
python experiments/aggregate_results.py results/
```

## Things to check on first run
- Camera frame convention (x forward, y left, z up) assumed in `geometry.py`.
- `TurtleBot3Burger` device names and wheel constants in `robot_utils.py`.
- `#VRML_SIM` version: pass `--version` to `make_world.py` to match your Webots.
- The slot thresholds in `reasoning.py` (`MIN_CONF`, `MIN_VISIBILITY`) should be
  set from validation data before the final runs.
- Report only numbers from step 3. Never report `policy_harness.py` output.
