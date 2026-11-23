# ShelfScan AMR

Uncertainty-aware autonomous inventory-auditing robot with active visual
verification and a self-updating digital twin. Simulated in **Webots R2025a**.

Each shelf slot ends in one of four states: **OK, MISPLACED, EMPTY, UNVERIFIED**.
"Nothing detected" is never reported as EMPTY. Unresolved slots are re-observed
from a better viewpoint (active verification).

- Demo video: [VIDEO LINK - add before submitting]
- Slides + concept note (one PDF): `docs/ShelfScan_submission.pdf` (editable deck: `docs/ShelfScan_deck.pptx`)
- Video voice-over script: `docs/video_script.md`

## Results (real Webots runs)

<!-- RESULTS:START -->
Webots R2025a, 30 unseen seeds (101-130), 18 slots per seed, mean ± std over seeds.

| Strategy | Runs | Slot accuracy | UNVERIFIED rate | False-EMPTY rate | Discrepancies caught | False alarms / run | Audit time (sim s) | Frames / run |
|---|---|---|---|---|---|---|---|---|
| Fixed patrol (V1) | 30 | 80.7 ± 9.0% | 19.3 ± 9.0% | 0.0 ± 0.0% | 69/88 | 0.00 ± 0.00 | 206 ± 0 | 18.0 ± 0.0 |
| Same-position rescan (V3) | 30 | 80.7 ± 9.0% | 19.3 ± 9.0% | 0.0 ± 0.0% | 69/88 | 0.00 ± 0.00 | 343 ± 45 | 21.5 ± 1.6 |
| Active verification (V4) | 30 | 98.5 ± 3.5% | 1.5 ± 3.5% | 0.0 ± 0.0% | 86/88 | 0.00 ± 0.00 | 354 ± 56 | 22.0 ± 2.5 |

Detector: YOLOv8n, validation mAP50 0.991 / mAP50-95 0.952 (see models/train_results.csv).
<!-- RESULTS:END -->

Definitions (computed per run in `webots/controllers/shelfscan_audit/shelfscan_audit.py`):

- **Slot accuracy**: final slot status equals ground truth; UNVERIFIED counts as wrong.
- **UNVERIFIED rate**: slots left UNVERIFIED / all slots.
- **False-EMPTY rate**: slots reported EMPTY that were not empty / all non-empty slots.
- **Audit time**: simulated seconds from start of the audit to the final report.

Every number above comes from `results/results.csv`, which is written only by the
audit controller running inside Webots on seeds never used for training.
`experiments/policy_harness.py` is an abstract logic check; its output is not a result.

## Strategies compared

| Name | What it does |
|---|---|
| `V1_fixed` | Fixed patrol: one look per slot from the centre viewpoint. |
| `V3_rescan` | Fixed patrol, then one more look from the SAME viewpoint for each UNVERIFIED slot. |
| `V4_active` | Fixed patrol, then up to 3 extra looks per UNVERIFIED slot from a NEW viewpoint chosen from the evidence (opposite side of the blocker, else closer). |

All three use the same detector, thresholds (confidence >= 0.60, visibility >= 0.70) and digital twin.

## How to run

Requirements: Windows/Linux, Webots R2025a, Python 3.10+ (3.12 used here) on PATH.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests          # 28 unit tests

# One world, one run (camera needs rendering, so no --no-rendering):
python webots/tools/make_world.py --seed 1 --out webots/worlds --controller shelfscan_audit --version R2025a
python webots/tools/run_webots.py webots/worlds/warehouse.wbt --timeout 900 --env SHELFSCAN_POLICY=V4_active
#   -> results/<policy>_seed1.json and _slots.csv, controller log in logs/

# Full pipeline used for the reported numbers
python experiments/run_batch.py capture --seeds 1-12 --jobs 6          # training frames -> dataset/
python webots/tools/overlay_check.py dataset/images/train dataset/labels/train overlay 20   # look at the boxes
python train/train_yolo.py --data dataset --epochs 15 --imgsz 320 --device cpu
cp runs/shelfscan/weights/best.pt models/shelfscan_yolov8n.pt
python experiments/run_batch.py eval --seeds 101-130 --jobs 6          # 3 strategies per seed -> results/runs/
python experiments/aggregate_results.py results/runs results           # results.csv + summary.md
python docs/build_submission.py                                        # deck, concept note, PDF
```

`run_webots.py` finds Webots via `WEBOTS_HOME` (default `C:\Program Files\Webots`).
Webots on Windows does not forward controller output to the terminal, so each
controller logs to `logs/<controller>_<tag>.log` and quits the simulation on any
exception, so batch runs never hang.

## Layout

```
shelfscan/            core logic (no Webots dependency)
  reasoning.py        observation + expected SKU -> slot status
  twin.py             digital twin (expected vs observed, report export)
  viewpoint.py        next-best-view rule for unresolved slots
  policies.py         V1 fixed patrol / V3 re-scan / V4 active verification
  geometry.py         camera projection, slot visibility and blocker side
  robot_utils.py      Webots helpers (go-to-pose driver with aisle routing, logging)
webots/tools/make_world.py          randomised world (.wbt, base nodes only) + layout.json
webots/tools/run_webots.py          launch Webots with timeout
webots/tools/overlay_check.py       draw labels on images to verify them
webots/controllers/data_capture/    builds the YOLO dataset (labels from projected 3D geometry)
webots/controllers/shelfscan_audit/ runs one strategy with the trained detector and scores it
webots/controllers/snapshot/        renders overview images for the docs
train/train_yolo.py                 YOLOv8n fine-tune (+ optional INT8 export)
models/                             trained detector used for the results + its training log
experiments/run_batch.py            parallel Webots runs (capture / eval)
experiments/aggregate_results.py    results/runs/*.json -> results/results.csv + summary.md
experiments/policy_harness.py       abstract logic check (NOT a Webots result)
docs/                               deck, concept note, video script, figures
tests/                              unit tests
```

## What had to be fixed to make it run (first Webots contact)

- The `webots://` PROTOs (TurtleBot3Burger, Floor, TexturedBackgroundLight) never
  downloaded on the test machine, so the world had no robot. The world now uses
  only base nodes: a differential-drive robot with TurtleBot3-Burger wheel
  geometry and the same device names, a plain floor, lights and background.
- The camera at 0.12 m could not see the 0.30 m items on a 0.32 m shelf; it now sits
  on a mast at item height (0.47 m).
- The robot would have driven through the next shelf row when changing aisles;
  the driver now routes via the aisle end.
- Webots Recognition boxes were loose; training labels are now projected from the
  known 3D geometry of the current row (checked with `overlay_check.py`).
- Side viewpoints kept the camera facing straight ahead, so the slot sat at the
  image edge; they now turn to look at the slot. The dataset was recaptured and
  the detector retrained after this fix.

## Limits

Simplified scene (coloured boxes as SKUs, one shelf level, static world, GPS pose),
thresholds are design defaults (not tuned on validation data), few truly empty
slots per seed, no real-robot test yet.
