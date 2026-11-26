# ShelfScan AMR

Uncertainty-aware autonomous inventory-auditing robot with active visual
verification and a self-updating digital twin. Simulated in **Webots R2025a**.

Each shelf slot ends in one of four states: **OK, MISPLACED, EMPTY, UNVERIFIED**.
"Nothing detected" is never reported as EMPTY. Unresolved slots are re-observed
from a better viewpoint (active verification).

**Everything here is simulation.** No part of this has run on a physical robot.

## The simulated warehouse

A narrow aisle between **two tall racking rows facing each other**, 3 levels x 5
bays on each side = **30 slots per seed**. Blue uprights, orange beams, concrete
floor with a yellow lane line, steel-truss ceiling. (`make_world --demo` repaints
the racking galvanised grey and closes the bay-mouth pallet's back for the
recording. It is off everywhere else on purpose: v4 was trained against the
blue-and-orange world, so those visuals change what the detector sees - see
`DEMO_PALETTE` in `webots/tools/make_world.py`.)

- **Contents are plain cardboard cartons** in five size classes (small, medium,
  large, tall, wide-flat), with random in-bay offset and rotation; some sit on
  wooden pallets, some bays hold two cartons.
- **There is no colour cue.** A carton's SKU is carried only by a large
  black-and-white pattern printed on a white shipping label: `sku_A` a solid
  square, `sku_B` horizontal bars, `sku_C` a 3x3 dot grid, `sku_D` vertical bars.
  Carton size is drawn independently of the SKU, so neither size nor shade is a
  shortcut. The detector has to work from shape and print.
- **A blocker** is an *unlabelled* carton or a pallet stood at the bay mouth; an
  **empty slot** is a bare bay. Both are told apart by shape and texture, not colour.
- **The robot is a scissor-lift AMR**: a low grey deck with yellow side skirts
  cut away over the drive wheels, a yellow scissor stack, and a grey top bar with
  white LED strips and a camera unit at each end. The 640x480 camera rides a
  powered lift that really raises it to each shelf level (0.56 / 1.21 / 1.86 m).
  Drive is differential (TurtleBot3-Burger wheel geometry) with GPS + IMU.
- **The scissor stack is a real linkage**, not a decoration: six stages of crossed
  bars in two parallel stacks, pinned at every stage end and at each crossing,
  with one sliding end on a chassis rail and one under the platform, and no
  vertical member. `shelfscan/scissor.py` solves the bar poses from the lift
  height on every control step - `theta = asin(stack_height / (n * L))` - so the
  X's fold nearly flat (11 deg) at the lowest level and extend (70 deg) at the
  highest. The links are visual-only `Pose` nodes with no bounding object, so
  they cannot change the dynamics or the audit result.
- **Scan order:** the audit completes one shelf side entirely, top level down and
  along the full length of each level, including any V4 re-verification detours,
  before crossing the aisle to the other side. The sides are never interleaved.

Team: VOLTIC D. NERDS, Chennai Institute of Technology (CIT) — Iffat Irfana M, Arunachalam R, Aravind K

- Code: https://github.com/iffatirfana/shelfscan-amr
- Demo video: [VIDEO LINK - add before submitting]
- Slides + concept note (one PDF): `docs/ShelfScan_submission.pdf` (editable deck: `docs/ShelfScan_deck.pptx`)
- Video voice-over script: `docs/video_script.md`

## Results (real Webots runs)

<!-- RESULTS:START -->
Webots R2025a, 30 unseen seeds (101-130), 30 slots per seed, mean ± std over seeds.

| Strategy | Runs | Slot accuracy | UNVERIFIED rate | False-EMPTY rate | Discrepancies caught | False alarms / run | Audit time (sim s) | Frames / run |
|---|---|---|---|---|---|---|---|---|
| Fixed patrol (V1) | 30 | 89.0 ± 5.7% | 10.9 ± 5.4% | 0.1 ± 0.6% | 134/152 | 0.03 ± 0.18 | 223 ± 0 | 30.0 ± 0.0 |
| Same-position rescan (V3) | 30 | 89.0 ± 5.7% | 10.8 ± 5.5% | 0.2 ± 0.9% | 134/152 | 0.07 ± 0.25 | 254 ± 21 | 33.3 ± 1.6 |
| Active verification (V4) | 30 | 98.4 ± 2.1% | 1.3 ± 2.0% | 0.1 ± 0.6% | 149/152 | 0.07 ± 0.25 | 284 ± 40 | 36.2 ± 3.5 |

Detector: YOLOv8n, validation mAP50 0.986 / mAP50-95 0.938 (see `models_v2/train_results.csv`). Per class recall: sku_A 0.914, sku_B 0.974, sku_C 0.956, sku_D 0.957, slot_empty 0.909, blocker 0.949.
<!-- RESULTS:END -->

Definitions (computed per run in `webots/controllers/shelfscan_audit/shelfscan_audit.py`):

- **Slot accuracy**: final slot status equals ground truth; UNVERIFIED counts as wrong.
- **UNVERIFIED rate**: slots left UNVERIFIED / all slots.
- **False-EMPTY rate**: slots reported EMPTY that were not empty / all non-empty slots.
- **Audit time**: simulated seconds from start of the audit to the final report.

Every number above comes from `results_v2/results.csv`, which is written only by the
audit controller running inside Webots on seeds never used for training.
`experiments/policy_harness.py` is an abstract logic check; its output is not a result.

## Extension: machine-readable labels (MVP)

A **separate experiment**, not the main result. In a variant world each carton
carries a QR code or a Code128 barcode instead of the printed SKU pattern, and
the robot reads it off the camera frame. Codes are matched to slots through the
map and the robot's pose, never ground truth. A decoded code decides the slot;
a slot with no readable code stays UNVERIFIED, because in label mode the
detector's SKU classes are deliberately not trusted — it was trained on printed
patterns and only guesses once the cartons carry codes.

V4_active, 10 evaluation seeds (101–110), 30 slots each, thresholds unchanged:

| | QR | Code128 |
|---|---|---|
| Slot accuracy | 89.0 ± 4.2 % | 85.7 ± 8.7 % |
| Decided accuracy | 99.6 ± 1.1 % | 99.6 ± 1.3 % |
| UNVERIFIED | 3.2 ± 1.3 slots/run | 4.2 ± 2.5 slots/run |
| Decode rate (mean of 10 per-seed rates) | 87.6 ± 4.4 % | 83.9 ± 9.0 % |
| Decode rate (pooled over all cartons) | 87.6 % (227/259) | 83.8 % (217/259) |
| **Wrong payloads** | **0** | **0** |
| Discrepancies caught | 48/52 | 46/52 |
| False alarms | 1 | 0 |
| Audit time | 358.7 ± 55.9 s | 460.7 ± 68.1 s |

The two label types are **not separated** by this evidence: the paired
per-seed decode difference is +3.7 pp with a 95 % CI of −1.1 to +8.4, and QR
decoded more on 4 of 10 seeds, fewer on 2 and the same on 4. Code128 has less
pixel headroom by construction (2.0–4.1 px per module against 3.7–6.2), which
is consistent with its wider spread, but 10 seeds do not establish a winner.

Slot accuracy is below the 98.4 % of the printed-pattern world above because a
carton standing behind a pallet has an unreadable code and there is no SKU
fallback by design. That figure comes from a different world, a different
decision path and the full 30-seed evaluation, so the two are **not comparable
line by line** and are kept in separate tables throughout.

```bash
python webots/tools/make_world.py --seed 101 --label qr        # or --label barcode
python experiments/run_batch.py eval --seeds 101-110 --policies V4_active     --seedwise --no-rendering --label qr --results results_qr/runs     --weights <abs path>/models_v2/shelfscan_yolov8n.pt
python experiments/summarise_labels.py results_qr/runs results_bc/runs
```

Full write-up, including why Code128 cannot reach 8 px per bar at 640×480:
[docs/MVP_LABELS.md](docs/MVP_LABELS.md). Needs `qrcode[pil]`, `python-barcode`
and `zxing-cpp`.

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

# One world, one run. --no-rendering only disables the main window; camera
# devices still render offscreen. Verified bit-identical and ~17% faster.
python webots/tools/make_world.py --seed 101 --out webots/worlds --controller shelfscan_audit --version R2025a
python webots/tools/run_webots.py webots/worlds/warehouse.wbt --timeout 900 --no-rendering     --env SHELFSCAN_POLICY=V4_active SHELFSCAN_WEIGHTS=models_v2/shelfscan_yolov8n.pt
#   -> results_v2/<policy>_seed101.json and _slots.csv, controller log in logs/

# Full pipeline used for the reported numbers
python experiments/run_batch.py capture --seeds 1-12 --jobs 6          # training frames -> dataset_v2/
python webots/tools/overlay_check.py dataset_v2/images/train dataset_v2/labels/train overlay 20
python train/train_yolo.py --data dataset_v2 --epochs 30 --imgsz 320 --device cpu     --project runs --name shelfscan_v2
cp runs/shelfscan_v2/weights/best.pt models_v2/shelfscan_yolov8n.pt
cp runs/shelfscan_v2/results.csv     models_v2/train_results.csv

# Evaluation: one seed at a time, all three strategies before the next seed, so
# stopping early still leaves a fair like-for-like comparison.
python experiments/run_batch.py eval --seeds 101-130 --seedwise --no-rendering     --results results_v2/runs --weights models_v2/shelfscan_yolov8n.pt
python experiments/aggregate_results.py results_v2/runs results_v2     # results.csv + summary.md
python webots/tools/make_shots.py --seed 101                           # docs/img/*.png
python webots/tools/make_shots.py --seed 101 --lift                    # docs/figures/scissor_L*.png
python webots/tools/make_shots.py --seed 101 --base                    # docs/figures/scissor_base.png
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
  scissor.py          scissor-link kinematics (visual only) + the field driver
  robot_utils.py      Webots helpers (go-to-pose driver, camera lift, logging)
webots/tools/make_world.py          randomised world (.wbt, base nodes only) + layout.json
webots/tools/run_webots.py          launch Webots with timeout
webots/tools/overlay_check.py       draw labels on images to verify them
webots/controllers/data_capture/    builds the YOLO dataset (labels from projected 3D geometry)
webots/controllers/shelfscan_audit/ runs one strategy with the trained detector and scores it
webots/controllers/snapshot/        renders overview images for the docs
webots/tools/make_shots.py          renders the docs/img and docs/figures images
webots/controllers/world_check/     dev-only: loads the world, checks the lift, saves frames
webots/controllers/lift_shots/      dev-only: holds each level and checks the link placement
docs/figures/                       the robot at each lift height, and a close view of its base
train/train_yolo.py                 YOLOv8n fine-tune (+ optional INT8 export)
models_v2/                          trained detector used for the results + its training log
models/                             earlier detector from the colour-coded world (superseded)
experiments/run_batch.py            parallel Webots runs (capture / eval)
experiments/aggregate_results.py    results_v2/runs/*.json -> results_v2/results.csv + summary.md
experiments/policy_harness.py       abstract logic check (NOT a Webots result)
docs/                               deck, concept note, video script, figures
tests/                              unit tests
```

## What had to be fixed to make it run (first Webots contact)

- The `webots://` PROTOs (TurtleBot3Burger, Floor, TexturedBackgroundLight) never
  downloaded on the test machine, so the world had no robot. The world now uses
  only base nodes: a differential-drive robot with TurtleBot3-Burger wheel
  geometry and the same device names, a plain floor, lights and background.
- The camera at 0.12 m could not see items on the shelf; it now rides a powered
  lift that raises it to the level being scanned (0.56 / 1.21 / 1.86 m).
- With three parallel rows the robot would have driven through a rack when
  changing aisles, so the driver detoured via the aisle end. The world is now a
  single aisle between two facing rows, where every viewpoint is reachable in a
  straight line, and that detour has been removed.
- `run_webots.py` claimed rendering had to stay on for the camera. Measured on
  seed 101: `--no-rendering` gives a bit-identical run JSON and per-slot CSV and
  is ~17% faster. The claim was wrong and the flag is now used for batches.
- Blockers were briefly excluded from training labels by the image-edge filter
  that (correctly) drops truncated cartons. Blockers sit 0.20 m off the bay
  centre only 0.70 m from the camera, so any blocker 0.45 m or wider always
  overruns the frame; filtering them cost 12x the blocker boxes. A truncated
  carton has an ambiguous SKU, a truncated blocker does not, so the asymmetry
  is deliberate.
- Webots Recognition boxes were loose; training labels are now projected from the
  known 3D geometry of the current row (checked with `overlay_check.py`).
- Side viewpoints kept the camera facing straight ahead, so the slot sat at the
  image edge; they now turn to look at the slot. The dataset was recaptured and
  the detector retrained after this fix.

## Limits

**Simulation only — nothing has run on a physical robot.** The cartons and their
labels are rendered geometry with flat printed patterns, not photographs of real
stock; the world is static; localisation is ideal (simulator GPS, not LiDAR/AMCL);
and although the scissor linkage really folds and extends, it is built on small
TurtleBot3-class running gear, so the robot's proportions are not those of a real
lift truck.

The two decision thresholds (confidence >= 0.60, visibility >= 0.70) are **design
defaults, not tuned** on validation data. 6 of the 12 slots V4 left UNVERIFIED had
the right label already, held back by the 0.70 visibility gate, so tuning it is
the obvious next step.

Truly empty slots are rare (65 of 900 across the evaluation seeds), which also
makes `slot_empty` the weakest detector class (recall 0.909, 141 training
instances). Wrong decisions are rare but not zero: 1-2 false EMPTY per strategy
across 900 slots.
