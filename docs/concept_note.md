# ShelfScan AMR: concept note

Uncertainty-aware autonomous shelf auditing with active visual verification. TIH IIT Bombay Physical AI Hackathon. [Team name / members - add before submitting]. Code: [GITHUB LINK - add before submitting]. Video: [VIDEO LINK - add before submitting].

## 1. Problem

Warehouses and stores run on inventory records that drift away from what is physically on the shelf: items are picked from the wrong slot, put back in the wrong place, or run out without the system noticing. Manual cycle counts are slow and expensive, so they are done rarely, and errors can persist for weeks, causing failed picks, stock-outs and excess safety stock.

Mobile robots with cameras can count shelves automatically, but a fixed patrol has a quiet failure mode. When something stands in front of a slot (a box left in the aisle, a person, a pallet) or the view is poor, the detector simply sees nothing. A system that treats 'nothing detected' as 'empty' raises false restock alarms; a system that silently skips the slot produces an audit that looks complete but is not. Either way, staff stop trusting the robot.

## 2. Approach

ShelfScan AMR makes uncertainty explicit and acts on it. Every slot ends in one of four states: OK, MISPLACED, EMPTY or UNVERIFIED. The central rule is that absence of a detection is never evidence of absence: a slot is EMPTY only if it is clearly visible (estimated visibility >= 0.70) and the detector positively recognises an empty slot with confidence >= 0.60. Anything less is UNVERIFIED.

UNVERIFIED slots are then handled by active verification. The robot reads the side of the image on which the blocking object appeared and drives to a viewpoint on the opposite side, or closer to the shelf if nothing was blocking, and observes again (up to three extra looks per slot). A new decisive observation replaces UNVERIFIED in a digital twin that stores expected versus observed state, confidence, viewpoint and number of observations for every slot, and exports an audit report.

We compare three strategies with identical perception: V1 fixed patrol (one look per slot), V3 same-position rescan (one more look from the same viewpoint for unresolved slots), and V4 active verification (new viewpoints chosen from the evidence).

## 3. System architecture

Simulation: Webots R2025a. Each seed generates a warehouse with 3 shelf rows x 6 slots, random SKUs (four coloured product types), gaps, misplaced items and grey blocking objects in front of about 20% of slots. The robot is a differential-drive base with TurtleBot3-Burger wheel geometry, GPS and IMU for pose, and a 640x480 camera on a mast at item height. The world uses only Webots base nodes, so it loads offline.

Perception: a YOLOv8n detector (6 classes: 4 SKUs, slot_empty, blocker) fine-tuned on 864 camera frames captured automatically in Webots from 12 training seeds, with labels projected from simulator geometry and checked visually. Validation: mAP50 0.991, mAP50-95 0.952 (15 epochs, CPU, 320 px).

Reasoning and control: slot regions are found by projecting the known shelf map with the robot pose; slot content comes only from the detector. A reasoning module applies the confidence and visibility gates; the digital twin aggregates observations; a policy module (V1/V3/V4) and a next-best-view selector decide where to look; a go-to-pose controller drives the robot, routing between aisles via the aisle end. The reasoning, twin, policy and viewpoint code is pure Python, unit-tested and independent of the simulator.

## 4. Results (real Webots runs)

Evaluation protocol: seeds 1-12 generated the training frames, seeds 13-14 were used only to debug the pipeline, and seeds 101-130 (30 warehouses, 540 slots per strategy) were used only for this evaluation. Each strategy ran as a separate Webots process per seed, with the same detector and thresholds. Ground truth from the world file is used only for scoring. Mean ± std over seeds:

Webots R2025a, 30 unseen seeds (101-130), 18 slots per seed, mean ± std over seeds.

| Strategy | Runs | Slot accuracy | UNVERIFIED rate | False-EMPTY rate | Discrepancies caught | False alarms / run | Audit time (sim s) | Frames / run |
|---|---|---|---|---|---|---|---|---|
| Fixed patrol (V1) | 30 | 80.7 ± 9.0% | 19.3 ± 9.0% | 0.0 ± 0.0% | 69/88 | 0.00 ± 0.00 | 206 ± 0 | 18.0 ± 0.0 |
| Same-position rescan (V3) | 30 | 80.7 ± 9.0% | 19.3 ± 9.0% | 0.0 ± 0.0% | 69/88 | 0.00 ± 0.00 | 343 ± 45 | 21.5 ± 1.6 |
| Active verification (V4) | 30 | 98.5 ± 3.5% | 1.5 ± 3.5% | 0.0 ± 0.0% | 86/88 | 0.00 ± 0.00 | 354 ± 56 | 22.0 ± 2.5 |

Slot accuracy counts UNVERIFIED as wrong. Active verification reached 98.5% slot accuracy with 1.5% UNVERIFIED, against 80.7% and 19.3% for the fixed patrol and 80.7% and 19.3% for same-position rescan. The cost is audit time: 354 s versus 206 s simulated. False EMPTY reports: 0, 0 and 0 respectively.

## 5. What we learned

Looking again only helps from a different place. Same-position rescan gave the same final status as the fixed patrol on 540 of 540 slots while adding 137 s per audit: in a static scene the same viewpoint returns the same image. Active verification spent a similar amount of extra time (148 s) but chose viewpoints on the far side of the blocker, and resolved most of the slots the patrol could not.

The conservative rule did not cost correctness. No strategy produced a false EMPTY or a false alarm, and every slot that received a decision was correct (100% decided accuracy for all three). The remaining errors are explicit UNVERIFIED slots, which a human can check. Active verification identified 86 of 88 real discrepancies, against 69 for the fixed patrol.

Where it still fails: 8 slots stayed UNVERIFIED (8 after all four viewpoints, 8 behind blockers). In 6 of them the detector already had the right label, but the 0.70 visibility gate held the decision back; tuning this gate on validation data is the obvious next step. During development we also found that the side viewpoints first faced straight ahead, which put the slot at the image edge so that re-looks never helped; aiming them at the slot, recapturing data and retraining fixed this before the evaluation above.

## 6. Expected impact

An audit robot is only useful if its reports can be acted on without re-checking. By refusing to guess and by spending extra effort only on the slots it could not resolve, ShelfScan aims to give managers a short, trustworthy exception list (missing and misplaced items) plus an explicit list of slots that still need a human look. Expected benefits are fewer false restock tasks, faster and more frequent counts than manual cycle counting, and audit coverage that is measured rather than assumed. The approach is hardware-agnostic: the decision logic transfers unchanged to any camera-equipped mobile base.

## 7. Path to reality and limits

Deployment path: a ROS 2 differential-drive base with LiDAR localisation on a shelf map (replacing GPS), an RGB camera on a mast, and an edge computer running the INT8-exported YOLOv8n; fine-tune on real shelf images and pilot on one aisle against a manual count. Current limits: simplified scene (coloured boxes, one shelf level, static world, ideal pose), thresholds not yet tuned on validation data, few truly empty slots in evaluation, and no hardware measurements yet.
