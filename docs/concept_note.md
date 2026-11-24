# ShelfScan AMR: concept note

Uncertainty-aware autonomous shelf auditing with active visual verification. TIH IIT Bombay Physical AI Hackathon. VOLTIC D. NERDS, Chennai Institute of Technology (CIT) — Iffat Irfana M, Arunachalam R, Aravind K. Code: https://github.com/iffatirfana/shelfscan-amr. Video: [VIDEO LINK - add before submitting].

## 1. Problem

Warehouses and stores run on inventory records that drift away from what is physically on the shelf: items are picked from the wrong slot, put back in the wrong place, or run out without the system noticing. Manual cycle counts are slow and expensive, so they are done rarely, and errors can persist for weeks, causing failed picks, stock-outs and excess safety stock.

Mobile robots with cameras can count shelves automatically, but a fixed patrol has a quiet failure mode. When something stands in front of a slot (a box left in the aisle, a person, a pallet) or the view is poor, the detector simply sees nothing. A system that treats 'nothing detected' as 'empty' raises false restock alarms; a system that silently skips the slot produces an audit that looks complete but is not. Either way, staff stop trusting the robot.

## 2. Approach

ShelfScan AMR makes uncertainty explicit and acts on it. Every slot ends in one of four states: OK, MISPLACED, EMPTY or UNVERIFIED. The central rule is that absence of a detection is never evidence of absence: a slot is EMPTY only if it is clearly visible (estimated visibility >= 0.70) and the detector positively recognises an empty slot with confidence >= 0.60. Anything less is UNVERIFIED.

UNVERIFIED slots are then handled by active verification. The robot reads the side of the image on which the blocking object appeared and drives to a viewpoint on the opposite side, or closer to the shelf if nothing was blocking, and observes again (up to three extra looks per slot). A new decisive observation replaces UNVERIFIED in a digital twin that stores expected versus observed state, confidence, viewpoint and number of observations for every slot, and exports an audit report.

We compare three strategies with identical perception: V1 fixed patrol (one look per slot), V3 same-position rescan (one more look from the same viewpoint for unresolved slots), and V4 active verification (new viewpoints chosen from the evidence).

## 3. System architecture

Simulation: Webots R2025a. Each seed generates a narrow aisle between two facing racking rows, 3 levels of 5 bays on each side, holding plain cardboard cartons in five size classes, with gaps, misplaced items and blocking objects in front of about 20% of slots. There is no colour cue: a carton's SKU is carried only by a large black-and-white pattern printed on a white shipping label, and a blocker is an unlabelled carton or a pallet stood at the bay mouth, so the detector must work from shape and print rather than colour. The robot is a scissor-lift AMR with differential drive (TurtleBot3-Burger wheel geometry), GPS and IMU for pose, and a 640x480 camera on a powered lift that rises to each shelf level. The world uses only Webots base nodes, so it loads offline.

Perception: a YOLOv8n detector (6 classes: 4 SKUs, slot_empty, blocker) fine-tuned on 1419 camera frames captured automatically in Webots from 12 training seeds, with labels projected from simulator geometry and checked visually. Validation: mAP50 0.986, mAP50-95 0.938 (30 epochs, CPU, 320 px).

Reasoning and control: slot regions are found by projecting the known shelf map with the robot pose; slot content comes only from the detector. A reasoning module applies the confidence and visibility gates; the digital twin aggregates observations; a policy module (V1/V3/V4) and a next-best-view selector decide where to look; a go-to-pose controller drives the robot, routing straight to each viewpoint along the open aisle. The audit finishes one shelf side entirely, top level down, before crossing to the other side. The reasoning, twin, policy and viewpoint code is pure Python, unit-tested and independent of the simulator.

## 4. Results (real Webots runs)

Evaluation protocol: seeds 1-12 generated the training frames, seeds 13-14 were used only to debug the pipeline, and seeds 101-130 (30 warehouses, 900 slots per strategy) were used only for this evaluation. Each strategy ran as a separate Webots process per seed, with the same detector and thresholds. Ground truth from the world file is used only for scoring. Mean ± std over seeds:

Webots R2025a, 30 unseen seeds (101-130), 30 slots per seed, mean ± std over seeds.

| Strategy | Runs | Slot accuracy | UNVERIFIED rate | False-EMPTY rate | Discrepancies caught | False alarms / run | Audit time (sim s) | Frames / run |
|---|---|---|---|---|---|---|---|---|
| Fixed patrol (V1) | 30 | 89.0 ± 5.7% | 10.9 ± 5.4% | 0.1 ± 0.6% | 134/152 | 0.03 ± 0.18 | 223 ± 0 | 30.0 ± 0.0 |
| Same-position rescan (V3) | 30 | 89.0 ± 5.7% | 10.8 ± 5.5% | 0.2 ± 0.9% | 134/152 | 0.07 ± 0.25 | 254 ± 21 | 33.3 ± 1.6 |
| Active verification (V4) | 30 | 98.4 ± 2.1% | 1.3 ± 2.0% | 0.1 ± 0.6% | 149/152 | 0.07 ± 0.25 | 284 ± 40 | 36.2 ± 3.5 |

Slot accuracy counts UNVERIFIED as wrong. Active verification reached 98.4% slot accuracy with 1.3% UNVERIFIED, against 89.0% and 10.9% for the fixed patrol and 89.0% and 10.8% for same-position rescan. The cost is audit time: 284 s versus 223 s simulated. False EMPTY reports: 1, 2 and 1 respectively.

## 5. What we learned

Looking again only helps from a different place. Same-position rescan gave the same final status as the fixed patrol on 899 of 900 slots while adding 31 s per audit: in a static scene the same viewpoint returns the same image. Active verification spent a similar amount of extra time (62 s) but chose viewpoints on the far side of the blocker, and resolved most of the slots the patrol could not.

The conservative rule kept wrong decisions rare, but not absent. Across 900 slot decisions per strategy there were 1, 2 and 1 false EMPTY reports out of 835 non-empty slots, and decided accuracy was 99.9%, 99.8% and 99.8% respectively. The great majority of remaining errors are explicit UNVERIFIED slots, which a human can check, rather than confident wrong answers. Active verification identified 149 of 152 real discrepancies, against 134 for the fixed patrol.

Where it still fails: 12 slots stayed UNVERIFIED (12 after all four viewpoints, 12 behind blockers). In 6 of them the detector already had the right label, but the 0.70 visibility gate held the decision back; tuning this gate on validation data is the obvious next step. Two of the 152 discrepancies were also decided wrongly rather than left open. During development we found that blockers were briefly excluded from the training labels by the image-edge filter that drops truncated cartons; because a blocker sits close to the camera and off the bay centre, almost every one is truncated, so the filter cost 12x the blocker boxes. A truncated carton has an ambiguous SKU and a truncated blocker does not, so the two classes are deliberately filtered differently.

## 6. Expected impact

An audit robot is only useful if its reports can be acted on without re-checking. By refusing to guess and by spending extra effort only on the slots it could not resolve, ShelfScan aims to give managers a short, trustworthy exception list (missing and misplaced items) plus an explicit list of slots that still need a human look. Expected benefits are fewer false restock tasks, faster and more frequent counts than manual cycle counting, and audit coverage that is measured rather than assumed. The approach is hardware-agnostic: the decision logic transfers unchanged to any camera-equipped mobile base.

## 7. Extension: machine-readable labels (MVP)

Separately from the result above, the same audit was run in a variant world where each carton carries a machine-readable code instead of a printed SKU pattern: a QR code, or a Code128 barcode. Codes are matched to slots through the map, not ground truth. Over 10 seeds (101-110, 30 slots each, V4): QR 89.0 ± 4.2% slot accuracy, 99.6 ± 1.1% decided accuracy, 3.2 ± 1.3 UNVERIFIED slots per run, 87.6 ± 4.4% decode rate (227/259 cartons), 0 wrong payloads, 359 ± 56 simulated s; Code128 85.7 ± 8.7% slot accuracy, 99.6 ± 1.3% decided accuracy, 4.2 ± 2.5 UNVERIFIED slots per run, 83.9 ± 9.0% decode rate (217/259 cartons), 0 wrong payloads, 461 ± 68 simulated s. No code of either type ever decoded to the wrong SKU. Slot accuracy is below the 98.4% of the printed-pattern world because a carton behind a pallet has an unreadable code, and in label mode the detector's SKU classes are deliberately not trusted, so those slots stay UNVERIFIED rather than becoming a wrong MISPLACED. MVP, 10 seeds, simulation only, and a different experiment from the headline figure. Details: docs/MVP_LABELS.md.

## 8. Path to reality and limits

Deployment path: a ROS 2 differential-drive base with LiDAR localisation on a shelf map (replacing GPS), an RGB camera on a mast, and an edge computer running the INT8-exported YOLOv8n; fine-tune on real shelf images and pilot on one aisle against a manual count. Current limits: simplified scene (rendered cartons with flat printed labels, static world, ideal pose, and a scissor linkage that really folds but rides on small TurtleBot3-class running gear), thresholds not yet tuned on validation data, few truly empty slots in evaluation, and no hardware measurements yet.
