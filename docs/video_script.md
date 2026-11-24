# ShelfScan AMR: 4-minute demo video voice-over

About 636 words, roughly 4 minutes at ~140 words per minute. All numbers are from results_v2/results.csv (30 unseen seeds, real Webots runs).

Covers the four required points: (1) problem, (2) simulation, (3) outcome, (4) path to reality.

## 0:00-0:40  1. The problem

*On screen:* Title slide, then problem slide; warehouse overview render (docs/img/sim_overview.png).

Every warehouse runs on an inventory record, and that record drifts away from what is physically on the shelf. Items get put back in the wrong slot, or run out without anyone noticing. Camera robots can count shelves automatically, but a robot that drives a fixed route has a quiet failure: when a box or a person stands in front of a slot, the detector sees nothing. If that 'nothing' is reported as empty, the store gets a false restock task. If it is silently skipped, the audit looks complete when it is not. ShelfScan AMR is built around one rule: not seeing something is not evidence that it is missing.

## 0:40-1:40  2. The simulation

*On screen:* Screen recording of Webots running an audit (robot driving along a row); then the camera frame with labels (docs/img/cam_example.png); then the architecture slide.

We built the full system in Webots R2025a. Each random seed creates a narrow aisle between two facing racking rows, three levels of five bays on each side, holding plain cardboard cartons in five different sizes, with gaps, misplaced items, and blocking cartons or pallets in front of some slots. Nothing is colour-coded: a carton's product type is readable only from a big black-and-white pattern printed on its shipping label, so the detector has to work from shape and print. The robot is a scissor-lift AMR whose camera rides a powered lift up to each shelf level. Its detector is a YOLOv8n network that we fine-tuned on 1419 frames captured automatically in the simulator from twelve training seeds. On held-out frames it reaches an mAP50 of 0.99. For every slot, the robot compares what it sees with the expected planogram and assigns one of four states: OK, misplaced, empty, or unverified. A slot is only called empty if it is clearly visible and the detector positively sees an empty slot. Anything uncertain becomes unverified, and the active-verification policy then drives to a new viewpoint, on the far side of the blocking object, and looks again.

## 1:40-3:05  3. The outcome

*On screen:* Results slide: table and chart from results_v2/results.csv.

We ran three strategies on 30 warehouse layouts the detector had never seen, all in Webots: a fixed patrol, a re-scan from the same position, and active verification. Slot accuracy, counting unverified slots as wrong, was 89.0 percent for the fixed patrol, 89.0 percent for same-position re-scan, and 98.4 percent for active verification. The share of slots left unverified went from 10.9 percent, to 10.8, to 1.3 percent. False empty reports: 1, 2 and 1 slots in total. Of the 152 real discrepancies, missing or misplaced items, the three strategies identified 134, 134 and 149. Decided accuracy stayed at 99.8 percent or better, so nearly every error is an honest 'unverified' rather than a confident wrong answer. Re-scanning from the same spot bought nothing, because the same viewpoint returns the same image; only moving to a new viewpoint helped. The cost of looking again is time: the average audit took 223 simulated seconds for the fixed patrol, 254 for re-scan and 284 for active verification.

## 3:05-4:00  4. Path to reality

*On screen:* Hardware slide; then limits slide; end on links slide.

The reasoning, digital twin and viewpoint selection are plain Python with unit tests and no simulator dependency, so they move to a real robot unchanged. On hardware we would use a ROS 2 differential-drive base with LiDAR localisation on a shelf map instead of simulator GPS, an RGB camera on a mast at shelf height, and an edge computer running the detector, exported to INT8. After fine-tuning on real shelf photos, the first trial would be one aisle, checked against a manual count. We are honest about the limits: everything here is simulation. The cartons and their labels are rendered, the world is static, localisation is ideal, the robot's scissor linkage really folds but rides on small TurtleBot3-class running gear, the two decision thresholds are design defaults rather than tuned values, and nothing has run on a physical robot yet. What the simulation does show is the core idea: a robot that knows what it has not seen, and goes to look again. Thank you.

## Recording checklist

- Screen-record one Webots audit run (GUI, realtime): `python webots/tools/make_world.py --seed 101 --controller shelfscan_audit`, then open `webots/worlds/warehouse.wbt` in the Webots GUI and press play (V4_active is the default policy; set `SHELFSCAN_POLICY` to switch). Webots closes itself when the audit ends.

- Show `results_v2/results.csv` or the results slide when reading numbers.

- Keep the final video at or under 4:00.
