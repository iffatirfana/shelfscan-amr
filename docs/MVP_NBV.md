# Learned next-best-view picker (V5) — MVP

**Status: MVP, evaluated on 10 seeds, in simulation only.** Nothing here feeds
the submission: `results/`, `results_v2/`, `models/`, `models_v2/` and the
default world files are untouched. This work lives on the `ml-extras` branch.

V4 picks the next viewpoint for an unresolved slot with a four-line rule: look
from the side opposite the blocker, else look closer, never repeat a viewpoint,
break ties on travel. V5 replaces that choice with a small decision tree
trained to imitate V4, and falls back to the V4 rule whenever the model
declines.

```bash
# 1. log every candidate viewpoint V4 considered, on training seeds only
python experiments/run_batch.py eval --seeds 1-14 --policies V4_active \
    --seedwise --no-rendering --results <scratch> \
    --weights <abs>/models_v2/shelfscan_yolov8n.pt \
    --nbv-log <abs>/results_ml/nbv_train.csv
# 2. train
python experiments/train_nbv.py --max-depth 6
# 3. evaluate V4 and V5 on the evaluation seeds, one seed at a time
python experiments/run_batch.py eval --seeds 101-110 --policies V4_active \
    --seedwise --no-rendering --results <abs>/results_ml/runs --weights <abs>/...
python experiments/run_batch.py eval --seeds 101-110 --policies V5_learned \
    --seedwise --no-rendering --results <abs>/results_ml/runs --weights <abs>/... \
    --nbv-model <abs>/models_ml/nbv_tree.joblib
python experiments/summarise_ml.py results_ml/runs
```

## The table

One row is one **candidate viewpoint** at one decision point; the label is
whether V4 chose that candidate. Features, all available to the robot at the
moment of choosing:

| feature | meaning |
|---|---|
| `uncertainty` | 1 − confidence of the slot's current best observation |
| `visibility` | last estimated visibility of the slot |
| `travel_m` | straight-line distance from the robot's current pose to that viewpoint |
| `blocker_present` | whether an occluder was seen |
| `occ_none/left/right/front` | which side the occluder was on |
| `cand_center/left/right/close` | which viewpoint this candidate is |
| `level`, `level_change` | shelf level, and whether it differs from the last capture |
| `n_tried`, `try_index` | how many viewpoints already tried for this slot |
| `side_is_B` | which racking row |

**The model is deliberately not given the V4 rule as a feature.** There is no
"is this the opposite of the blocker side" input. That single feature *is* the
rule, and handing it over would make the tree a one-split re-encoding of the
thing it is being compared against. It gets the occluder side and the candidate
identity separately and has to find the interaction itself.

Logging is provably side-effect free: the `on_decision` hook fires *after*
`choose_viewpoint` has already returned, and V4 seed 101 run with logging on
produces a run JSON and per-slot CSV **identical to `results_v2`** on every
field. The same check was repeated after V5 was added to the policy table.

## Seed discipline

| seeds | use |
|---|---|
| 1–12 | train the tree |
| 13–14 | dev, held out |
| 101–110 | evaluation only — `train_nbv.py` asserts no row from 101–130 ever reaches the model |

The detector `models_v2` was itself trained on frames from seeds 1–12, so those
layouts are familiar to it. That does not leak into this experiment: the label
being learned is *which viewpoint V4 picked*, which depends on the twin's state,
not on how well the detector generalises.

## Training result

291 candidate rows over 132 decisions from 14 seeds; 249 rows / 113 decisions
for training, 42 rows / 19 decisions held out.

The honest metric is **per-decision top-1 agreement** — did the model's
highest-scoring candidate match V4's pick — not per-row accuracy, which a model
that answered "not chosen" to everything would score well on.

| max_depth | train top-1 | dev top-1 |
|---|---|---|
| 2–3 | 74.3 % | 73.7 % |
| 4 | 78.8 % | 73.7 % |
| 5 | 86.7 % | 78.9 % |
| **6** | **94.7 %** | **94.7 %** |
| 8 | 98.2 % | 89.5 % |
| none | 100 % | 94.7 % |

Depth 6 was chosen: same dev score as an unlimited tree with far less
overfitting. **The dev set is 19 decisions**, so one decision is 5.3 points —
these percentages are coarse and should not be read as precise.

Feature importances at depth 6:

```
cand_close   0.399    occ_right  0.125    travel_m    0.030
try_index    0.279    occ_left   0.122    visibility  0.020
                                          cand_right  0.016
```

The tree rediscovers roughly what V4 does — avoid `close` early, respond to the
occluder side — plus a `try_index` shortcut V4 does not have. At depth 5 and
below it ignored the occluder features entirely and scored worse, which is why
the depth sweep is reported rather than a single number.

## Results

V4 and V5 on seeds 101–110, 30 slots per seed, **paired** (both ran the same
seeds, so the per-seed difference is the informative statistic). Webots ran one
seed at a time throughout.

Pipeline check: the V4 runs in `results_ml/runs` are **identical to the
committed `results_v2` runs on all 10 seeds**, every field. The comparison is
against the real V4, not a re-implementation of it.

| metric | V4_active | V5_learned | paired diff (V5 − V4) |
|---|---|---|---|
| Slot accuracy | 97.7 ± 2.1 % | 98.3 ± 1.7 % | +0.7 pp (95 % CI −0.2 … +1.6) |
| Decided accuracy | 100.0 ± 0.0 % | 100.0 ± 0.0 % | 0.0 (identical every seed) |
| UNVERIFIED slots/run | 0.7 ± 0.6 | 0.5 ± 0.5 | −0.2 (95 % CI −0.5 … +0.1) |
| Discrepancies caught | 50/52 | 50/52 | 0 |
| False alarms | 0 | 0 | 0 |
| False EMPTY | 0 | 0 | 0 |
| Audit time | 299 ± 40 s | 305 ± 43 s | +6 s (95 % CI −1 … +14) |
| Frames per run | 37.9 ± 4.0 | 38.7 ± 4.2 | **+0.8 (95 % CI +0.2 … +1.4)** |
| Distance | 24.9 ± 4.2 m | 25.4 ± 4.5 m | +0.5 m (95 % CI −0.4 … +1.5) |

The picker scored 87 decisions and **never once fell back** to the V4 rule.

### V5 is not better

Say it plainly: **this does not show V5 beating V4.**

* The slot-accuracy difference is +0.7 pp with a 95 % CI of **−0.2 to +1.6**,
  which includes zero. V5 was ahead on 2 of 10 seeds, behind on 0, and
  identical on 8.
* Every safety number is a dead heat: same 50/52 discrepancies caught, zero
  false alarms and zero false EMPTY for both, and decided accuracy 100 % on
  every single seed for both.
* The one difference whose confidence interval **excludes** zero is that V5
  takes **more** frames (+0.8 per run). The only measurable effect is a small
  extra cost.

### Why the two wins are not evidence of better reasoning

Both seeds where V5 came out ahead are the same story, and it is not a flattering one:

| seed | slot | V4 | V5 |
|---|---|---|---|
| 101 | S0-1-1 (behind a pallet) | 4 looks, ends at `close`, conf 0.45 → UNVERIFIED | 4 looks, ends at `close`, conf 0.832 → OK |
| 110 | S0-2-0 (behind a pallet) | 4 looks, ends at `close`, conf 0.598 → UNVERIFIED | 4 looks, ends at `close`, conf 0.834 → OK |

In both cases V5 visited **the same four viewpoints** as V4 and finished at the
**same** one. The only difference was the order: V4 tried right then left, V5
tried left then right. Arriving at `close` from a different direction leaves
the robot at a slightly different settled pose (the driver's tolerance is
0.04 m / 0.04 rad), which produces a slightly different final frame, which
pushed the detector's confidence across the 0.60 threshold.

That is a pose-jitter lottery, not better viewpoint selection. With a threshold
sitting right where these confidences land, reordering two equivalent
viewpoints is enough to flip a slot. It would be dishonest to bank it as a win
for learning.

### What this experiment could and could not show

The label was "which viewpoint did V4 pick", and V4 is a deterministic
four-line rule. **Imitation learning cannot exceed its teacher**: the ceiling
of this setup is to match V4, and 94.7 % dev agreement means V5 mostly does,
diverging on the ~5 % of decisions where it mimics the rule imperfectly. Those
divergences are as likely to hurt as to help; here two happened to help and
none happened to hurt, on 10 seeds.

To actually beat V4, the label would have to be *outcome* rather than
*imitation* — for example, which candidate viewpoint actually resolved the slot
— which needs counterfactual data (trying several viewpoints from the same
state) that a single V4 pass does not produce. That is a bigger experiment than
this MVP.

### Honest limits

* MVP, 10 evaluation seeds, simulation only. No hardware.
* The dev set for model selection was 19 decisions; one decision is 5.3 points.
* The thresholds (0.60 confidence / 0.70 visibility) are untouched design
  defaults, and the two V5 wins sit right on the confidence threshold.
* V5 is trained to imitate V4, so it inherits every weakness of the V4 rule.

