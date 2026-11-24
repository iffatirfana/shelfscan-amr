# Machine-readable carton labels — MVP

**Status: MVP, evaluated on 10 seeds, in simulation only.** Nothing here feeds
the submission: `results/`, `results_v2/`, `models_v2/` and
`docs/ShelfScan_submission.pdf` are untouched. This work lives on the
`ml-extras` branch.

The audit normally reads a carton's SKU from a printed black-and-white pattern
with the YOLOv8n detector. This MVP replaces that pattern with a
machine-readable code — a QR code or a Code128 barcode — and reads it directly
off the camera frame, falling back to the detector when no code can be read.

```
python webots/tools/make_world.py --seed 101 --label qr
python webots/tools/make_world.py --seed 101 --label barcode
python experiments/run_batch.py eval --seeds 101-110 --policies V4_active \
    --seedwise --no-rendering --label qr --results results_qr/runs \
    --weights <abs path>/models_v2/shelfscan_yolov8n.pt
python experiments/summarise_labels.py results_qr/runs results_bc/runs
```

## How a code becomes a decision

Every frame is read for codes. Each code is matched to a slot by projecting
that slot's *nominal* content region — the one derived from the bay geometry in
the robot's map, never from the true carton size — into the image and testing
which region contains the code's centre. A matched code decides the slot
outright: **OK** if the SKU in the payload is the expected one, **MISPLACED** if
not.

A slot with **no readable code stays UNVERIFIED**. In label mode the detector's
SKU classes are deliberately ignored: `models_v2` was trained on printed SKU
patterns, so once the cartons carry codes its SKU head is guessing, and a guess
over the 0.60 threshold turns a blocked slot into a false MISPLACED. Its
`slot_empty` class *is* still used — that is a judgement about shape, not about
print.

Three properties are deliberate:

* **A code can never produce EMPTY.** A decoded code means an item is present.
  The EMPTY rule and both thresholds (confidence ≥ 0.60, visibility ≥ 0.70) are
  untouched, and "nothing decoded" still never means empty.
* **An unreadable code never produces MISPLACED.** The robot reports that it
  could not verify the slot, rather than inventing a discrepancy from a
  detector guess.
* **Codes read for other slots are kept.** A code that decodes is unambiguous,
  so when a frame taken for one slot also resolves a neighbour on the same
  shelf level, that neighbour is recorded rather than discarded.

Plain printed-pattern mode is completely unchanged: the gate is inside
`if LABEL:`, and the plain seed-101 world is byte-identical to the committed
one.

Ground truth is read only in the scoring section, as before.

## Payloads

| Label | Payload | Example |
|---|---|---|
| QR | `SKU=<id>;INV=<invoice>` | `SKU=sku_A;INV=INV-101-13` |
| Code128 | `<sku digit><seed><slot>` | `110113` |

Both carry the same two facts: which SKU, and an invoice number unique to
(seed, slot). They are derived from the seed and the slot id alone and consume
no random numbers, so a seed's layout is byte for byte what it was before
labels existed — verified by comparing `make(101)` with
`make(101, label=...)` field by field.

## Why Code128 cannot reach 8 px per bar here

This was a requirement that the geometry does not allow, so it is worth setting
out plainly.

A QR code is two-dimensional: a 25×25-module symbol plus a 2-module quiet zone
is 29 modules across *and* 29 down. Code128 is one-dimensional — it spends 11
modules per encoded character, plus start, check, stop and two quiet zones, all
along a single axis.

The camera is 640×480 with a 1.0 rad horizontal field of view, so the focal
length is `320 / tan(0.5) = 585.7 px`. At the centre viewpoint (0.90 m from the
carton face) the **widest** carton class covers about 312 px of the frame.

| Payload | Characters | Modules | Width needed for 8 px/bar |
|---|---|---|---|
| `SKU=sku_A;INV=INV-101-13` | 24 | 333 | 2664 px |
| `A;101-13` | 8 | 133 | 1064 px |
| `110113` (Code128-C digits) | 6 | 77 | 616 px |

Even the shortest useful payload needs 616 px of barcode, and the whole frame
is 640 px wide. **8 px per bar is unreachable at this camera resolution for any
payload**, and raising the camera resolution would break comparability with
every result already reported, which were all measured at 640×480.

What is reachable, measured on rendered symbols at the widths each carton class
actually gets:

| Class | Face width | Barcode width on frame | px per module |
|---|---|---|---|
| small | 0.25 m | 156 px | 2.03 |
| tall | 0.30 m | 187 px | 2.43 |
| medium | 0.35 m | 218 px | 2.83 |
| large | 0.45 m | 281 px | 3.65 |
| wide_flat | 0.50 m | 312 px | 4.05 |

So the barcode is built at the largest size the carton allows, with the compact
6-digit payload, and the decode rate is **measured** rather than assumed. The
QR, being two-dimensional, has it much easier: the same information fits in
29 modules across a square that is 108–180 px on the frame, i.e. 3.7–6.2 px per
module.

## Decoders

| Library | Outcome |
|---|---|
| `cv2.QRCodeDetector` | used for QR. `detectAndDecodeMulti` misses codes on the native frame that `detectAndDecode` reads, so the frame is resampled 2× first (seed 101: 8/11 → 9/11 frames), about 55 ms per frame. |
| `pyzbar` | **unusable on this machine** — `libzbar-64.dll` fails to load ("or one of its dependencies"; it needs a Visual C++ runtime that is not installed). |
| `cv2.barcode.BarcodeDetector` | **cannot decode Code128.** OpenCV's barcode module targets EAN/UPC; it returned nothing even for a 1247 px native-resolution Code128 render. |
| `zxing-cpp` | used for Code128. Self-contained wheel, decoded the same renders correctly. |

Using `zxing-cpp` is a deviation from the brief, which asked for pyzbar or
`cv2.barcode`. Neither works for Code128 here; the evidence is in the table
above.

## Results

All V4_active, evaluation seeds only, 30 slots per seed, thresholds unchanged
at 0.60 / 0.70, detector `models_v2` **not** retrained.

### QR, 10 seeds (101–110) — `results_qr/`

| | after the SKU gate | before |
|---|---|---|
| Slot accuracy | 89.0 ± 4.2 % | 89.0 ± 4.7 % |
| UNVERIFIED | 10.7 ± 4.4 % | 9.0 ± 4.2 % |
| **Decided accuracy** | **99.6 ± 1.1 %** | 97.8 ± 1.8 % |
| **False alarms** | **1** | 5 |
| **Code decode rate** | **87.6 ± 4.4 %** (227/259) | 86.8 ± 5.2 % |
| **Wrong payloads** | **0** | 0 |
| Discrepancies caught | 48 / 52 | 47 / 52 |
| False EMPTY | 1 of 10 runs | 1 of 10 runs |
| Audit time | 358.7 ± 55.9 sim s | 352.6 ± 52.9 |

Ignoring the detector's SKU classes in label mode costs nothing in slot
accuracy and cuts false alarms five-fold. The slots it gives up on were ones
the detector was guessing at; a few of those guesses had been right, which is
why UNVERIFIED rises by the same order as the false alarms fall.

Decode rate by carton size class, same 10 seeds:

| small | medium | large | wide_flat | tall |
|---|---|---|---|---|
| 95.9 % | 93.1 % | 88.5 % | 85.7 % | 75.9 % |

`tall` is the weakest despite having one of the largest codes on screen: a
0.45 m carton is the most likely to be clipped by the camera's vertical field
at close range, which is an occlusion problem, not a legibility one.

### Seed 101, both label types

| | QR | Code128 | plain (`results_v2`) |
|---|---|---|---|
| Slot accuracy | 90.0 % | 93.3 % | 96.7 % |
| UNVERIFIED | 3 | 2 | 1 |
| **Decided accuracy** | **100.0 %** | **100.0 %** | 100.0 % |
| **Decode rate** | **87.5 %** (21/24) | **91.7 %** (22/24) | — |
| **Wrong payloads** | **0** | **0** | — |
| Discrepancies caught | 2/2 | 2/2 | 2/2 |
| False EMPTY | 0 | 0 | 0 |
| **False alarms** | **0** | **0** | 0 |
| Audit time | 350 s | 431 s | 320 s |

Both label types now decide nothing they cannot read. Every remaining error in
both runs is the same kind: a slot standing behind a pallet or a carton, left
UNVERIFIED. Code128 reads one more carton than QR here (22 vs 21) because the
extra UNVERIFIED slots send V4 on more re-look detours, which is also why its
audit time is longer.

Before the gate, the Code128 run had decided all 30 slots but called S1-2-2
MISPLACED on a detector output of `sku_D` at 0.614 confidence against a true
`sku_A`. That slot is now UNVERIFIED.

## What this says, honestly

* **Where a code is read, it is right.** 270 codes read across all the runs
  above, zero wrong payloads. That is the appeal: no confidence threshold, no
  confusion between similar printed patterns.
* **What the robot cannot read, it now declines to guess.** Decided accuracy is
  99.6 % over 10 QR seeds and 100 % on both seed-101 runs, with one false alarm
  in 300 slots. The cost is a higher UNVERIFIED rate, which is the intended
  trade: an UNVERIFIED slot asks for a human look, a false MISPLACED sends one
  to the wrong aisle.
* **QR still does not beat the printed pattern on slot accuracy.** 89.0 % over
  10 seeds against 98.4 % for the detector on the printed-pattern world. The
  reason is not the codes: a blocked carton's code cannot be read at all, and
  with the SKU gate in place there is deliberately no fallback. Closing that
  gap means either retraining the detector on code-labelled cartons so the
  fallback is trustworthy again, or accepting more re-look detours.
* **Code128 is at the edge of what this camera can resolve.** It works here
  only because the payload was compressed to six digits and the reader runs a
  cascade; the brief's 8 px per bar is unreachable by a factor of two to nine
  depending on payload. QR has far more headroom at the same information
  content.
* This is an MVP on 10 seeds in simulation. No hardware, no retraining, no
  tuning of the two thresholds.
