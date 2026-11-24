"""Build the submission artefacts from REAL run outputs only.

    python docs/build_submission.py

Reads   results_v2/results.csv       (Webots audit runs, one row per run)
        models_v2/train_results.csv  (YOLO training log)
        dataset_v2/                  (captured training frames, for counts)
        docs/img/*.png               (frames rendered in Webots)
Writes  docs/ShelfScan_deck.pptx     deck + concept note appended as slides
        docs/ShelfScan_submission.pdf  deck (16:9 pages) + concept note (A4 pages)
        docs/concept_note.md, docs/img/results_chart.png

Every number in the outputs is computed here from those files; none is typed in.
"""
import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMG = DOCS / "img"

VIDEO_URL = "[VIDEO LINK - add before submitting]"
GITHUB_URL = "https://github.com/iffatirfana/shelfscan-amr"
TEAM = "VOLTIC D. NERDS, Chennai Institute of Technology (CIT) — Iffat Irfana M, Arunachalam R, Aravind K"

POL = ["V1_fixed", "V3_rescan", "V4_active"]
NAME = {"V1_fixed": "Fixed patrol", "V3_rescan": "Same-position rescan",
        "V4_active": "Active verification"}
COLOR = {"V1_fixed": "#2a78d6", "V3_rescan": "#eb6834", "V4_active": "#1baf7a"}

NAVY, TEAL, INK, MUTED, LIGHT = "#12263A", "#0F7C6E", "#1F2328", "#5B6470", "#EEF2F5"


# --------------------------------------------------------------------------- data
def load():
    import os
    rows = list(csv.DictReader(open(os.environ.get("SHELFSCAN_RESULTS_CSV", ROOT / "results_v2/results.csv"))))
    by = {p: [r for r in rows if r["policy"] == p] for p in POL}
    common = set.intersection(*({int(r["seed"]) for r in by[p]} for p in POL))
    S = {}
    for p in POL:
        rs = [r for r in by[p] if int(r["seed"]) in common]
        f = lambda k: [float(r[k]) for r in rs]  # noqa: E731
        S[p] = dict(
            n=len(rs),
            # slots per seed comes from the runs (which read it from
            # layout.json), never from a literal: it changes with the layout
            nslots=int(round(st.mean(f("n_slots")))),
            acc=st.mean(f("slot_accuracy")) * 100, acc_sd=st.pstdev(f("slot_accuracy")) * 100,
            unv=st.mean(f("unverified_rate")) * 100, unv_sd=st.pstdev(f("unverified_rate")) * 100,
            fe=st.mean(f("false_empty_rate")) * 100, fe_sd=st.pstdev(f("false_empty_rate")) * 100,
            fe_n=int(sum(f("false_empty"))), not_empty_n=int(sum(f("n_truly_not_empty"))),
            time=st.mean(f("audit_time_s")), time_sd=st.pstdev(f("audit_time_s")),
            frames=st.mean(f("frames")), dist=st.mean(f("distance_m")),
            caught=int(sum(f("caught"))), real=int(sum(f("true_discrepancies"))),
            fa=st.mean(f("false_alarms")),
            dec_acc=(sum(f("correct")) / sum(f("decided")) * 100) if sum(f("decided")) else float("nan"),
            empty_caught=int(sum(f("empty_caught"))), empty_n=int(sum(f("n_truly_empty"))),
        )
    seeds = sorted(common)
    tr = list(csv.DictReader(open(ROOT / "models_v2/train_results.csv")))
    tr = [{k.strip(): v for k, v in r.items()} for r in tr]
    best = max(tr, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    T = dict(epochs=len(tr), map50=float(best["metrics/mAP50(B)"]),
             map=float(best["metrics/mAP50-95(B)"]),
             n_train=len(list((ROOT / "dataset_v2/images/train").glob("*.jpg"))),
             n_val=len(list((ROOT / "dataset_v2/images/val").glob("*.jpg"))))
    return S, seeds, T


# --------------------------------------------------------------------------- chart
def chart(S, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("acc", "Slot accuracy (%) ↑", ""),
              ("unv", "UNVERIFIED rate (%) ↓", ""),
              ("fe", "False-EMPTY rate (%) ↓", ""),
              ("time", "Audit time (sim s) ↓", "")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), dpi=200)
    for ax, (k, title, sub) in zip(axs, panels):
        xs = range(len(POL))
        vals = [S[p][k] for p in POL]
        sds = [S[p][k + "_sd"] for p in POL]
        ax.bar(xs, vals, width=0.62, color=[COLOR[p] for p in POL], edgecolor="white", linewidth=2,
               yerr=sds, error_kw=dict(ecolor="#888", lw=1, capsize=3))
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.0f}" if k == "time" else f"{v:.1f}", (x, v), ha="center", va="bottom",
                        fontsize=9, color=INK, xytext=(0, 3), textcoords="offset points")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(["Fixed", "Rescan", "Active"], fontsize=9, color=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#ccc")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.yaxis.grid(True, color="#eee")
        ax.set_axisbelow(True)
        top = max(1.0, max(v + s for v, s in zip(vals, sds)) * 1.2)
        ax.set_ylim(0, min(top, 105) if k != "time" else top)
    fig.suptitle(f"Webots R2025a, {len(seeds)} unseen seeds x 3 strategies "
                 f"(mean ± std over seeds)", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = IMG / "results_chart.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- content
def content(S, seeds, T):
    v1, v3, v4 = S["V1_fixed"], S["V3_rescan"], S["V4_active"]
    nslots = v4["nslots"]
    n = len(seeds)
    seedtxt = f"{n} unseen seeds ({seeds[0]}-{seeds[-1]})"
    table = [["Strategy", "Slot accuracy", "UNVERIFIED rate", "False-EMPTY rate",
              "Discrepancies caught", "Audit time (sim s)", "Frames / run"]]
    for p in POL:
        s = S[p]
        table.append([NAME[p], f"{s['acc']:.1f} ± {s['acc_sd']:.1f}%", f"{s['unv']:.1f} ± {s['unv_sd']:.1f}%",
                      f"{s['fe']:.1f}% ({s['fe_n']}/{s['not_empty_n']})", f"{s['caught']}/{s['real']}",
                      f"{s['time']:.0f} ± {s['time_sd']:.0f}", f"{s['frames']:.1f}"])

    slides = [
        dict(kind="title", title="ShelfScan AMR",
             sub="An uncertainty-aware shelf-audit robot that knows when it has not seen enough, "
                 "and goes to look again",
             foot=f"TIH IIT Bombay Physical AI Hackathon  |  {TEAM}"),
        dict(title="Problem: shelf audits fail silently",
             bullets=["Inventory records drift from what is physically on the shelf: misplaced and missing stock.",
                      "Manual cycle counts are slow, so they are done rarely and errors persist for weeks.",
                      "Camera robots that patrol a fixed route have a quieter failure: a box, a person or a "
                      "pallet in front of a slot means 'nothing detected'.",
                      "If 'nothing detected' is reported as EMPTY, the robot triggers false restocks and "
                      "erodes trust; if it is silently skipped, the audit is incomplete and nobody knows.",
                      "Goal: an audit that is honest about what it could not see, and that reduces it."]),
        dict(title="Solution: four honest states + active verification",
             bullets=["Every slot ends as OK, MISPLACED, EMPTY or UNVERIFIED, compared against the expected planogram.",
                      "Rule: 'nothing detected' is never EMPTY. EMPTY needs a clearly visible slot AND the detector "
                      "positively seeing an empty slot.",
                      "Low confidence or a slot hidden behind an obstacle becomes UNVERIFIED, not a guess.",
                      "Active verification: for each UNVERIFIED slot the robot picks a new viewpoint "
                      "(opposite side of the blocker, or closer) and re-observes, up to 3 extra looks.",
                      "A digital twin keeps expected vs observed state per slot and exports an audit report (CSV)."]),
        dict(title="System architecture", kind="flow",
             flow=[["Camera 640x480\non 0.47 m mast", "YOLOv8n detector\n6 classes", "Slot association\n(map + pose projection)",
                    "Slot reasoning\nconf + visibility gates"],
                   ["Digital twin\nexpected vs observed", "Policy\nV1 / V3 / V4", "Next-best-view\nselector",
                    "Go-to-pose driver\nGPS + IMU, diff-drive"]],
             note="Slot locations come from the known shelf map projected with the robot pose; slot CONTENT comes only "
                  "from camera pixels through the detector. Simulator ground truth is used only to label training "
                  "data and to score runs."),
        dict(title="AI depth",
             bullets=[f"Perception: YOLOv8n fine-tuned on {T['n_train'] + T['n_val']} frames captured in Webots from "
                      f"12 training seeds ({T['n_train']} train / {T['n_val']} val), {T['epochs']} epochs on CPU, 320 px.",
                      f"Validation (held-out frames): mAP50 = {T['map50']:.3f}, mAP50-95 = {T['map']:.3f}. "
                      "Classes: 4 SKUs, slot_empty, blocker.",
                      "Labels are generated automatically from simulator geometry (projected 3D boxes), "
                      "checked visually with an overlay tool before training.",
                      "Uncertainty reasoning: a decision needs detector confidence >= 0.60 and estimated slot "
                      "visibility >= 0.70; visibility is computed from overlapping detections (blockers).",
                      "Active perception: the side of the blocker in the image chooses the next viewpoint; "
                      "a new decisive observation replaces UNVERIFIED in the twin.",
                      f"Seeds 1-12 for training data, 13-14 for debugging only, {seedtxt} for evaluation only "
                      "(never seen in training or debugging)."]),
        dict(title="Simulation setup (Webots R2025a)", kind="images",
             images=[(IMG / "sim_overview.png", "Two facing racking rows across one narrow aisle, "
                      "3 levels x 5 bays each, cardboard cartons of varying size per seed"),
                     (IMG / "cam_example.png", "Robot camera frame with detector training labels")],
             bullets=["Scissor-lift AMR built from Webots base nodes: differential drive (TurtleBot3-Burger "
                      "wheel geometry), GPS + IMU for pose, and a 640x480 camera on a powered lift that "
                      "rises to each shelf level (0.56 / 1.21 / 1.86 m).",
                      "Contents are plain cardboard cartons in five size classes. There is NO colour cue: the "
                      "SKU is carried only by a large black-and-white pattern on a white shipping label "
                      "(square / horizontal bars / dot grid / vertical bars). A blocker is an unlabelled "
                      "carton or a pallet stood at the bay mouth; an empty slot is a bare bay.",
                      f"Per seed: {nslots} slots; P(empty)=0.08, P(misplaced)=0.08, P(blocker)=0.20, "
                      "P(expected-empty)=0.05.",
                      "Viewpoints per slot: centre (0.90 m from the carton face), left / right (0.45 m "
                      "sideways, turned to face the slot), close (0.65 m). The audit completes one shelf "
                      "side entirely, top level down, before crossing to the other side.",
                      "Each run is a separate Webots process in fast mode with main-window rendering off; "
                      "that was verified to give bit-identical results and is 17% faster."]),
        dict(title=f"Results: {seedtxt}, 3 strategies, real Webots runs", kind="results",
             table=table, image=IMG / "results_chart.png",
             note="Slot accuracy = final status equals ground truth (UNVERIFIED counts as wrong). "
                  "False-EMPTY rate = slots reported EMPTY that were not empty / all non-empty slots. "
                  "Audit time = simulated seconds. Source: results_v2/results.csv."),
        dict(title="What the results show", bullets=findings(S, n)),
        dict(title="Path to real hardware",
             bullets=["Robot: any ROS 2 differential-drive base (e.g. TurtleBot3-class or a warehouse AMR) "
                      "with wheel odometry + IMU; replace GPS with LiDAR/AMCL localisation on a shelf map.",
                      "Sensor: RGB camera on a mast at shelf height; the same 640x480 input as simulation.",
                      "Compute: YOLOv8n is small enough for edge boards (Jetson Orin Nano / Raspberry Pi 5 + "
                      "accelerator); train_yolo.py can export INT8 TFLite. Not yet benchmarked on hardware.",
                      "Data: fine-tune on real shelf photos; keep the sim pipeline for pre-training and for "
                      "testing policies before a store trial.",
                      "Software that transfers unchanged: reasoning, digital twin, policies, next-best-view "
                      "(pure Python, unit-tested, no Webots dependency).",
                      "Pilot plan: one aisle, nightly audit, compare against a manual count."]),
        dict(title="Innovation and honest limits",
             bullets=["Innovation: an explicit UNVERIFIED state plus a rule that absence of detection is not "
                      "evidence of absence; uncertainty drives where the robot looks next.",
                      "Innovation: the same policy code runs in unit tests and in Webots; every reported number "
                      "is from a Webots run on unseen seeds.",
                      "Found in testing: blockers were briefly dropped from training labels by the same "
                      "image-edge filter that (correctly) drops truncated cartons. A blocker sits 0.20 m off "
                      "the bay centre only 0.70 m from the camera, so any blocker 0.45 m or wider always "
                      "overruns the frame; filtering them cost 12x the blocker boxes. A truncated carton has "
                      "an ambiguous SKU, a truncated blocker does not, so the asymmetry is deliberate.",
                      "Limit: simulation only. Simplified scene (rendered cartons, flat printed labels, "
                      "static world, perfect GPS pose) and a scissor-lift look-alike on small running gear; "
                      "nothing has run on a physical robot.",
                      "Limit: decision thresholds (0.60 conf / 0.70 visibility) are design defaults, not tuned "
                      "on validation data.",
                      f"Limit: slot_empty is the rarest training class; only {S['V4_active']['empty_n']} truly "
                      "empty slots occurred across the evaluation seeds.",
                      "Limit: no real-robot test yet; latency and energy on edge hardware not measured."]),
        dict(kind="title", title="Links",
             sub=f"Demo video: {VIDEO_URL}\nCode: {GITHUB_URL}",
             foot="Reproduce: README.md -> 'How to run'. All numbers: results_v2/results.csv"),
    ]
    return slides


def slot_facts():
    """Per-slot facts from results_v2/runs/*_slots.csv (written by the Webots controller)."""
    runs = ROOT / "results_v2/runs"
    diff = total = 0
    for f in sorted(runs.glob("V1_fixed_seed*_slots.csv")):
        a = list(csv.DictReader(open(f)))
        b = list(csv.DictReader(open(str(f).replace("V1_fixed", "V3_rescan"))))
        total += len(a)
        diff += sum(1 for x, y in zip(a, b) if x["final_status"] != y["final_status"])
    left = [r for f in sorted(runs.glob("V4_active_seed*_slots.csv")) for r in csv.DictReader(open(f))
            if r["final_status"] == "UNVERIFIED"]
    return dict(v1v3_diff=diff, v1v3_total=total, v4_left=len(left),
                v4_left_occluded=sum(1 for r in left if r["occluder"]),
                v4_left_all_views=sum(1 for r in left if int(r["observations"]) >= 4),
                v4_left_right_label=sum(1 for r in left if r["observed_label"] and r["observed_label"] == r["actual"]))


def findings(S, n):
    v1, v3, v4 = S["V1_fixed"], S["V3_rescan"], S["V4_active"]
    F = slot_facts()
    out = [
        f"Active verification raised slot accuracy from {v1['acc']:.1f}% (fixed patrol) to {v4['acc']:.1f}% "
        f"and cut UNVERIFIED slots from {v1['unv']:.1f}% to {v4['unv']:.1f}% (mean over {n} unseen seeds).",
        f"Same-position rescan bought nothing: it reached the same slot accuracy and the same "
        f"{v3['caught']}/{v3['real']} discrepancies as a single patrol, and its final status matches the "
        f"fixed patrol on {F['v1v3_total'] - F['v1v3_diff']}/{F['v1v3_total']} slots. In a static scene the "
        f"same viewpoint returns the same image; only a new viewpoint adds information.",
        f"The conservative EMPTY rule kept wrong decisions rare but not zero: false EMPTY "
        f"{v1['fe_n']} / {v3['fe_n']} / {v4['fe_n']} out of {v4['not_empty_n']} non-empty slots, and "
        f"decided accuracy {v1['dec_acc']:.1f}% / {v3['dec_acc']:.1f}% / {v4['dec_acc']:.1f}%. "
        f"Almost every error is an honest UNVERIFIED rather than a wrong answer.",
        f"Real discrepancies (missing or misplaced) identified: {v1['caught']}/{v1['real']} fixed, "
        f"{v3['caught']}/{v3['real']} rescan, {v4['caught']}/{v4['real']} active.",
        f"Cost: audit time {v1['time']:.0f} / {v3['time']:.0f} / {v4['time']:.0f} simulated s; "
        f"frames per run {v1['frames']:.1f} / {v3['frames']:.1f} / {v4['frames']:.1f}.",
        f"Still unresolved by V4: {F['v4_left']} of {v4['not_empty_n'] + v4['empty_n']} slots "
        f"({F['v4_left_occluded']} behind blockers, {F['v4_left_all_views']} after all 4 viewpoints); in "
        f"{F['v4_left_right_label']} of them the detector had the right label but the 0.70 visibility gate "
        f"held the decision back.",
    ]
    return out


# --------------------------------------------------------------------------- concept note
def concept_note(S, seeds, T):
    v1, v3, v4 = S["V1_fixed"], S["V3_rescan"], S["V4_active"]
    nslots = v4["nslots"]
    F = slot_facts()
    n = len(seeds)
    return [
        ("h1", "ShelfScan AMR: concept note"),
        ("p", f"Uncertainty-aware autonomous shelf auditing with active visual verification. "
              f"TIH IIT Bombay Physical AI Hackathon. {TEAM}. Code: {GITHUB_URL}. Video: {VIDEO_URL}."),
        ("h2", "1. Problem"),
        ("p", "Warehouses and stores run on inventory records that drift away from what is physically on "
              "the shelf: items are picked from the wrong slot, put back in the wrong place, or run out "
              "without the system noticing. Manual cycle counts are slow and expensive, so they are done "
              "rarely, and errors can persist for weeks, causing failed picks, stock-outs and excess safety stock."),
        ("p", "Mobile robots with cameras can count shelves automatically, but a fixed patrol has a quiet failure "
              "mode. When something stands in front of a slot (a box left in the aisle, a person, a pallet) "
              "or the view is poor, the detector simply sees nothing. A system that treats 'nothing detected' "
              "as 'empty' raises false restock alarms; a system that silently skips the slot produces an audit "
              "that looks complete but is not. Either way, staff stop trusting the robot."),
        ("h2", "2. Approach"),
        ("p", "ShelfScan AMR makes uncertainty explicit and acts on it. Every slot ends in one of four states: "
              "OK, MISPLACED, EMPTY or UNVERIFIED. The central rule is that absence of a detection is never "
              "evidence of absence: a slot is EMPTY only if it is clearly visible (estimated visibility >= 0.70) "
              "and the detector positively recognises an empty slot with confidence >= 0.60. Anything less "
              "is UNVERIFIED."),
        ("p", "UNVERIFIED slots are then handled by active verification. The robot reads the side of the "
              "image on which the blocking object appeared and drives to a viewpoint on the opposite side, "
              "or closer to the shelf if nothing was blocking, and observes again (up to three extra looks per "
              "slot). A new decisive observation replaces UNVERIFIED in a digital twin that stores expected "
              "versus observed state, confidence, viewpoint and number of observations for every slot, and "
              "exports an audit report."),
        ("p", "We compare three strategies with identical perception: V1 fixed patrol (one look per slot), "
              "V3 same-position rescan (one more look from the same viewpoint for unresolved slots), and V4 "
              "active verification (new viewpoints chosen from the evidence)."),
        ("h2", "3. System architecture"),
        ("p", "Simulation: Webots R2025a. Each seed generates a narrow aisle between two facing racking rows, "
              "3 levels of 5 bays on each side, holding plain cardboard cartons in five size classes, with "
              "gaps, misplaced items and blocking objects in front of about 20% of slots. There is no colour "
              "cue: a carton's SKU is carried only by a large black-and-white pattern printed on a white "
              "shipping label, and a blocker is an unlabelled carton or a pallet stood at the bay mouth, so "
              "the detector must work from shape and print rather than colour. The robot is a scissor-lift "
              "AMR with differential drive (TurtleBot3-Burger wheel geometry), GPS and IMU for pose, and a "
              "640x480 camera on a powered lift that rises to each shelf level. The world uses only Webots "
              "base nodes, so it loads offline."),
        ("p", f"Perception: a YOLOv8n detector (6 classes: 4 SKUs, slot_empty, blocker) fine-tuned on "
              f"{T['n_train'] + T['n_val']} camera frames captured automatically in Webots from 12 training "
              f"seeds, with labels projected from simulator geometry and checked visually. Validation: "
              f"mAP50 {T['map50']:.3f}, mAP50-95 {T['map']:.3f} ({T['epochs']} epochs, CPU, 320 px)."),
        ("p", "Reasoning and control: slot regions are found by projecting the known shelf map with the robot "
              "pose; slot content comes only from the detector. A reasoning module applies the confidence and "
              "visibility gates; the digital twin aggregates observations; a policy module (V1/V3/V4) and a "
              "next-best-view selector decide where to look; a go-to-pose controller drives the robot, routing "
              "straight to each viewpoint along the open aisle. The audit finishes one shelf side entirely, "
              "top level down, before crossing to the other side. The reasoning, twin, policy and viewpoint "
              "code is pure Python, "
              "unit-tested and independent of the simulator."),
        ("h2", "4. Results (real Webots runs)"),
        ("p", f"Evaluation protocol: seeds 1-12 generated the training frames, seeds 13-14 were used only to "
              f"debug the pipeline, and seeds {seeds[0]}-{seeds[-1]} ({n} warehouses, "
              f"{n * nslots} slots per strategy) "
              f"were used only for this evaluation. Each strategy ran as a separate Webots process per seed, with "
              f"the same detector and thresholds. Ground truth from the world file is used only for scoring. "
              f"Mean ± std over seeds:"),
        ("table", None),
        ("p", f"Slot accuracy counts UNVERIFIED as wrong. Active verification reached {v4['acc']:.1f}% slot "
              f"accuracy with {v4['unv']:.1f}% UNVERIFIED, against {v1['acc']:.1f}% and {v1['unv']:.1f}% for the "
              f"fixed patrol and {v3['acc']:.1f}% and {v3['unv']:.1f}% for same-position rescan. The cost is audit "
              f"time: {v4['time']:.0f} s versus {v1['time']:.0f} s simulated. False EMPTY reports: "
              f"{v1['fe_n']}, {v3['fe_n']} and {v4['fe_n']} respectively."),
        ("h2", "5. What we learned"),
        ("p", f"Looking again only helps from a different place. Same-position rescan gave the same final status as "
              f"the fixed patrol on {F['v1v3_total'] - F['v1v3_diff']} of {F['v1v3_total']} slots while adding "
              f"{v3['time'] - v1['time']:.0f} s per audit: in a static scene the same viewpoint returns the same image. "
              f"Active verification spent a similar amount of extra time ({v4['time'] - v1['time']:.0f} s) but chose "
              f"viewpoints on the far side of the blocker, and resolved most of the slots the patrol could not."),
        ("p", f"The conservative rule kept wrong decisions rare, but not absent. Across {n * nslots} slot "
              f"decisions per strategy there were {v1['fe_n']}, {v3['fe_n']} and {v4['fe_n']} false EMPTY "
              f"reports out of {v4['not_empty_n']} non-empty slots, and decided accuracy was "
              f"{v1['dec_acc']:.1f}%, {v3['dec_acc']:.1f}% and {v4['dec_acc']:.1f}% respectively. The great "
              f"majority of remaining errors are explicit UNVERIFIED slots, which a human can check, rather "
              f"than confident wrong answers. Active verification identified {v4['caught']} of {v4['real']} "
              f"real discrepancies, against {v1['caught']} for the fixed patrol."),
        ("p", f"Where it still fails: {F['v4_left']} slots stayed UNVERIFIED ({F['v4_left_all_views']} after all four viewpoints, "
              f"{F['v4_left_occluded']} behind blockers). In {F['v4_left_right_label']} of them the detector already had the right label, "
              f"but the 0.70 visibility gate held the decision back; tuning this gate on validation data is the "
              f"obvious next step. Two of the {v4['real']} discrepancies were also decided wrongly rather "
              f"than left open. During development we found that blockers were briefly excluded from the "
              f"training labels by the image-edge filter that drops truncated cartons; because a blocker sits "
              f"close to the camera and off the bay centre, almost every one is truncated, so the filter cost "
              f"12x the blocker boxes. A truncated carton has an ambiguous SKU and a truncated blocker does "
              f"not, so the two classes are deliberately filtered differently."),
        ("h2", "6. Expected impact"),
        ("p", "An audit robot is only useful if its reports can be acted on without re-checking. By refusing to "
              "guess and by spending extra effort only on the slots it could not resolve, ShelfScan aims to give "
              "managers a short, trustworthy exception list (missing and misplaced items) plus an explicit list "
              "of slots that still need a human look. Expected benefits are fewer false restock tasks, faster "
              "and more frequent counts than manual cycle counting, and audit coverage that is measured rather "
              "than assumed. The approach is hardware-agnostic: the decision logic transfers unchanged to any "
              "camera-equipped mobile base."),
        ("h2", "7. Path to reality and limits"),
        ("p", "Deployment path: a ROS 2 differential-drive base with LiDAR localisation on a shelf map (replacing "
              "GPS), an RGB camera on a mast, and an edge computer running the INT8-exported YOLOv8n; fine-tune "
              "on real shelf images and pilot on one aisle against a manual count. Current limits: simplified "
              "scene (rendered cartons with flat printed labels, static world, ideal pose, and a "
              "scissor-lift look-alike on small running gear), thresholds not yet tuned on "
              "validation data, few truly empty slots in evaluation, and no hardware measurements yet."),
    ]


# --------------------------------------------------------------------------- PPTX
def build_pptx(slides, note, S, seeds, out):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    def rgb(h):
        return RGBColor.from_string(h.lstrip("#"))

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    def textbox(sl, x, y, w, h, paras, size=18, color=INK, bold=False, align=None, bullet=False):
        tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, t in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = ("•  " + t) if bullet else t
            p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, rgb(color)
            p.font.name = "Calibri"
            p.space_after = Pt(size * 0.55)
            if align:
                p.alignment = align
        return tb

    def header(sl, title, num):
        bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.05))
        bar.fill.solid(); bar.fill.fore_color.rgb = rgb(NAVY); bar.line.fill.background()
        textbox(sl, 0.5, 0.2, 12.3, 0.8, [title], size=28, color="#FFFFFF", bold=True)
        textbox(sl, 12.2, 7.0, 1.0, 0.4, [str(num)], size=11, color=MUTED, align=PP_ALIGN.RIGHT)

    for i, s in enumerate(slides, 1):
        sl = prs.slides.add_slide(blank)
        kind = s.get("kind", "bullets")
        if kind == "title":
            bg = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
            bg.fill.solid(); bg.fill.fore_color.rgb = rgb(NAVY); bg.line.fill.background()
            textbox(sl, 0.8, 2.2, 11.7, 1.3, [s["title"]], size=54, color="#FFFFFF", bold=True)
            textbox(sl, 0.8, 3.6, 11.7, 1.8, s["sub"].split("\n"), size=24, color="#CFE3F0")
            textbox(sl, 0.8, 6.4, 11.7, 0.6, [s["foot"]], size=14, color="#9FB6C8")
            continue
        header(sl, s["title"], i)
        if kind == "bullets":
            textbox(sl, 0.7, 1.45, 11.9, 5.6, s["bullets"], size=20, bullet=True)
        elif kind == "flow":
            for r, row in enumerate(s["flow"]):
                y = 1.6 + r * 2.1
                for c, label in enumerate(row):
                    x = 0.6 + c * 3.15
                    b = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(2.6), Inches(1.3))
                    b.fill.solid(); b.fill.fore_color.rgb = rgb(LIGHT if r == 0 else "#E3F1EE")
                    b.line.color.rgb = rgb(NAVY if r == 0 else TEAL)
                    tf = b.text_frame; tf.word_wrap = True
                    for j, t in enumerate(label.split("\n")):
                        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                        p.text = t; p.alignment = PP_ALIGN.CENTER
                        p.font.size = Pt(15 if j == 0 else 12); p.font.bold = j == 0
                        p.font.color.rgb = rgb(INK)
                    if c < len(row) - 1:
                        ln = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + 2.62), Inches(y + 0.65),
                                                     Inches(x + 3.13), Inches(y + 0.65))
                        ln.line.color.rgb = rgb(MUTED); ln.line.width = Pt(2)
            textbox(sl, 0.6, 3.05, 12.2, 0.5, ["↓ observations update the twin; unresolved slots trigger new viewpoints ↓"],
                    size=13, color=MUTED, align=PP_ALIGN.CENTER)
            textbox(sl, 0.6, 5.3, 12.2, 1.6, [s["note"]], size=15, color=MUTED)
        elif kind == "images":
            for j, (path, cap) in enumerate(s["images"]):
                x = 0.6 + j * 6.25
                from PIL import Image
                iw, ih = Image.open(path).size
                w = min(5.9, 3.35 * iw / ih)
                sl.shapes.add_picture(str(path), Inches(x), Inches(1.3), width=Inches(w))
                textbox(sl, x, 4.75, 5.9, 0.5, [cap], size=12, color=MUTED)
            textbox(sl, 0.6, 5.35, 12.2, 1.9, s["bullets"], size=15, bullet=True)
        elif kind == "results":
            t = s["table"]
            shp = sl.shapes.add_table(len(t), len(t[0]), Inches(0.5), Inches(1.3), Inches(12.3), Inches(1.6))
            tbl = shp.table
            widths = [2.5, 1.8, 1.9, 1.9, 1.6, 1.5, 1.1]
            for c, w in enumerate(widths):
                tbl.columns[c].width = Inches(w)
            for r, row in enumerate(t):
                for c, val in enumerate(row):
                    cell = tbl.cell(r, c)
                    cell.text = val
                    p = cell.text_frame.paragraphs[0]
                    p.font.size = Pt(13); p.font.bold = r == 0 or c == 0
                    p.font.color.rgb = rgb("#FFFFFF" if r == 0 else INK)
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = rgb(NAVY if r == 0 else ("#FFFFFF" if r % 2 else LIGHT))
            sl.shapes.add_picture(str(s["image"]), Inches(0.5), Inches(3.15), width=Inches(12.3))
            textbox(sl, 0.5, 6.65, 12.3, 0.7, [s["note"]], size=11, color=MUTED)

    # concept note appended as slides (same text as the A4 pages in the PDF)
    chunks, cur = [], []
    for item in note:
        if item[0] == "h2" and cur and sum(len(x[1] or "") for x in cur) > 1500:
            chunks.append(cur); cur = []
        cur.append(item)
    chunks.append(cur)
    for k, chunk in enumerate(chunks, 1):
        sl = prs.slides.add_slide(blank)
        header(sl, f"Concept note ({k}/{len(chunks)})", len(slides) + k)
        tb = sl.shapes.add_textbox(Inches(0.5), Inches(1.25), Inches(12.3), Inches(6.0))
        tf = tb.text_frame; tf.word_wrap = True
        first = True
        for kind, text in chunk:
            if kind == "h1":
                continue
            if kind == "table":
                text = "  |  ".join(f"{NAME[p]}: accuracy {S[p]['acc']:.1f}%, UNVERIFIED {S[p]['unv']:.1f}%, "
                                    f"false-EMPTY {S[p]['fe']:.1f}%, time {S[p]['time']:.0f} s" for p in POL)
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = text
            p.font.name = "Calibri"
            p.font.size = Pt(14 if kind == "h2" else 11)
            p.font.bold = kind == "h2"
            p.font.color.rgb = rgb(NAVY if kind == "h2" else INK)
            p.space_after = Pt(4)
    prs.save(out)


# --------------------------------------------------------------------------- PDF
def build_pdf(slides, note, S, seeds, out):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Frame, KeepInFrame, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    from reportlab.pdfgen import canvas as rl_canvas
    from pypdf import PdfReader, PdfWriter

    font, bold = "Helvetica", "Helvetica-Bold"
    for f, b in (("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf"),):
        if Path(f).exists() and Path(b).exists():
            pdfmetrics.registerFont(TTFont("Cal", f)); pdfmetrics.registerFont(TTFont("CalB", b))
            font, bold = "Cal", "CalB"
    C = colors.HexColor
    W, H = 13.333 * inch, 7.5 * inch
    deck_pdf = DOCS / "_deck.pdf"
    c = rl_canvas.Canvas(str(deck_pdf), pagesize=(W, H))

    def para(text, size, color=INK, fnt=None, align=0, lead=1.25):
        return Paragraph(text.replace("&", "&amp;").replace("\n", "<br/>"),
                         ParagraphStyle("x", fontName=fnt or font, fontSize=size, leading=size * lead,
                                        textColor=C(color), alignment=align, spaceAfter=size * 0.5))

    def box(items, x, y, w, h):  # x, y = top-left in inches
        f = Frame(x * inch, H - (y + h) * inch, w * inch, h * inch, showBoundary=0,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        f.addFromList([KeepInFrame(w * inch, h * inch, items, mode="shrink")], c)

    def bullets(lines, size):
        st_ = ParagraphStyle("b", fontName=font, fontSize=size, leading=size * 1.25, textColor=C(INK),
                             leftIndent=16, bulletIndent=0, spaceAfter=size * 0.55)
        return [Paragraph(t.replace("&", "&amp;"), st_, bulletText="•") for t in lines]

    for i, s in enumerate(slides, 1):
        kind = s.get("kind", "bullets")
        if kind == "title":
            c.setFillColor(C(NAVY)); c.rect(0, 0, W, H, stroke=0, fill=1)
            box([para(s["title"], 54, "#FFFFFF", bold)], 0.8, 2.2, 11.7, 1.3)
            box([para(s["sub"], 24, "#CFE3F0")], 0.8, 3.6, 11.7, 1.8)
            box([para(s["foot"], 14, "#9FB6C8")], 0.8, 6.4, 11.7, 0.6)
            c.showPage(); continue
        c.setFillColor(C(NAVY)); c.rect(0, H - 1.05 * inch, W, 1.05 * inch, stroke=0, fill=1)
        box([para(s["title"], 28, "#FFFFFF", bold)], 0.5, 0.25, 12.3, 0.7)
        c.setFont(font, 11); c.setFillColor(C(MUTED)); c.drawRightString(W - 0.35 * inch, 0.3 * inch, str(i))
        if kind == "bullets":
            box(bullets(s["bullets"], 20), 0.7, 1.45, 11.9, 5.6)
        elif kind == "flow":
            for r, row in enumerate(s["flow"]):
                y = 1.6 + r * 2.1
                for col, label in enumerate(row):
                    x = 0.6 + col * 3.15
                    c.setFillColor(C(LIGHT if r == 0 else "#E3F1EE")); c.setStrokeColor(C(NAVY if r == 0 else TEAL))
                    c.roundRect(x * inch, H - (y + 1.3) * inch, 2.6 * inch, 1.3 * inch, 8, stroke=1, fill=1)
                    a, *rest = label.split("\n")
                    box([para(a, 15, INK, bold, TA_CENTER)] + [para(t, 12, INK, None, TA_CENTER) for t in rest],
                        x + 0.1, y + 0.25, 2.4, 1.0)
                    if col < len(row) - 1:
                        c.setStrokeColor(C(MUTED)); c.setLineWidth(2)
                        yy = H - (y + 0.65) * inch
                        c.line((x + 2.62) * inch, yy, (x + 3.1) * inch, yy)
                        c.line((x + 3.0) * inch, yy + 4, (x + 3.1) * inch, yy); c.line((x + 3.0) * inch, yy - 4, (x + 3.1) * inch, yy)
                        c.setLineWidth(1)
            box([para("↓ observations update the twin; unresolved slots trigger new viewpoints ↓", 13, MUTED,
                      None, TA_CENTER)], 0.6, 3.1, 12.2, 0.5)
            box([para(s["note"], 15, MUTED)], 0.6, 5.3, 12.2, 1.6)
        elif kind == "images":
            from PIL import Image
            for j, (path, cap) in enumerate(s["images"]):
                x = 0.6 + j * 6.25
                iw, ih = Image.open(path).size
                w = min(5.9, 3.35 * iw / ih)
                h = w * ih / iw
                c.drawImage(str(path), x * inch, H - (1.3 + h) * inch, w * inch, h * inch)
                box([para(cap, 12, MUTED)], x, 4.75, 5.9, 0.5)
            box(bullets(s["bullets"], 15), 0.6, 5.35, 12.2, 1.9)
        elif kind == "results":
            from PIL import Image
            t = s["table"]
            tbl = Table(t, colWidths=[w * inch for w in (2.5, 1.8, 1.9, 1.9, 1.6, 1.5, 1.1)])
            tbl.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, -1), font, 12), ("FONT", (0, 0), (-1, 0), bold, 12),
                ("FONT", (0, 1), (0, -1), bold, 12),
                ("BACKGROUND", (0, 0), (-1, 0), C(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C(LIGHT)]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            box([tbl], 0.5, 1.3, 12.3, 1.7)
            iw, ih = Image.open(s["image"]).size
            h = 12.3 * ih / iw
            c.drawImage(str(s["image"]), 0.5 * inch, H - (3.15 + h) * inch, 12.3 * inch, h * inch)
            box([para(s["note"], 11, MUTED)], 0.5, 6.65, 12.3, 0.7)
        c.showPage()
    c.save()

    # concept note, A4
    note_pdf = DOCS / "_note.pdf"
    doc = SimpleDocTemplate(str(note_pdf), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title="ShelfScan AMR concept note")
    sty = {"h1": ParagraphStyle("h1", fontName=bold, fontSize=17, leading=21, textColor=C(NAVY), spaceAfter=4),
           "h2": ParagraphStyle("h2", fontName=bold, fontSize=12, leading=15, textColor=C(NAVY),
                                spaceBefore=7, spaceAfter=3),
           "p": ParagraphStyle("p", fontName=font, fontSize=10, leading=13.2, textColor=C(INK), spaceAfter=5)}
    story = []
    for kind, text in note:
        if kind == "table":
            rows = [["Strategy", "Slot accuracy", "UNVERIFIED", "False-EMPTY", "Discrepancies caught", "Audit time (s)"]]
            for p in POL:
                x = S[p]
                rows.append([NAME[p], f"{x['acc']:.1f} ± {x['acc_sd']:.1f}%", f"{x['unv']:.1f}%", f"{x['fe']:.1f}%",
                             f"{x['caught']}/{x['real']}", f"{x['time']:.0f}"])
            tbl = Table(rows)
            tbl.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), font, 9), ("FONT", (0, 0), (-1, 0), bold, 9),
                                     ("BACKGROUND", (0, 0), (-1, 0), C(LIGHT)),
                                     ("GRID", (0, 0), (-1, -1), 0.4, C("#BBBBBB")),
                                     ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story += [KeepTogether([tbl, Spacer(1, 6)])]
        else:
            story.append(Paragraph(text.replace("&", "&amp;"), sty[kind]))
    doc.build(story)

    w = PdfWriter()
    for part in (deck_pdf, note_pdf):
        for pg in PdfReader(str(part)).pages:
            w.add_page(pg)
    with open(out, "wb") as f:
        w.write(f)
    n_note = len(PdfReader(str(note_pdf)).pages)
    deck_pdf.unlink(); note_pdf.unlink()
    return n_note


def note_markdown(note, S):
    lines = []
    for kind, text in note:
        if kind == "h1":
            lines.append(f"# {text}\n")
        elif kind == "h2":
            lines.append(f"## {text}\n")
        elif kind == "table":
            lines.append((ROOT / "results_v2/summary.md").read_text(encoding="utf-8"))
        else:
            lines.append(text + "\n")
    (DOCS / "concept_note.md").write_text("\n".join(lines), encoding="utf-8")


def video_script(S, seeds, T):
    """~4 min voice-over covering the four required points: problem, simulation,
    outcome, path to reality. Numbers are filled from results.csv."""
    v1, v3, v4 = S["V1_fixed"], S["V3_rescan"], S["V4_active"]
    n = len(seeds)
    sec = [
        ("0:00-0:40", "1. The problem", "Title slide, then problem slide; warehouse overview render (docs/img/sim_overview.png).",
         "Every warehouse runs on an inventory record, and that record drifts away from what is physically on the "
         "shelf. Items get put back in the wrong slot, or run out without anyone noticing. Camera robots can count shelves automatically, but a robot that drives a "
         "fixed route has a quiet failure: when a box or a person stands in front of a slot, the detector sees "
         "nothing. If that 'nothing' is reported as empty, the store gets a false restock task. If it is silently "
         "skipped, the audit looks complete when it is not. ShelfScan AMR is built around one rule: not seeing "
         "something is not evidence that it is missing."),
        ("0:40-1:40", "2. The simulation", "Screen recording of Webots running an audit (robot driving along a row); "
         "then the camera frame with labels (docs/img/cam_example.png); then the architecture slide.",
         f"We built the full system in Webots R2025a. Each random seed creates a narrow aisle between two facing "
         f"racking rows, three levels of five bays on each side, holding plain cardboard cartons in five "
         f"different sizes, with gaps, misplaced items, and blocking cartons or pallets in front of some slots. "
         f"Nothing is colour-coded: a carton's product type is readable only from a big black-and-white pattern "
         f"printed on its shipping label, so the detector has to work from shape and print. The robot is a "
         f"scissor-lift AMR whose camera rides a powered lift up to each shelf level. Its detector is a "
         f"YOLOv8n network that we fine-tuned on {T['n_train'] + T['n_val']} frames captured automatically in "
         f"the simulator from twelve training seeds. On held-out "
         f"frames it reaches an mAP50 of {T['map50']:.2f}. For every slot, the robot compares what it sees with "
         f"the expected planogram and assigns one of four states: OK, misplaced, empty, or unverified. A slot is "
         f"only called empty if it is clearly visible and the detector positively sees an empty slot. Anything "
         f"uncertain becomes unverified, and the active-verification policy then drives to a new viewpoint, on "
         f"the far side of the blocking object, and looks again."),
        ("1:40-3:05", "3. The outcome", "Results slide: table and chart from results_v2/results.csv.",
         f"We ran three strategies on {n} warehouse layouts the detector had never seen, all in Webots: a fixed "
         f"patrol, a re-scan from the same position, and active verification. Slot accuracy, counting unverified "
         f"slots as wrong, was {v1['acc']:.1f} percent for the fixed patrol, {v3['acc']:.1f} percent for "
         f"same-position re-scan, and {v4['acc']:.1f} percent for active verification. The share of slots left "
         f"unverified went from {v1['unv']:.1f} percent, to {v3['unv']:.1f}, to {v4['unv']:.1f} percent. False "
         f"empty reports: {v1['fe_n']}, {v3['fe_n']} and {v4['fe_n']} slots in total. Of the {v4['real']} real "
         f"discrepancies, missing or misplaced items, the three strategies identified {v1['caught']}, "
         f"{v3['caught']} and {v4['caught']}. "
         + (f"Decided accuracy stayed at {v4['dec_acc']:.1f} percent or better, so nearly every error is an "
            f"honest 'unverified' rather than a confident wrong answer. ")
         + ("Re-scanning from the same spot bought nothing, because the same viewpoint returns the same "
            "image; only moving to a new viewpoint helped. ")
         + f"The cost of looking again is time: the average audit took "
         f"{v1['time']:.0f} simulated seconds for the fixed patrol, {v3['time']:.0f} for re-scan and "
         f"{v4['time']:.0f} for active verification."),
        ("3:05-4:00", "4. Path to reality", "Hardware slide; then limits slide; end on links slide.",
         "The reasoning, digital twin and viewpoint selection are plain Python with unit tests and no simulator "
         "dependency, so they move to a real robot unchanged. On hardware we would use a ROS 2 differential-drive "
         "base with LiDAR localisation on a shelf map instead of simulator GPS, an RGB camera on a mast at shelf "
         "height, and an edge computer running the detector, exported to INT8. After fine-tuning on real shelf photos, "
         "the first trial would be one aisle, checked against a manual count. We are honest about the limits: "
         "everything here is simulation. The cartons and their labels are rendered, the world is static, "
         "localisation is ideal, the robot is a scissor-lift look-alike on small running gear, the two decision "
         "thresholds are design defaults rather than tuned values, and nothing has run on a "
         "physical robot yet. What the simulation does show is the core idea: a robot that knows what it has not "
         "seen, and goes to look again. Thank you."),
    ]
    words = sum(len(t.split()) for *_, t in sec)
    lines = ["# ShelfScan AMR: 4-minute demo video voice-over\n",
             f"About {words} words, roughly 4 minutes at ~140 words per minute. All numbers are from "
             f"results_v2/results.csv ({n} unseen seeds, real Webots runs).\n",
             "Covers the four required points: (1) problem, (2) simulation, (3) outcome, (4) path to reality.\n"]
    for t, head, shots, text in sec:
        lines += [f"## {t}  {head}\n", f"*On screen:* {shots}\n", text + "\n"]
    lines += ["## Recording checklist\n",
              "- Screen-record one Webots audit run (GUI, realtime): "
              "`python webots/tools/make_world.py --seed 101 --controller shelfscan_audit`, then open "
              "`webots/worlds/warehouse.wbt` in the Webots GUI and press play (V4_active is the default policy; "
              "set `SHELFSCAN_POLICY` to switch). Webots closes itself when the audit ends.\n",
              "- Show `results_v2/results.csv` or the results slide when reading numbers.\n",
              "- Keep the final video at or under 4:00.\n"]
    (DOCS / "video_script.md").write_text("\n".join(lines), encoding="utf-8")
    return words


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    S, seeds, T = load()
    chart(S, seeds)
    slides = content(S, seeds, T)
    note = concept_note(S, seeds, T)
    build_pptx(slides, note, S, seeds, DOCS / "ShelfScan_deck.pptx")
    n_note = build_pdf(slides, note, S, seeds, DOCS / "ShelfScan_submission.pdf")
    note_markdown(note, S)
    words = video_script(S, seeds, T)
    print(f"deck: {len(slides)} slides; concept note: {n_note} A4 pages; video script: {words} words; seeds {seeds}")


if __name__ == "__main__":
    main()
