"""Fine-tune YOLOv8n on the captured dataset, then export INT8 for edge use.

    pip install ultralytics
    python train/train_yolo.py --data dataset --epochs 60
    # v3, the QR/barcode world's 4-class detector
    python train/train_yolo.py --data datasets_v3 --classes v3 --epochs 30
        --imgsz 320 --device cpu --project models_v3 --name yolov8n_v3

Capture data from SEVERAL generated worlds (different seeds) so the model sees
varied layouts. Keep a few whole seeds out of training for the final evaluation.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shelfscan.robot_utils import class_set                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--classes", choices=["v2", "v3"], default="v2",
                    help="v2 = the 4 SKUs + slot_empty + blocker; v3 = "
                         "cardboard_box + label + slot_empty + blocker, for "
                         "the QR/barcode world")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. cpu or 0")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--export", action="store_true", help="export TFLite INT8 after training")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="shelfscan",
                    help="run name under --project; change it to keep an earlier run")
    a = ap.parse_args()

    root = Path(a.data).resolve()
    yaml = root / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(class_set(a.classes)))
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")

    from ultralytics import YOLO
    model = YOLO(a.weights)
    model.train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=str(Path(a.project).resolve()), plots=True,
                name=a.name, exist_ok=True,
                # colour/lighting augmentation reduces overfitting to the renderer
                hsv_h=0.02, hsv_s=0.5, hsv_v=0.5, degrees=3, translate=0.08, scale=0.3,
                fliplr=0.0)   # no flips: left/right occluder side carries meaning
    metrics = model.val()
    b = metrics.box
    print(f"\n{'class':<14}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    print(f"{'all':<14}{b.mp:>8.3f}{b.mr:>8.3f}{b.map50:>9.3f}{b.map:>10.3f}")
    for i, c in enumerate(b.ap_class_index):
        p_, r_, ap50, ap = b.class_result(i)
        print(f"{metrics.names[int(c)]:<14}{p_:>8.3f}{r_:>8.3f}"
              f"{ap50:>9.3f}{ap:>10.3f}")
    if a.export:
        model.export(format="tflite", int8=True, data=str(yaml))


if __name__ == "__main__":
    main()
