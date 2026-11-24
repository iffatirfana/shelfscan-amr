"""Fine-tune YOLOv8n on the captured dataset, then export INT8 for edge use.

    pip install ultralytics
    python train/train_yolo.py --data dataset --epochs 60

Capture data from SEVERAL generated worlds (different seeds) so the model sees
varied layouts. Keep a few whole seeds out of training for the final evaluation.
"""
import argparse
from pathlib import Path

CLASSES = ["sku_A", "sku_B", "sku_C", "sku_D", "slot_empty", "blocker"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
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
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
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
    print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
    if a.export:
        model.export(format="tflite", int8=True, data=str(yaml))


if __name__ == "__main__":
    main()
