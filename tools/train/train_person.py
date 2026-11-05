"""Fine-tune a 1-class person detector with a chosen attention form.

Usage (training PC):
  python make_person_model.py yolo26n.pt yolo26n_person_init.pt
  python train_person.py DATA.yaml --model yolo26n_person_init.pt --attention sigmoid --name y26n_sig
Results: runs/<name>/weights/best.pt. Export with:
  python ../export_yolo26_onnx.py runs/<name>/weights/best.pt --attention <same value>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attention  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--model", default="yolo26n_person_init.pt")
    p.add_argument("--attention", choices=sorted(attention.FORWARDS), default="sigmoid")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)  # Windows + ROCm: in-process loading is the stable choice
    p.add_argument("--fraction", type=float, default=1.0, help="use part of the training set (smoke tests)")
    a = p.parse_args()

    attention.use(a.attention)
    from ultralytics import YOLO
    results = YOLO(a.model).train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
        project="runs", name=a.name, exist_ok=False, fraction=a.fraction, patience=30, plots=True)
    print("TRAIN_DONE", getattr(results, "save_dir", ""), flush=True)


if __name__ == "__main__":  # required on Windows
    main()
    # On the Windows/ROCm training PC the interpreter can hang during GPU teardown after the
    # work is done (seen 2026-11-06); exit without running finalizers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
