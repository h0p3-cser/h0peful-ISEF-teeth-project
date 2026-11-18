"""Train or evaluate the Stage 1 YOLOv8n caries detector on prepared Zenodo data."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "public_rgb" / "full"
OUT_DIR = ROOT / "outputs" / "rgb_yolo_detection"
SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "caries"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dataset_config(data: Path) -> tuple[dict, dict]:
    """Resolve split paths against the actual location of the source YAML."""
    path = data / "data.yaml" if data.is_dir() else data
    path = path.resolve()
    original = yaml.safe_load(path.read_text(encoding="utf-8"))
    if original.get("names") not in (CLASS_NAMES, ["caries"]):
        raise ValueError("Prepared YOLO dataset must define a single caries class")

    # The source YAML stays portable. The generated runtime config contains
    # explicit paths and ignores any legacy path set before moving the data.
    runtime = {"path": path.parent.as_posix(), "names": CLASS_NAMES}
    counts = {}
    for split in SPLITS:
        img_dir = (path.parent / original[split]).resolve()
        lbl_dir = img_dir.parent / "labels"
        images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")}
        if img_dir.name != "images" or not images or len(labels) != len(images) or set(labels) != {p.stem for p in images}:
            raise ValueError(f"Missing images or mismatched labels in {split}: {img_dir}")
        positive = sum(bool(label.stat().st_size) for label in labels.values())
        counts[split] = {"images": len(images), "positive": positive, "background": len(images) - positive}
        runtime[split] = img_dir.as_posix()
        print(f"labels ready: {split} -> {counts[split]}", flush=True)
    return runtime, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET, help="prepared dataset directory or data.yaml")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="new output directory")
    parser.add_argument("--model", type=Path, default=Path("yolov8n.pt"), help="pretrained weights for training")
    parser.add_argument("--eval-only", action="store_true", help="evaluate a local checkpoint without training")
    parser.add_argument("--weights", type=Path, help="trusted trained checkpoint for --eval-only")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Ultralytics device (default: cpu)")
    args = parser.parse_args()

    if args.eval_only and (args.weights is None or not args.weights.is_file()):
        parser.error("--eval-only requires a trusted local checkpoint via --weights")
    if not args.eval_only and args.weights is not None:
        parser.error("--weights is for --eval-only; use --model to initialize training")
    if not args.eval_only and args.model != Path("yolov8n.pt") and not args.model.is_file():
        parser.error(f"Pretrained weights not found: {args.model}")
    if min(args.epochs, args.imgsz, args.batch) < 1 or args.workers < 0:
        parser.error("--epochs, --imgsz and --batch must be positive; --workers cannot be negative")
    out = args.out.resolve()
    if out.exists():
        parser.error(f"Choose a new --out directory: {out}")
    try:
        runtime, label_counts = dataset_config(args.data)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.error(f"Invalid prepared dataset: {exc}")

    out.mkdir(parents=True, exist_ok=False)
    config_dir = out / ".ultralytics"
    config_dir.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ.setdefault("NO_COLOR", "1")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    data_yaml = out / "dataset_runtime.yaml"
    data_yaml.write_text(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False), encoding="utf-8")

    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    if args.eval_only:
        checkpoint = args.weights.resolve()
    else:
        if args.model.is_file():
            initial_weights = args.model.resolve()
        else:
            initial_weights = out / "pretrained" / "yolov8n.pt"
            initial_weights.parent.mkdir()
        model = YOLO(str(initial_weights))
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            deterministic=True,
            device=args.device,
            workers=args.workers,
            project=str(out / "runs"),
            name="train",
            verbose=True,
        )
        checkpoint = out / "runs" / "train" / "weights" / "best.pt"

    metrics = YOLO(str(checkpoint)).val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(out / "runs"),
        name="test",
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "model": checkpoint.name if args.eval_only else args.model.name,
        "eval_only": args.eval_only,
        "epochs": None if args.eval_only else args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "classes": CLASS_NAMES,
        "dataset": "Zenodo 14827784 Dataset.zip (single-class localization)",
        "label_counts": label_counts,
        "test_metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }
    (out / "yolo_detection_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
