"""Train a 3-class RGB classifier on Caries-Spectra (normal/early/advanced).

Reports the held-out confusion matrix for early, advanced, and no enamel caries.

Uses a torchvision ResNet18 (ImageNet pretrained) with frozen backbone parameters
and a trainable three-class head. BatchNorm statistics update as in the baseline.

Usage:
  python scripts/prepare_cariesspectra.py --src <source> --out data/cariesspectra
  python scripts/train_cariesspectra_3class.py --manifest data/cariesspectra/manifest.csv --out outputs/cariesspectra_3class
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "cariesspectra" / "manifest.csv"
OUT_DIR = ROOT / "outputs" / "cariesspectra_3class"

CLASSES = ["early", "advanced", "no_caries"]
CLASS_CN = {"early": "早期龋", "advanced": "明显龋", "no_caries": "正常"}
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DS(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        image_path, label = self.rows[i]
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB")), CLASS_IDX[label]


def load_splits(manifest, required_splits):
    splits = {"train": [], "val": [], "test": []}
    seen = set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "class", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must contain image_path, class, and split columns")
        for row in reader:
            path = (manifest.parent / row["image_path"]).resolve()
            split = row["split"]
            label = row["class"]
            if split not in splits or label not in CLASS_IDX:
                raise ValueError(f"Invalid class or split at manifest line {reader.line_num}")
            if path in seen:
                raise ValueError(f"Repeated image path at manifest line {reader.line_num}: {path}")
            seen.add(path)
            if split in required_splits and not path.is_file():
                raise FileNotFoundError(f"Missing image at manifest line {reader.line_num}: {path}")
            splits[split].append((path, label))
    for split in required_splits:
        if {label for _, label in splits[split]} != set(CLASSES):
            raise ValueError(f"Split {split!r} must contain all three classes")
    return splits


def build_model(pretrained=True, backbone_weights=None):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained and backbone_weights is None else None
    model = models.resnet18(weights=weights)
    if backbone_weights is not None:
        model.load_state_dict(torch.load(backbone_weights, map_location="cpu", weights_only=True))
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            p = torch.softmax(out, dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(out.argmax(1).tolist())
            y_prob.append(p.cpu().numpy())
    y_prob = np.concatenate(y_prob, axis=0) if y_prob else np.empty((0, len(CLASSES)))
    return y_true, y_pred, y_prob


def metrics(y_true, y_pred, y_prob=None):
    n = len(CLASSES)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    total = len(y_true)
    per_class = {}
    f1s, sens, specs = [], [], []
    for c in range(n):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n)) - tp
        fn = sum(cm[c][r] for r in range(n)) - tp
        tn = total - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        per_class[CLASSES[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "sensitivity": round(rec, 4),
            "specificity": round(specificity, 4),
        }
        f1s.append(f1)
        sens.append(rec)
        specs.append(specificity)
    acc = sum(cm[i][i] for i in range(n)) / max(1, total)
    roc_auc = None
    if y_prob is not None and len(y_prob) == total and total > 0:
        try:
            roc_auc = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(n)))), 4)
        except ValueError:
            roc_auc = None
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(sum(f1s) / n, 4),
        "macro_sensitivity": round(sum(sens) / n, 4),
        "macro_specificity": round(sum(specs) / n, 4),
        "roc_auc_ovr_macro": roc_auc,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def plot_confusion(cm, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Early caries", "Advanced caries", "No enamel caries"]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title("Stage 1: RGB caries-stage classification")
    thresh = max(1, max(max(r) for r in cm)) / 2
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="New result directory")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    ap.add_argument("--weights", type=Path, help="Local three-class checkpoint for --eval-only")
    ap.add_argument("--backbone-weights", type=Path, help="Optional local torchvision ResNet18 ImageNet V1 state dict")
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate a local three-class checkpoint without retraining",
    )
    args = ap.parse_args()
    if min(args.epochs, args.batch, args.imgsz, args.threads) < 1 or args.lr <= 0:
        ap.error("epochs, batch, imgsz, threads and lr must be positive")
    if args.eval_only and (args.weights is None or not args.weights.is_file()):
        ap.error("--eval-only requires an existing --weights file")
    if args.eval_only and args.backbone_weights is not None:
        ap.error("--backbone-weights applies to training only")
    if args.weights is not None and not args.eval_only:
        ap.error("Use --eval-only with --weights")
    if args.backbone_weights is not None and not args.backbone_weights.is_file():
        ap.error(f"Backbone weights do not exist: {args.backbone_weights}")
    manifest = args.manifest.resolve()
    output = args.out.resolve()
    required_splits = ("test",) if args.eval_only else ("train", "val", "test")
    if output.exists():
        ap.error(f"Output directory already exists: {output}")
    try:
        splits = load_splits(manifest, required_splits)
    except (OSError, ValueError) as error:
        ap.error(str(error))

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output.mkdir(parents=True, exist_ok=False)

    # Resize(int) keeps aspect ratio (shorter side -> imgsz); CenterCrop then makes
    # it square without stretching. For this dataset (already 224x224) it is a no-op.
    tr = transforms.Compose([
        transforms.Resize(args.imgsz),
        transforms.CenterCrop(args.imgsz),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ev = transforms.Compose([
        transforms.Resize(args.imgsz),
        transforms.CenterCrop(args.imgsz),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_loader = DataLoader(DS(splits["test"], ev), batch_size=args.batch, shuffle=False)
    print(" ".join(f"{split}={len(splits[split])}" for split in required_splits), flush=True)

    device = "cpu"
    model = build_model(pretrained=not args.eval_only, backbone_weights=args.backbone_weights).to(device)
    best = {"f1": -1.0, "epoch": -1}
    history = []
    if args.eval_only:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        best = {"f1": None, "epoch": None}
    else:
        train_loader = DataLoader(DS(splits["train"], tr), batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(DS(splits["val"], ev), batch_size=args.batch, shuffle=False)
        opt = torch.optim.AdamW(model.fc.parameters(), lr=args.lr)
        lossfn = nn.CrossEntropyLoss()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, correct, running = 0, 0, 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = lossfn(out, y)
                loss.backward()
                opt.step()
                total += y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                running += loss.item() * y.size(0)
            yv, pv, _ = evaluate(model, val_loader, device)
            vm = metrics(yv, pv)
            if vm["macro_f1"] > best["f1"]:
                best = {"f1": vm["macro_f1"], "epoch": epoch}
                torch.save(model.state_dict(), output / "best.pt")
            train_loss = running / total
            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(correct / total, 4),
                "val_acc": vm["accuracy"],
                "val_macro_f1": vm["macro_f1"],
            })
            print(f"epoch {epoch}/{args.epochs} loss={train_loss:.4f} train_acc={correct/total:.3f} "
                  f"val_acc={vm['accuracy']:.3f} val_f1={vm['macro_f1']:.3f}", flush=True)
        model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    yt, pt, yprob = evaluate(model, test_loader, device)
    m = metrics(yt, pt, yprob)
    plot_confusion(m["confusion_matrix"], output / "rgb_capability_boundary.png")

    summary = {
        "model": "resnet18 (ImageNet pretrained, frozen backbone + 3-class head)",
        "classes": CLASSES,
        "class_zh": CLASS_CN,
        "epochs": None if args.eval_only else args.epochs,
        "batch": args.batch,
        "image_size": args.imgsz,
        "learning_rate": None if args.eval_only else args.lr,
        "seed": args.seed,
        "cpu_threads": args.threads,
        "split_images": {name: len(rows) for name, rows in splits.items()},
        "eval_only": args.eval_only,
        "best_epoch": best["epoch"],
        "test_metrics": m,
        "history": history,
    }
    (output / "cariesspectra_3class_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["test_metrics"], ensure_ascii=False, indent=2))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
