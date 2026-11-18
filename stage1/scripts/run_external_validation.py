"""Cross-dataset stress test for Stage 1 (RGB only).

Neither public dataset can validate the other's full label space, so this
script tests the one task both share: image-level "caries present / absent".
Two zero-shot directions are evaluated, with no fine-tuning on the target:

  A. CariesSpectra (close-up) -> Zenodo test (intraoral photo)
     The 3-class classifier's caries probability = p(early) + p(advanced).
     Target label = the image has at least one valid YOLO box.

  B. Zenodo (intraoral photo) -> CariesSpectra test (close-up)
     The YOLO detector is run on each close-up; score = max box confidence.
     Target label = early or advanced (i.e. any caries).

Both directions are deliberate domain-shift stress tests. A poor result is a
finding, not a bug: it quantifies how much the RGB baseline depends on the
acquisition protocol (field of view, working distance, lighting), which is
exactly the boundary the QLF arm is meant to complement.

Protocols for direction A:
  full_resize  : resize the whole photo to 224x224 (keeps all teeth, distorts
                 aspect ratio, shrinks lesions) -- primary protocol.
  center_crop  : resize shorter side to 224 then centre-crop (matches training
                 geometry, but discards lateral teeth) -- sensitivity check.

Outputs (outputs/external_validation/):
  external_validation_metrics.json
  external_validation_predictions.csv
  external_validation_summary.md
  external_validation_confusion.png

Usage:
  python scripts/run_external_validation.py --help
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms

import train_cariesspectra_3class as cs

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "external_validation"
ZENODO_ROOT = ROOT / "data" / "public_rgb" / "full"
CS_MANIFEST = ROOT / "data" / "cariesspectra" / "manifest.csv"
CS_WEIGHTS = ROOT / "outputs" / "cariesspectra_3class" / "best.pt"
YOLO_WEIGHTS = ROOT / "outputs" / "rgb_yolo_detection" / "runs" / "train" / "weights" / "best.pt"
IMGSZ = 224
YOLO_IMGSZ = 320
YOLO_THRESHOLDS = [0.05, 0.10, 0.25, 0.50]
PRIMARY_YOLO_THRESHOLD = 0.25


def binary_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    auroc = None
    if len(set(y_true.tolist())) == 2:
        auroc = round(float(roc_auc_score(y_true, y_score)), 4)
    return {
        "n": int(len(y_true)),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round((tp + tn) / max(1, len(y_true)), 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "npv": round(npv, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round((sens + spec) / 2, 4),
        "auroc": auroc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def build_classifier(weights):
    model = models.resnet18(weights=None)  # weights come from the local state dict
    model.fc = nn.Linear(model.fc.in_features, len(cs.CLASSES))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model


def classifier_scores(model, paths, transform, batch_size=32):
    scores = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            images = []
            for path in batch:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            xs = torch.stack(images)
            probs = torch.softmax(model(xs), dim=1).numpy()
            scores.extend((probs[:, 0] + probs[:, 1]).tolist())  # early + advanced
    return scores


def zenodo_test_items(root):
    images_dir = root / "test" / "images"
    labels_dir = root / "test" / "labels"
    items = []
    for img in sorted(images_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label = labels_dir / f"{img.stem}.txt"
        lines = [ln for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        items.append({"image": img, "label": 1 if lines else 0, "detail": f"boxes={len(lines)}"})
    return items


def cariesspectra_test_items(manifest):
    items = []
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            items.append(
                {
                    "image": manifest.parent / row["image_path"],
                    "label": 0 if row["class"] == "no_caries" else 1,
                    "class": row["class"],
                    "detail": row["class"],
                }
            )
    return items


def yolo_scores(paths, weights):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    max_scores, box_counts = [], []
    for i, r in enumerate(
        model.predict(
            source=[str(p) for p in paths],
            imgsz=YOLO_IMGSZ,
            conf=0.01,
            device="cpu",
            verbose=False,
            stream=True,
        )
    ):
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.array([])
        max_scores.append(float(confs.max()) if len(confs) else 0.0)
        box_counts.append(int(len(confs)))
        if (i + 1) % 50 == 0:
            print(f"  yolo {i + 1}/{len(paths)}", flush=True)
    return max_scores, box_counts


def plot_confusions(a_cm, b_cm, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    panels = [
        (
            axes[0],
            a_cm,
            "A: CariesSpectra model on Zenodo test",
            "Predicted",
            "Reference",
        ),
        (
            axes[1],
            b_cm,
            "B: Zenodo model on CariesSpectra test",
            "Predicted",
            "Reference",
        ),
    ]
    for ax, cm, title, xlabel, ylabel in panels:
        arr = np.array(cm)
        ax.imshow(arr, cmap="Oranges")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No caries", "Caries"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No caries", "Caries"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        thresh = arr.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > thresh else "black", fontsize=13,
                )
    fig.suptitle("Stage 1: Cross-dataset caries detection without fine-tuning", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 zero-shot cross-dataset evaluation.")
    parser.add_argument("--zenodo", type=Path, default=ZENODO_ROOT, help="prepared detection dataset root")
    parser.add_argument("--cariesspectra", type=Path, default=CS_MANIFEST, help="prepared classification manifest")
    parser.add_argument("--classifier-weights", type=Path, default=CS_WEIGHTS)
    parser.add_argument("--detector-weights", type=Path, default=YOLO_WEIGHTS)
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="new output directory")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    args = parser.parse_args()
    for path in (args.classifier_weights, args.detector_weights, args.cariesspectra):
        if not path.is_file():
            parser.error(f"Required file does not exist: {path}")
    if args.threads < 1:
        parser.error("--threads must be positive")
    for split_dir in ("images", "labels"):
        if not (args.zenodo / "test" / split_dir).is_dir():
            parser.error(f"Missing prepared test/{split_dir} in {args.zenodo}")
    zen = zenodo_test_items(args.zenodo.resolve())
    cs_items = cariesspectra_test_items(args.cariesspectra.resolve())
    if not zen or not cs_items:
        parser.error("Both datasets need nonempty test sets")
    torch.set_num_threads(args.threads)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config_dir = out / ".ultralytics"
    config_dir.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

    # ---------- Direction A ----------
    z_paths = [it["image"] for it in zen]
    z_true = [it["label"] for it in zen]
    print(f"A: CariesSpectra -> Zenodo test, n={len(z_paths)}", flush=True)
    model = build_classifier(args.classifier_weights)
    protocols = {
        "full_resize": transforms.Compose(
            [
                transforms.Resize((IMGSZ, IMGSZ)),
                transforms.ToTensor(),
                transforms.Normalize(cs.IMAGENET_MEAN, cs.IMAGENET_STD),
            ]
        ),
        "center_crop": transforms.Compose(
            [
                transforms.Resize(IMGSZ),
                transforms.CenterCrop(IMGSZ),
                transforms.ToTensor(),
                transforms.Normalize(cs.IMAGENET_MEAN, cs.IMAGENET_STD),
            ]
        ),
    }
    a = {"direction": "CariesSpectra -> Zenodo test", "protocols": {}}
    a_predictions = []
    for name, tf in protocols.items():
        scores = classifier_scores(model, z_paths, tf)
        a["protocols"][name] = {
            "metrics_at_0_5": binary_metrics(z_true, scores, 0.5),
            "metrics_at_0_3": binary_metrics(z_true, scores, 0.3),
        }
        if name == "full_resize":
            for it, s in zip(zen, scores):
                a_predictions.append(
                    {
                        "direction": "A",
                        "protocol": name,
                        "image": it["image"].name,
                        "true_label": it["label"],
                        "score": round(s, 6),
                        "threshold": 0.5,
                        "prediction": int(s >= 0.5),
                        "detail": it["detail"],
                    }
                )

    # ---------- Direction B ----------
    cs_paths = [it["image"] for it in cs_items]
    cs_true = [it["label"] for it in cs_items]
    print(f"B: Zenodo YOLO -> CariesSpectra test, n={len(cs_paths)}", flush=True)
    scores_b, boxes_b = yolo_scores(cs_paths, args.detector_weights)
    b = {"direction": "Zenodo YOLO -> CariesSpectra test", "by_threshold": {}}
    for thr in YOLO_THRESHOLDS:
        b["by_threshold"][str(thr)] = binary_metrics(cs_true, scores_b, thr)
    # per-class detection rate at the primary threshold (useful for staging)
    per_class = {}
    for cls in ("early", "advanced", "no_caries"):
        idx = [i for i, it in enumerate(cs_items) if it["class"] == cls]
        detected = sum(1 for i in idx if scores_b[i] >= PRIMARY_YOLO_THRESHOLD)
        per_class[cls] = {
            "n": len(idx),
            "detected": detected,
            "rate": round(detected / max(1, len(idx)), 4),
        }
    b["per_class_detection_at_0_25"] = per_class
    for it, s, nb in zip(cs_items, scores_b, boxes_b):
        a_predictions.append(
            {
                "direction": "B",
                "protocol": f"yolo_conf_{PRIMARY_YOLO_THRESHOLD}",
                "image": it["image"].name,
                "true_label": it["label"],
                "score": round(s, 6),
                "threshold": PRIMARY_YOLO_THRESHOLD,
                "prediction": int(s >= PRIMARY_YOLO_THRESHOLD),
                "detail": f"{it['detail']};boxes={nb}",
            }
        )

    b_primary = b["by_threshold"][str(PRIMARY_YOLO_THRESHOLD)]
    summary = {
        "purpose": "strict zero-shot cross-dataset external validation of the RGB baseline",
        "no_fine_tuning_on_target": True,
        "A_cariesspectra_to_zenodo": a,
        "B_zenodo_yolo_to_cariesspectra": b,
        "interpretation": (
            "Both directions are deliberately harsh domain-shift tests. Low scores show that "
            "the RGB baseline's apparent performance is tied to its own acquisition protocol, "
            "not that the model implementation is wrong. This is the gap a second modality "
            "must be validated against."
        ),
    }
    (out / "external_validation_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out / "external_validation_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["direction", "protocol", "image", "true_label", "score", "threshold", "prediction", "detail"],
        )
        writer.writeheader()
        writer.writerows(a_predictions)

    a_primary = a["protocols"]["full_resize"]["metrics_at_0_5"]
    plot_confusions(
        a_primary["confusion_matrix"],
        b_primary["confusion_matrix"],
        out / "external_validation_confusion.png",
    )

    md = [
        "# 终点一 跨数据集外部验证（零样本）",
        "",
        "目标：检验 RGB 基线换一个采集来源、不做任何微调时还能不能成立。",
        "两个数据集共享的可评估任务只有图像级「有无龋」，因此这是二元外部验证，不是分期外部验证。",
        "",
        "## A. CariesSpectra 三分类模型 → Zenodo test（特写 → 口内照）",
        "",
        f"- 样本 {a_primary['n']}（有龋 {a_primary['positives']} / 无龋 {a_primary['negatives']}）",
        f"- 主协议 full_resize，阈值 0.5：准确率 {a_primary['accuracy']:.3f}，"
        f"敏感度 {a_primary['sensitivity']:.3f}，特异度 {a_primary['specificity']:.3f}，"
        f"F1 {a_primary['f1']:.3f}，AUROC {a_primary['auroc']}",
        f"- 敏感性检查 center_crop：准确率 "
        f"{a['protocols']['center_crop']['metrics_at_0_5']['accuracy']:.3f}，"
        f"敏感度 {a['protocols']['center_crop']['metrics_at_0_5']['sensitivity']:.3f}，"
        f"特异度 {a['protocols']['center_crop']['metrics_at_0_5']['specificity']:.3f}",
        "",
        "## B. Zenodo YOLO → CariesSpectra test（口内照 → 特写）",
        "",
        f"- 样本 {b_primary['n']}（有龋 {b_primary['positives']} / 无龋 {b_primary['negatives']}）",
        f"- 阈值 {PRIMARY_YOLO_THRESHOLD}：准确率 {b_primary['accuracy']:.3f}，"
        f"敏感度 {b_primary['sensitivity']:.3f}，特异度 {b_primary['specificity']:.3f}，"
        f"F1 {b_primary['f1']:.3f}，AUROC {b_primary['auroc']}",
        "- 各阶段检出率（阈值 0.25）："
        + "，".join(
            f"{cs.CLASS_CN[k]} {v['detected']}/{v['n']}（{v['rate']:.1%}）"
            for k, v in per_class.items()
        ),
        "",
        "## 解释口径",
        "",
        "- 这是**零样本跨域压力测试**，不是临床外部验证，也不是对模型实现错误的证明。",
        "- 结果差主要说明：RGB 基线的表现强依赖采集协议（视场、距离、光照、构图）；",
        "  特写与口内照之间不能直接互换，这与“需要补第二模态并统一采集规范”的立项逻辑一致。",
        "- 不能把本结果写成“RGB 无效”：在同域测试集上检测 mAP50 0.918、三分类准确率 0.873 仍然成立。",
        "- 也不能把本结果写成“外部验证通过”。正式外部验证需要同一任务、同一标签体系、独立采集的第三方数据。",
        "",
        "明细见同目录 `external_validation_predictions.csv`。",
    ]
    (out / "external_validation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "A_full_resize_at_0.5": a_primary,
            "A_center_crop_at_0.5": a["protocols"]["center_crop"]["metrics_at_0_5"],
            "B_yolo_at_0.25": b_primary,
            "B_per_class": per_class,
        },
        ensure_ascii=False, indent=2,
    ))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
