#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def early_cuda_visible_devices() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    for idx, arg in enumerate(sys.argv):
        if arg == "--gpu" and idx + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[idx + 1]
            return
        if arg.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return


early_cuda_visible_devices()

import torch
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["has_usable_box"] = str(row.get("has_usable_box", "")).lower() == "true"
            rows.append(row)
    return rows


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {str(k): v.astype("float32") for k, v in data.items()}


def encode_fullframe_dinov2(
    rows: list[dict[str, Any]],
    model_name: str,
    device: str,
    batch_size: int,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    cache = load_embeddings(cache_path) if cache_path.exists() else {}
    todo = [r for r in rows if str(r["id"]) not in cache]
    if not todo:
        return cache
    model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    for start in tqdm(range(0, len(todo), batch_size), desc=f"encode full-frame {model_name}"):
        batch = todo[start : start + batch_size]
        tensors, ids = [], []
        for row in batch:
            try:
                img = Image.open(str(row["primary_image"])).convert("RGB")
                tensors.append(transform(img))
                ids.append(str(row["id"]))
            except Exception:
                continue
        if not tensors:
            continue
        tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feat = model(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        arr = feat.detach().cpu().float().numpy()
        for sample_id, vec in zip(ids, arr):
            cache[sample_id] = vec.astype("float32")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cache)
    return cache


def split_xy(rows: list[dict[str, Any]], embeddings: dict[str, np.ndarray], split: str):
    kept = [r for r in rows if r["split"] == split and str(r["id"]) in embeddings]
    x = np.stack([embeddings[str(r["id"])] for r in kept]).astype("float32")
    y = [str(r["gold"]) for r in kept]
    return x, y, kept


def tune_linear(x_train, y_train, x_dev, y_dev):
    best = None
    grid = []
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2500, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_dev)
        row = {
            "C": c,
            "dev_accuracy": float(accuracy_score(y_dev, pred)),
            "dev_macro_f1": float(f1_score(y_dev, pred, average="macro", zero_division=0)),
        }
        grid.append(row)
        if best is None or row["dev_macro_f1"] > best[0]["dev_macro_f1"]:
            best = (row, clf)
    assert best is not None
    return best[1], {"selected": best[0], "grid": grid}


def frequency_bins(rows: list[dict[str, Any]]) -> dict[str, str]:
    counts = Counter(str(r["gold"]) for r in rows if r["split"] == "train")
    bins = {}
    for label, n in counts.items():
        if n >= 100:
            bins[label] = "head"
        elif n >= 20:
            bins[label] = "medium"
        else:
            bins[label] = "tail"
    return bins


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], test_rows, y_test, preds, scores, model: str, method: str, extra: dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(y_test) | set(preds) | {str(r["gold"]) for r in rows})
    cm = confusion_matrix(y_test, preds, labels=labels)
    pred_rows = []
    for row, gold, pred, score in zip(test_rows, y_test, preds, scores):
        pred_rows.append(
            {
                "id": row["id"],
                "gold": gold,
                "pred": pred,
                "correct": gold == pred,
                "score": f"{score:.6f}",
                "has_usable_box": row.get("has_usable_box", False),
                "best_conf": row.get("best_conf", ""),
                "primary_image": row.get("primary_image", ""),
            }
        )
    with (out_dir / "species_predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)
    bins = frequency_bins(rows)
    per_class = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append(
            {
                "class": label,
                "frequency_bin": bins.get(label, "tail"),
                "support": int(cm[i, :].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    with (out_dir / "per_class_f1.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "frequency_bin", "support", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(per_class)
    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gold\\pred"] + labels)
        for label, vals in zip(labels, cm.tolist()):
            writer.writerow([label] + vals)
    long_tail = []
    for bin_name in ["head", "medium", "tail"]:
        part = [r for r in per_class if r["frequency_bin"] == bin_name]
        long_tail.append(
            {
                "frequency_bin": bin_name,
                "n_classes": len(part),
                "test_support": int(sum(r["support"] for r in part)),
                "macro_f1": float(np.mean([r["f1"] for r in part])) if part else 0.0,
            }
        )
    with (out_dir / "long_tail_breakdown.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frequency_bin", "n_classes", "test_support", "macro_f1"])
        writer.writeheader()
        writer.writerows(long_tail)
    confusions = []
    for i, gold in enumerate(labels):
        for j, pred in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusions.append({"gold": gold, "pred": pred, "count": int(cm[i, j])})
    confusions.sort(key=lambda r: r["count"], reverse=True)
    summary = {
        "task": "Species classification",
        "model": model,
        "method": method,
        "test_n": len(y_test),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_macro_f1": float(f1_score(y_test, preds, labels=labels, average="macro", zero_division=0)),
        "long_tail_breakdown": long_tail,
        "top_confused_species_pairs": confusions[:20],
        "extra": extra,
        "outputs": {
            "species_predictions": str(out_dir / "species_predictions.csv"),
            "per_class_f1": str(out_dir / "per_class_f1.csv"),
            "confusion_matrix": str(out_dir / "confusion_matrix.csv"),
            "long_tail_breakdown": str(out_dir / "long_tail_breakdown.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_linear(rows, embeddings, out_dir: Path, model_label: str, extra: dict[str, Any]):
    x_train, y_train, _ = split_xy(rows, embeddings, "train")
    x_dev, y_dev, _ = split_xy(rows, embeddings, "dev")
    x_test, y_test, test_rows = split_xy(rows, embeddings, "test")
    clf, tuning = tune_linear(x_train, y_train, x_dev, y_dev)
    preds = [str(x) for x in clf.predict(x_test)]
    scores = list(np.max(clf.predict_proba(x_test), axis=1))
    return write_outputs(
        out_dir,
        rows,
        test_rows,
        y_test,
        preds,
        scores,
        model_label,
        "StandardScaler + LogisticRegression(class_weight=balanced)",
        {**extra, "tuning": tuning},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/nighttrap_supervised_baselines/detector_crop_retrieval_v1/crop_manifest.csv")
    parser.add_argument("--crop-embeddings", default="results/nighttrap_supervised_baselines/detector_crop_dinov2_knn_v1/dinov2_crop_or_fullframe_embeddings.npz")
    parser.add_argument("--fullframe-embeddings", default="results/nighttrap_supervised_baselines/fullframe_dinov2_linear_species_v1/dinov2_fullframe_embeddings.npz")
    parser.add_argument("--crop-linear-out", default="results/nighttrap_supervised_baselines/detector_crop_dinov2_linear_v1")
    parser.add_argument("--fullframe-linear-out", default="results/nighttrap_supervised_baselines/fullframe_dinov2_linear_species_v1")
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    parser.add_argument("--tasks", default="crop_linear,fullframe_linear")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    rows = read_manifest(Path(args.manifest))
    requested = {x.strip() for x in args.tasks.split(",") if x.strip()}
    outputs = {}
    if "crop_linear" in requested:
        crop_embeddings = load_embeddings(Path(args.crop_embeddings))
        outputs["crop_linear"] = run_linear(
            rows,
            crop_embeddings,
            Path(args.crop_linear_out),
            "DINOv2 detector-crop linear classifier",
            {"dinov2_model": args.dinov2_model, "input": "detector crop with full-frame fallback"},
        )
    if "fullframe_linear" in requested:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        full_embeddings = encode_fullframe_dinov2(rows, args.dinov2_model, device, args.batch_size, Path(args.fullframe_embeddings))
        outputs["fullframe_linear"] = run_linear(
            rows,
            full_embeddings,
            Path(args.fullframe_linear_out),
            "DINOv2 full-frame linear classifier",
            {"dinov2_model": args.dinov2_model, "input": "primary full-frame event image"},
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
