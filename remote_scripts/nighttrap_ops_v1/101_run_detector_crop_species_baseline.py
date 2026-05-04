#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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

import clip
import torch


ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list in {path}")
    return payload


def choice_label(item: dict[str, Any]) -> str:
    answer = item.get("answer")
    choices = item.get("choices") or []
    if not isinstance(answer, int) or answer < 0 or answer >= len(choices):
        return "missing"
    text = str(choices[answer]).strip()
    if len(text) > 3 and text[0] == "(" and text[2] == ")":
        text = text[3:].strip()
    return text


def item_paths(item: dict[str, Any]) -> list[str]:
    paths = [str(p) for p in (item.get("images") or []) if p]
    if not paths and item.get("image"):
        paths = [str(item["image"])]
    out = []
    seen = set()
    for path in paths:
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def numeric_event_id(sample_id: str) -> str | None:
    match = re.search(r"(\d+)$", sample_id or "")
    return match.group(1) if match else None


def load_detector(detector_jsonl: Path) -> dict[str, dict[str, Any]]:
    by_event: dict[str, dict[str, Any]] = {}
    with detector_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("event_key", ""))
            if key:
                by_event[key] = row
    return by_event


def best_box(det: dict[str, Any] | None, conf_min: float) -> tuple[list[float] | None, float | None, int]:
    if not det:
        return None, None, 0
    boxes = det.get("boxes") or []
    usable = []
    for box in boxes:
        conf = float(box.get("conf", 0.0) or 0.0)
        xyxy = box.get("xyxy")
        if xyxy and len(xyxy) == 4 and conf >= conf_min:
            usable.append((conf, [float(v) for v in xyxy]))
    if not usable:
        return None, float(det.get("target_confidence", 0.0) or 0.0), len(boxes)
    conf, xyxy = max(usable, key=lambda x: x[0])
    return xyxy, conf, len(boxes)


def crop_image(img: Image.Image, xyxy: list[float] | None, pad_frac: float) -> Image.Image:
    if xyxy is None:
        return img
    width, height = img.size
    x1, y1, x2, y2 = xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * pad_frac
    pad_y = bh * pad_frac
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return img
    return img.crop((left, top, right, bottom))


def build_manifest(
    splits: dict[str, list[dict[str, Any]]],
    detector: dict[str, dict[str, Any]],
    out_path: Path,
    conf_min: float,
) -> list[dict[str, Any]]:
    rows = []
    for split, items in splits.items():
        for item in items:
            sample_id = str(item.get("id"))
            event_id = numeric_event_id(sample_id)
            det = detector.get(event_id or "")
            xyxy, conf, boxes_count = best_box(det, conf_min)
            paths = item_paths(item)
            primary = paths[0] if paths else ""
            rows.append(
                {
                    "split": split,
                    "id": sample_id,
                    "event_key": event_id or "",
                    "gold": choice_label(item),
                    "primary_image": primary,
                    "n_images": len(paths),
                    "detector_has_row": bool(det),
                    "detector_has_target": bool(det.get("has_target")) if det else False,
                    "boxes_count": boxes_count,
                    "best_conf": "" if conf is None else f"{conf:.6f}",
                    "has_usable_box": xyxy is not None,
                    "fallback": "crop" if xyxy is not None else "full_frame",
                    "x1": "" if xyxy is None else f"{xyxy[0]:.2f}",
                    "y1": "" if xyxy is None else f"{xyxy[1]:.2f}",
                    "x2": "" if xyxy is None else f"{xyxy[2]:.2f}",
                    "y2": "" if xyxy is None else f"{xyxy[3]:.2f}",
                }
            )
    fieldnames = list(rows[0].keys()) if rows else []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_coverage(rows: list[dict[str, Any]], detector_jsonl: Path, out_path: Path) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({r["split"] for r in rows}):
        part = [r for r in rows if r["split"] == split]
        n = len(part)
        has_row = sum(bool(r["detector_has_row"]) for r in part)
        has_box = sum(bool(r["has_usable_box"]) for r in part)
        by_split[split] = {
            "n": n,
            "detector_row_n": has_row,
            "detector_row_rate": has_row / n if n else 0.0,
            "usable_box_n": has_box,
            "usable_box_rate": has_box / n if n else 0.0,
            "full_frame_fallback_n": n - has_box,
            "full_frame_fallback_rate": (n - has_box) / n if n else 0.0,
        }
    all_n = len(rows)
    all_has_box = sum(bool(r["has_usable_box"]) for r in rows)
    payload = {
        "detector_file": str(detector_jsonl),
        "unit": "species event sample; one best MegaDetector animal box per event when available",
        "fallback_policy": "Samples without a usable detector box are evaluated with full-frame CLIP embeddings, so the full species test set remains in scope.",
        "overall": {
            "n": all_n,
            "usable_box_n": all_has_box,
            "usable_box_rate": all_has_box / all_n if all_n else 0.0,
            "full_frame_fallback_n": all_n - all_has_box,
            "full_frame_fallback_rate": (all_n - all_has_box) / all_n if all_n else 0.0,
        },
        "by_split": by_split,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def encode_rows(
    rows: list[dict[str, Any]],
    model: Any,
    preprocess: Any,
    device: str,
    batch_size: int,
    pad_frac: float,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cache = {str(k): v.astype("float32") for k, v in data.items()}
    todo = [r for r in rows if str(r["id"]) not in cache]
    if todo:
        for start in tqdm(range(0, len(todo), batch_size), desc="encode detector crops"):
            batch = todo[start : start + batch_size]
            tensors = []
            ids = []
            for row in batch:
                try:
                    img = Image.open(str(row["primary_image"])).convert("RGB")
                    xyxy = None
                    if row["has_usable_box"]:
                        xyxy = [float(row[k]) for k in ["x1", "y1", "x2", "y2"]]
                    img = crop_image(img, xyxy, pad_frac)
                    tensors.append(preprocess(img))
                    ids.append(str(row["id"]))
                except Exception:
                    continue
            if not tensors:
                continue
            tensor = torch.stack(tensors).to(device)
            with torch.no_grad():
                feats = model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.detach().cpu().float().numpy()
            for sample_id, feat in zip(ids, arr):
                cache[sample_id] = feat.astype("float32")
        np.savez_compressed(cache_path, **cache)
    return cache


def majority_vote(labels: list[str], sims: np.ndarray, train_y: list[str], idx: np.ndarray) -> tuple[str, float]:
    votes: dict[str, float] = defaultdict(float)
    for rank, train_idx in enumerate(idx):
        label = train_y[int(train_idx)]
        # Similarity-weighted vote with tiny rank tie-breaker.
        votes[label] += float(sims[rank]) + 1e-6 * (len(idx) - rank)
    pred = max(votes.items(), key=lambda kv: kv[1])[0]
    return pred, votes[pred]


def run_knn(rows: list[dict[str, Any]], embeds: dict[str, np.ndarray], k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_rows = {split: [r for r in rows if r["split"] == split and str(r["id"]) in embeds] for split in ["train", "dev", "test"]}
    labels = sorted({r["gold"] for r in rows})
    train_rows = split_rows["train"]
    test_rows = split_rows["test"]
    x_train = np.stack([embeds[str(r["id"])] for r in train_rows]).astype("float32")
    x_test = np.stack([embeds[str(r["id"])] for r in test_rows]).astype("float32")
    y_train = [str(r["gold"]) for r in train_rows]
    y_test = [str(r["gold"]) for r in test_rows]
    sims = x_test @ x_train.T
    top_idx = np.argpartition(-sims, kth=min(k, x_train.shape[0] - 1), axis=1)[:, :k]
    preds = []
    pred_rows = []
    for i, row in enumerate(test_rows):
        order = top_idx[i][np.argsort(-sims[i, top_idx[i]])]
        pred, vote_score = majority_vote(labels, sims[i, order], y_train, order)
        preds.append(pred)
        pred_rows.append(
            {
                "id": row["id"],
                "gold": row["gold"],
                "pred": pred,
                "correct": pred == row["gold"],
                "score": f"{vote_score:.6f}",
                "fallback": row["fallback"],
                "has_usable_box": row["has_usable_box"],
                "best_conf": row["best_conf"],
            }
        )
    acc = accuracy_score(y_test, preds)
    macro = f1_score(y_test, preds, labels=labels, average="macro", zero_division=0)
    return pred_rows, {"test_accuracy": float(acc), "test_macro_f1": float(macro), "labels": labels, "y_test": y_test, "preds": preds}


def class_frequency_bins(train_rows: list[dict[str, Any]]) -> dict[str, str]:
    counts = Counter(str(r["gold"]) for r in train_rows)
    bins = {}
    for label, n in counts.items():
        if n >= 100:
            bins[label] = "head"
        elif n >= 20:
            bins[label] = "medium"
        else:
            bins[label] = "tail"
    return bins


def write_metrics(out_dir: Path, rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], metrics: dict[str, Any], k: int, coverage: dict[str, Any], clip_model: str) -> None:
    labels = metrics["labels"]
    y_test = metrics["y_test"]
    preds = metrics["preds"]
    cm = confusion_matrix(y_test, preds, labels=labels)

    with (out_dir / "species_predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(pred_rows[0].keys()) if pred_rows else ["id", "gold", "pred"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)

    per_class = []
    train_bins = class_frequency_bins([r for r in rows if r["split"] == "train"])
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(cm[i, :].sum())
        per_class.append({"class": label, "frequency_bin": train_bins.get(label, "tail"), "support": support, "precision": precision, "recall": recall, "f1": f1})
    with (out_dir / "per_class_f1.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "frequency_bin", "support", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(per_class)

    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gold\\pred"] + labels)
        for label, row in zip(labels, cm.tolist()):
            writer.writerow([label] + row)

    long_tail = []
    for bin_name in ["head", "medium", "tail"]:
        cls = [r["class"] for r in per_class if r["frequency_bin"] == bin_name]
        supports = [r["support"] for r in per_class if r["frequency_bin"] == bin_name]
        f1s = [r["f1"] for r in per_class if r["frequency_bin"] == bin_name]
        long_tail.append(
            {
                "frequency_bin": bin_name,
                "n_classes": len(cls),
                "test_support": int(sum(supports)),
                "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
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
        "model": "CLIP detector-crop kNN",
        "clip_model": clip_model,
        "detector": "MegaDetector v5a",
        "k": k,
        "test_n": len(y_test),
        "test_accuracy": metrics["test_accuracy"],
        "test_macro_f1": metrics["test_macro_f1"],
        "coverage": coverage,
        "fallback_policy": "full-frame embedding for samples without a usable detector box",
        "long_tail_breakdown": long_tail,
        "top_confused_species_pairs": confusions[:20],
        "outputs": {
            "crop_manifest": str(out_dir / "crop_manifest.csv"),
            "detection_coverage": str(out_dir / "detection_coverage.json"),
            "species_predictions": str(out_dir / "species_predictions.csv"),
            "per_class_f1": str(out_dir / "per_class_f1.csv"),
            "confusion_matrix": str(out_dir / "confusion_matrix.csv"),
            "long_tail_breakdown": str(out_dir / "long_tail_breakdown.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="<ANIMALLAMA_ROOT>")
    parser.add_argument("--species-root", default="results/nighttrap_ops_v1_build/track_b_species")
    parser.add_argument("--detector-jsonl", default="results/night_wildlife_detector_v1/night_all_68267_md_v5a.jsonl")
    parser.add_argument("--out-dir", default="results/nighttrap_supervised_baselines/detector_crop_retrieval_v1")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--box-conf-min", type=float, default=0.05)
    parser.add_argument("--crop-pad-frac", type=float, default=0.10)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    species_root = root / args.species_root
    detector_jsonl = root / args.detector_jsonl
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {split: load_json(species_root / f"{split}.json") for split in ["train", "dev", "test"]}
    detector = load_detector(detector_jsonl)
    rows = build_manifest(splits, detector, out_dir / "crop_manifest.csv", args.box_conf_min)
    coverage = write_coverage(rows, detector_jsonl, out_dir / "detection_coverage.json")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()
    embeds = encode_rows(rows, model, preprocess, device, args.batch_size, args.crop_pad_frac, out_dir / "clip_crop_or_fullframe_embeddings.npz")
    pred_rows, metrics = run_knn(rows, embeds, args.k)
    write_metrics(out_dir, rows, pred_rows, metrics, args.k, coverage, args.clip_model)
    print(json.dumps({"out_dir": str(out_dir), "test_accuracy": metrics["test_accuracy"], "test_macro_f1": metrics["test_macro_f1"], "coverage": coverage["by_split"]["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
