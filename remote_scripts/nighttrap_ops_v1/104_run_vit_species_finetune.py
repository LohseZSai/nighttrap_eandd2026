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
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16
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


def primary_image(item: dict[str, Any]) -> str:
    if item.get("image"):
        return str(item["image"])
    images = item.get("images") or []
    return str(images[0]) if images else ""


class SpeciesDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], label_to_idx: dict[str, int], transform: Any) -> None:
        self.rows = []
        for item in items:
            path = primary_image(item)
            label = choice_label(item)
            if path and label in label_to_idx:
                self.rows.append({"id": str(item.get("id")), "image": path, "label": label, "target": label_to_idx[label]})
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        row = self.rows[idx]
        try:
            img = Image.open(row["image"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        return self.transform(img), int(row["target"]), row["id"], row["label"]


def evaluate(model: nn.Module, loader: DataLoader, idx_to_label: list[str], device: str) -> tuple[float, float, list[dict[str, Any]], list[str], list[str]]:
    model.eval()
    preds: list[str] = []
    golds: list[str] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets, ids, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            conf, pred_idx = probs.max(dim=-1)
            for sample_id, gold, pred_i, score in zip(ids, labels, pred_idx.cpu().tolist(), conf.cpu().tolist()):
                pred = idx_to_label[int(pred_i)]
                golds.append(str(gold))
                preds.append(pred)
                rows.append({"id": str(sample_id), "gold": str(gold), "pred": pred, "correct": pred == str(gold), "score": f"{score:.6f}"})
    labels_all = sorted(set(golds) | set(preds) | set(idx_to_label))
    acc = float(accuracy_score(golds, preds))
    macro = float(f1_score(golds, preds, labels=labels_all, average="macro", zero_division=0))
    return acc, macro, rows, golds, preds


def frequency_bins(train_items: list[dict[str, Any]]) -> dict[str, str]:
    counts = Counter(choice_label(item) for item in train_items)
    bins = {}
    for label, n in counts.items():
        if n >= 100:
            bins[label] = "head"
        elif n >= 20:
            bins[label] = "medium"
        else:
            bins[label] = "tail"
    return bins


def write_outputs(
    out_dir: Path,
    train_items: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    y_test: list[str],
    preds: list[str],
    summary_extra: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(y_test) | set(preds) | set(choice_label(x) for x in train_items))
    cm = confusion_matrix(y_test, preds, labels=labels)
    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "gold", "pred", "correct", "score"])
        writer.writeheader()
        writer.writerows(pred_rows)
    bins = frequency_bins(train_items)
    per_class = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({"class": label, "frequency_bin": bins.get(label, "tail"), "support": int(cm[i, :].sum()), "precision": precision, "recall": recall, "f1": f1})
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
        long_tail.append({"frequency_bin": bin_name, "n_classes": len(part), "test_support": int(sum(r["support"] for r in part)), "macro_f1": float(np.mean([r["f1"] for r in part])) if part else 0.0})
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
        "model": "ViT-B/16 supervised fine-tune",
        "input": "primary full-frame event image",
        "test_n": len(y_test),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_macro_f1": float(f1_score(y_test, preds, labels=labels, average="macro", zero_division=0)),
        "long_tail_breakdown": long_tail,
        "top_confused_species_pairs": confusions[:20],
        "extra": summary_extra,
        "outputs": {
            "predictions": str(out_dir / "predictions.csv"),
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
    parser.add_argument("--out-dir", default="results/nighttrap_supervised_baselines/vit_b16_finetune_species_v1")
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    root = Path(args.root)
    out_dir = root / args.out_dir
    species_root = root / args.species_root
    train_items = load_json(species_root / "train.json")
    dev_items = load_json(species_root / "dev.json")
    test_items = load_json(species_root / "test.json")
    labels = sorted({choice_label(x) for x in train_items})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    train_ds = SpeciesDataset(train_items, label_to_idx, transform)
    dev_ds = SpeciesDataset(dev_items, label_to_idx, transform)
    test_ds = SpeciesDataset(test_items, label_to_idx, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(labels))
    model.to(device)

    counts = Counter(choice_label(item) for item in train_items)
    class_weights = torch.tensor([1.0 / max(1, counts[label]) for label in labels], dtype=torch.float32)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    history = []
    best_state = None
    best_dev = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets, _, _ in tqdm(train_loader, desc=f"train epoch {epoch}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        dev_acc, dev_macro, _, _, _ = evaluate(model, dev_loader, labels, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_accuracy": dev_acc, "dev_macro_f1": dev_macro}
        history.append(row)
        if dev_macro > best_dev:
            best_dev = dev_macro
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(json.dumps(row, ensure_ascii=False))
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_macro, pred_rows, y_test, preds = evaluate(model, test_loader, labels, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels, "history": history}, out_dir / "best_model.pt")
    write_outputs(
        out_dir,
        train_items,
        pred_rows,
        y_test,
        preds,
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "dev_history": history,
            "best_dev_macro_f1": best_dev,
            "test_accuracy_check": test_acc,
            "test_macro_f1_check": test_macro,
        },
    )
    print((out_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
