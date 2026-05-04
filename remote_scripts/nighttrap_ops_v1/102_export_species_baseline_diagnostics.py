#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def frequency_bins(train_items: list[dict[str, Any]]) -> dict[str, str]:
    counts = Counter(choice_label(item) for item in train_items)
    out = {}
    for label, n in counts.items():
        if n >= 100:
            out[label] = "head"
        elif n >= 20:
            out[label] = "medium"
        else:
            out[label] = "tail"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="<ANIMALLAMA_ROOT>")
    parser.add_argument("--species-root", default="results/nighttrap_ops_v1_build/track_b_species")
    parser.add_argument("--baseline-dir", default="results/nighttrap_supervised_baselines/clip_linear_probe_v1/species")
    parser.add_argument("--out-dir", default="results/nighttrap_supervised_baselines/fullframe_clip_linear_species_v1")
    args = parser.parse_args()

    root = Path(args.root)
    species_root = root / args.species_root
    baseline_dir = root / args.baseline_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_json(species_root / "train.json")
    preds_in = load_predictions(baseline_dir / "predictions.jsonl")
    labels = sorted({row["gold"] for row in preds_in} | {row["pred"] for row in preds_in})
    y_true = [row["gold"] for row in preds_in]
    y_pred = [row["pred"] for row in preds_in]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    bins = frequency_bins(train)

    pred_rows = []
    for row in preds_in:
        pred_rows.append({"id": row["id"], "gold": row["gold"], "pred": row["pred"], "correct": row["gold"] == row["pred"]})
    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "gold", "pred", "correct"])
        writer.writeheader()
        writer.writerows(pred_rows)

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
        for label, row in zip(labels, cm.tolist()):
            writer.writerow([label] + row)

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
    confusions.sort(key=lambda row: row["count"], reverse=True)

    summary_in = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "task": "Species classification",
        "model": "CLIP full-frame/event linear probe",
        "source_summary": str(baseline_dir / "summary.json"),
        "test_n": len(y_true),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "reported_source_accuracy": summary_in.get("test_accuracy"),
        "reported_source_macro_f1": summary_in.get("test_macro_f1"),
        "long_tail_breakdown": long_tail,
        "top_confused_species_pairs": confusions[:20],
        "outputs": {
            "predictions": str(out_dir / "predictions.csv"),
            "per_class_f1": str(out_dir / "per_class_f1.csv"),
            "confusion_matrix": str(out_dir / "confusion_matrix.csv"),
            "long_tail_breakdown": str(out_dir / "long_tail_breakdown.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
