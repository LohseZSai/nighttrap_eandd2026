#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score


def load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list in {path}")
    return payload


def primary_image(item: dict[str, Any]) -> str:
    if item.get("image"):
        return str(item["image"])
    images = item.get("images") or []
    return str(images[0]) if images else ""


def source_from_path(path: str) -> str:
    if "/CCT/" in path:
        return "Caltech Camera Traps"
    if "/SS/" in path:
        return "Snapshot Serengeti"
    if "/WCS/" in path:
        return "WCS Camera Traps"
    if "/Idaho/" in path:
        return "Idaho Camera Traps"
    return "unknown"


def load_sensor_modes(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mode = row.get("sensor_mode_fine_v2") or "unknown"
            for key in [row.get("abs_image_path"), row.get("sample_image_path")]:
                if key:
                    out[str(key)] = mode
    return out


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix == ".jsonl":
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        else:
            for row in csv.DictReader(fh):
                rows.append(row)
    return rows


def normalize_mode(mode: str | None) -> str:
    mode = str(mode or "unknown")
    if mode in {"night_ir", "night_color", "night_lowlight", "night_white_flash"}:
        return mode
    if mode.startswith("night_"):
        return mode
    return "unknown"


def summarize(rows: list[dict[str, Any]], group_key: str, model: str) -> list[dict[str, Any]]:
    out = []
    labels_all = sorted({str(r["gold"]) for r in rows} | {str(r["pred"]) for r in rows})
    for group in sorted({str(r[group_key]) for r in rows}):
        part = [r for r in rows if str(r[group_key]) == group]
        y = [str(r["gold"]) for r in part]
        p = [str(r["pred"]) for r in part]
        out.append(
            {
                "task": "Species",
                "model": model,
                group_key: group,
                "N": len(part),
                "Accuracy": float(accuracy_score(y, p)) if part else None,
                "Macro-F1": float(f1_score(y, p, labels=labels_all, average="macro", zero_division=0)) if part else None,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(x: Any) -> str:
    if x is None:
        return "--"
    return f"{float(x) * 100:.2f}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    keep = [r for r in rows if r["imaging_mode"] in {"night_ir", "night_color", "night_lowlight", "night_white_flash", "unknown"}]
    lines = [
        "\\begin{table*}[h]",
        "\\centering",
        "\\caption{Species baselines by inferred night imaging mode. Mode labels are joined from the sensor-context asset inventory by primary image path; missing joins remain unknown.}",
        "\\label{tab:species_imaging_breakdown_v1}",
        "\\small",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Model & Imaging mode & N & Accuracy & Macro-F1 \\\\",
        "\\midrule",
    ]
    for r in keep:
        mode = str(r["imaging_mode"]).replace("_", "\\_")
        lines.append(f"{r['model']} & {mode} & {r['N']:,} & {pct(r['Accuracy'])} & {pct(r['Macro-F1'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="<ANIMALLAMA_ROOT>")
    parser.add_argument("--species-test", default="results/nighttrap_ops_v1_build/track_b_species/test.json")
    parser.add_argument("--sensor-labels", default="results/camtrap_asset_inventory/v1/derived/sensor_context_v1/event_sensor_labels.csv")
    parser.add_argument("--out-dir", default="results/nighttrap_supervised_baselines/species_imaging_breakdown_v1")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = root / args.out_dir
    test_items = load_json(root / args.species_test)
    by_id = {str(item.get("id")): primary_image(item) for item in test_items}
    mode_by_path = load_sensor_modes(root / args.sensor_labels)
    models = {
        "CLIP full-frame linear": root / "results/nighttrap_supervised_baselines/fullframe_clip_linear_species_v1/predictions.csv",
        "CLIP crop kNN": root / "results/nighttrap_supervised_baselines/detector_crop_retrieval_v1/species_predictions.csv",
        "CLIP crop linear": root / "results/nighttrap_supervised_baselines/detector_crop_clip_linear_v1/species_predictions.csv",
        "DINOv2 crop kNN": root / "results/nighttrap_supervised_baselines/detector_crop_dinov2_knn_v1/species_predictions.csv",
        "ViT-B/16 fine-tune": root / "results/nighttrap_supervised_baselines/vit_b16_finetune_species_v1/predictions.csv",
    }
    imaging_rows = []
    source_rows = []
    coverage = {}
    for model, pred_path in models.items():
        preds = load_predictions(pred_path)
        enriched = []
        joined = 0
        for row in preds:
            sample_id = str(row.get("id"))
            image = by_id.get(sample_id, "")
            mode = mode_by_path.get(image) or mode_by_path.get(image.replace("source_data/", ""))
            if mode:
                joined += 1
            enriched.append(
                {
                    "id": sample_id,
                    "gold": row.get("gold"),
                    "pred": row.get("pred"),
                    "imaging_mode": normalize_mode(mode),
                    "source": source_from_path(image),
                }
            )
        coverage[model] = {"N": len(enriched), "mode_joined": joined, "mode_join_rate": joined / len(enriched) if enriched else 0.0}
        imaging_rows.extend(summarize(enriched, "imaging_mode", model))
        source_rows.extend(summarize(enriched, "source", model))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "imaging_mode_summary.json").write_text(json.dumps(imaging_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "source_summary.json").write_text(json.dumps(source_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "mode_join_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "imaging_mode_summary.csv", imaging_rows)
    write_csv(out_dir / "source_summary.csv", source_rows)
    write_table(root / "results/nighttrap_tables/table_species_imaging_breakdown_v1.tex", imaging_rows)
    print(json.dumps({"out_dir": str(out_dir), "coverage": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
