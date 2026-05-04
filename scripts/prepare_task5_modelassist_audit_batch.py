#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gold_bin(row: dict[str, Any]) -> str:
    return "review" if row.get("review_priority") in {"review", "priority_review"} else "routine"


def pred_bin(row: dict[str, Any]) -> str:
    if row.get("pred_taxon") in {"review", "priority_review"}:
        return "review"
    if row.get("pred") in {"(B)", "(C)"}:
        return "review"
    return "routine"


def needs_review_score(row: dict[str, Any]) -> float:
    probs = row.get("choice_probs") or {}
    return float(probs.get("(B)", 0.0) or 0.0) + float(probs.get("(C)", 0.0) or 0.0)


def spread(items: list[str], n: int, rank_by_event: dict[str, int]) -> list[str]:
    items = sorted(items, key=lambda event_id: (rank_by_event.get(event_id, 999999), event_id))
    if len(items) <= n:
        return items
    idxs = []
    for i in range(n):
        idx = round(i * (len(items) - 1) / (n - 1)) if n > 1 else 0
        if idx not in idxs:
            idxs.append(idx)
    out = [items[idx] for idx in idxs]
    for event_id in items:
        if len(out) >= n:
            break
        if event_id not in out:
            out.append(event_id)
    return out[:n]


def species_in_common(row: dict[str, Any]) -> str:
    species = str(row.get("species_label_audit_only") or "").strip().lower()
    names = [
        str(item).strip().lower()
        for item in ((row.get("reference") or {}).get("site_common_species_names") or [])
    ]
    if not species or not names:
        return "unknown"
    return "yes" if species in names else "no"


def context_lite(row: dict[str, Any]) -> str:
    ref = row.get("reference") or {}
    return json.dumps(
        {
            "exposes_counts": False,
            "exposes_current_species": False,
            "exposes_novelty_or_rarity": False,
            "season": row.get("season", ""),
            "site_common_species_names": ref.get("site_common_species_names") or [],
            "top_k_source": "reference_site_summary",
            "used_sqlite_fallback": bool(ref.get("used_sqlite_fallback", False)),
        },
        ensure_ascii=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--batch-id", default="002")
    parser.add_argument("--n-per-cell", type=int, default=25)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    base = root / "results/nighttrap_needs_review_plausibility_audit_v1"
    manifest_path = (
        root
        / "results/nighttrap_ops_v1_track_d_build/track_d_expanded_v4/routine_stress_6000/routine_stress_manifest.jsonl"
    )
    report_path = (
        root
        / "results/nighttrap_ops_v1_track_d_build/track_d_expanded_v4/routine_stress_6000/qwen3vl8b_full_context_lite/report.json"
    )
    old_batch_path = base / "needs_review_plausibility_batch_001.csv"

    manifest_by_event = {str(row["event_id"]): row for row in read_jsonl(manifest_path)}
    report_rows = json.loads(report_path.read_text(encoding="utf-8"))["results"]
    model_by_event = {str(row["id"]).split("__")[-1]: row for row in report_rows}
    old_events = {
        row.get("event_id", "")
        for row in read_csv(old_batch_path)
        if old_batch_path.exists()
    }

    ranked = sorted(((needs_review_score(row), event_id) for event_id, row in model_by_event.items()), reverse=True)
    rank_by_event = {event_id: idx + 1 for idx, (_, event_id) in enumerate(ranked)}

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for event_id, manifest_row in manifest_by_event.items():
        if event_id in old_events or event_id not in model_by_event:
            continue
        buckets[(gold_bin(manifest_row), pred_bin(model_by_event[event_id]))].append(event_id)

    selected: list[str] = []
    for key in [("routine", "routine"), ("routine", "review"), ("review", "routine"), ("review", "review")]:
        selected.extend(spread(buckets[key], args.n_per_cell, rank_by_event))
    target_n = args.n_per_cell * 4
    if len(selected) < target_n:
        used = set(selected)
        remaining = []
        for items in buckets.values():
            remaining.extend(event_id for event_id in items if event_id not in used)
        remaining = sorted(
            remaining,
            key=lambda event_id: (
                gold_bin(manifest_by_event[event_id]),
                pred_bin(model_by_event[event_id]),
                rank_by_event.get(event_id, 999999),
                event_id,
            ),
        )
        selected.extend(remaining[: target_n - len(selected)])
    selected = selected[:target_n]

    rows: list[dict[str, Any]] = []
    for idx, event_id in enumerate(selected, 1):
        manifest_row = manifest_by_event[event_id]
        model_row = model_by_event[event_id]
        probs = model_row.get("choice_probs") or {}
        score = needs_review_score(model_row)
        model_pred = pred_bin(model_row)
        images = manifest_row.get("images") or {}
        rows.append(
            {
                "sample_id": f"routine_stress_modelassist_{args.batch_id}_{idx:03d}",
                "source_set": "routine_stress_6000_modelassist",
                "event_id": event_id,
                "dataset": manifest_row.get("dataset_key") or manifest_row.get("dataset_name", ""),
                "site_id": manifest_row.get("site_key", ""),
                "gold_label": gold_bin(manifest_row),
                "image_path_first": images.get("first") or model_row.get("image", ""),
                "image_path_middle": images.get("middle") or images.get("first") or model_row.get("image", ""),
                "image_path_last": images.get("last") or images.get("middle") or images.get("first") or model_row.get("image", ""),
                "species_label_audit_only": manifest_row.get("species_label_audit_only", "not available"),
                "count_label_audit_only": manifest_row.get("count_bin") or "not available",
                "species_in_site_common_species_audit_only": species_in_common(manifest_row),
                "dataset_species_source_audit_only": "source dataset label",
                "context_lite_json": context_lite(manifest_row),
                "decision_meaning": "Routine means deprioritize; needs_review means send to human review because identity, count, ambiguity, or context may need confirmation.",
                "binary_mapping": "routine=0; needs_review=1",
                "model_name": "Qwen3-VL-8B",
                "model_setting": "full context-lite, routine-stress 6000",
                "model_pred_label": "needs_review" if model_pred == "review" else "routine",
                "model_needs_review_score": f"{score:.6f}",
                "model_rank": str(rank_by_event.get(event_id, "")),
                "model_choice_probs": json.dumps(
                    {
                        "routine_A": probs.get("(A)", 0.0),
                        "review_B": probs.get("(B)", 0.0),
                        "priority_review_C": probs.get("(C)", 0.0),
                    },
                    ensure_ascii=False,
                ),
                "manual_label": "",
                "manual_binary_label": "",
                "image_support": "",
                "context_support": "",
                "error_type": "",
                "notes": "",
            }
        )

    fields = list(rows[0].keys())
    batch_path = base / f"needs_review_matched_context_modelassist_batch_{args.batch_id}.csv"
    completed_path = base / f"needs_review_matched_context_modelassist_batch_{args.batch_id}_completed.csv"
    answer_path = base / f"needs_review_matched_context_modelassist_batch_{args.batch_id}_answer_key.csv"
    write_csv(batch_path, rows, fields)
    write_csv(completed_path, rows, fields)

    answer_rows = []
    for row in rows:
        answer_rows.append(
            {
                "sample_id": row["sample_id"],
                "event_id": row["event_id"],
                "dataset": row["dataset"],
                "site_id": row["site_id"],
                "benchmark_label": "needs_review" if row["gold_label"] == "review" else "routine",
                "benchmark_binary_label": "1" if row["gold_label"] == "review" else "0",
                "model_pred_label": row["model_pred_label"],
                "model_needs_review_score": row["model_needs_review_score"],
                "model_rank": row["model_rank"],
                "species_label_audit_only": row["species_label_audit_only"],
                "species_in_site_common_species_audit_only": row["species_in_site_common_species_audit_only"],
            }
        )
    write_csv(answer_path, answer_rows, list(answer_rows[0].keys()))

    selected_confusion = Counter(
        f"gold_{row['gold_label']}__model_{'review' if row['model_pred_label'] == 'needs_review' else 'routine'}"
        for row in rows
    )
    summary = {
        "batch_csv": str(batch_path),
        "completed_csv": str(completed_path),
        "answer_key": str(answer_path),
        "n": len(rows),
        "bucket_available": {f"{key[0]}__model_{key[1]}": len(value) for key, value in buckets.items()},
        "selected_gold": dict(Counter(row["gold_label"] for row in rows)),
        "selected_model_pred": dict(Counter(row["model_pred_label"] for row in rows)),
        "selected_confusion": dict(selected_confusion),
    }
    summary_path = base / f"needs_review_matched_context_modelassist_batch_{args.batch_id}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
