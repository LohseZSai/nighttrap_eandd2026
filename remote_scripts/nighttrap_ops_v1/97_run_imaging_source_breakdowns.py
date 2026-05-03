#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


ROOT = Path("<ANIMALLAMA_ROOT>")
CATALOG = ROOT / "results/nighttrap_ops_v1_build/00_catalog/night_event_catalog.jsonl"
ROUTINE_STRESS = ROOT / "results/nighttrap_ops_v1_track_d_build/track_d_expanded_v4/routine_stress_6000/routine_stress_manifest.jsonl"
OUT = ROOT / "results/nighttrap_diagnostics/robustness_v08"
TABLE_DIR = ROOT / "results/nighttrap_tables"
AUDIT_MODE_FILES = [
    ROOT / "results/route_v08_detector_pools/empty_candidate_review_200/audits/route_review_sample.csv",
    ROOT / "results/empty_disagreement_review_bundle/audits/route_review_sample.csv",
    ROOT / "results/empty_false_trigger_review_bundle/audits/route_review_sample.csv",
    ROOT / "results/empty_rebuild_review_bundle/audits/route_review_sample.csv",
]


TASKS = {
    "Image usability": {
        "qwen": ROOT / "results/nighttrap_local_model_eval_v1/qwen3vl8b_full_freeze_v1/track_d_hard/evidence_hard_full_input/report.json",
        "clip": ROOT / "results/nighttrap_supervised_baselines/clip_linear_probe_v1/usability/predictions.jsonl",
    },
    "Empty-event": {
        "qwen": ROOT / "results/nighttrap_local_model_eval_v1/qwen3vl8b_full_freeze_v1/abc/track_a_empty/report.json",
        "clip": ROOT / "results/nighttrap_supervised_baselines/clip_linear_probe_v1/empty/predictions.jsonl",
        "megadetector": ROOT / "results/nighttrap_supervised_baselines/megadetector_empty_threshold_v1/test_predictions.jsonl",
    },
    "Species": {
        "qwen": ROOT / "results/nighttrap_local_model_eval_v1/qwen3vl8b_full_freeze_v1/abc/track_b_species/report.json",
        "clip": ROOT / "results/nighttrap_supervised_baselines/clip_linear_probe_v1/species/predictions.jsonl",
    },
    "Count-bin": {
        "qwen": ROOT / "results/nighttrap_local_model_eval_v1/qwen3vl8b_full_freeze_v1/abc/track_c_count/report.json",
        "clip": ROOT / "results/nighttrap_supervised_baselines/clip_linear_probe_v1/count/predictions.jsonl",
    },
    "Needs-review": {
        "qwen": ROOT / "results/nighttrap_ops_v1_track_d_build/track_d_expanded_v4/routine_stress_6000/qwen3vl8b_full_context_lite/report.json",
        "clip/context": ROOT / "results/nighttrap_supervised_baselines/needs_review_clip_context_ranker_v1/test_predictions.jsonl",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event_id_from(value: object) -> str | None:
    if value is None:
        return None
    m = re.search(r"(\d+)$", str(value))
    return m.group(1) if m else None


def load_catalog() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for row in read_jsonl(CATALOG):
        eid = str(row["event_key"])
        meta[eid] = {
            "event_id": eid,
            "dataset_name": row.get("dataset_name") or "unknown",
            "dataset_key": row.get("dataset_key") or "unknown",
            "site_key": row.get("site_key") or "unknown",
            "sample_image_path": row.get("sample_image_path"),
            "images": row.get("images") or [],
            "count_available": row.get("count_available"),
        }
    for row in read_jsonl(ROUTINE_STRESS):
        eid = str(row["event_id"])
        meta.setdefault(eid, {}).update(
            {
                "event_id": eid,
                "dataset_name": row.get("dataset_name") or meta.get(eid, {}).get("dataset_name") or "unknown",
                "dataset_key": row.get("dataset_key") or meta.get(eid, {}).get("dataset_key") or "unknown",
                "site_key": row.get("site_key") or meta.get(eid, {}).get("site_key") or "unknown",
                "sample_image_path": next(iter((row.get("images") or {}).values()), None),
            }
        )
    return meta


def load_audit_modes() -> dict[str, str]:
    import csv

    modes = {}
    for path in AUDIT_MODE_FILES:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                eid = str(row.get("event_key") or "")
                mode = row.get("sensor_mode_fine_v2") or ""
                if eid and mode:
                    modes[eid] = mode
    return modes


def imaging_mode(meta: dict, audit_modes: dict[str, str]) -> str:
    eid = str(meta.get("event_id") or "")
    if eid in audit_modes:
        return audit_modes[eid]
    dataset = meta.get("dataset_name") or ""
    if dataset in {"Caltech Camera Traps", "Idaho Camera Traps"}:
        return "night_ir"
    if dataset in {"WCS Camera Traps", "Snapshot Serengeti"}:
        return "night_color"
    return "unknown"


def normalize_empty(x: str) -> str:
    s = str(x)
    if "empty_false_trigger" in s or s.startswith("(A)"):
        return "empty_false_trigger"
    if "nonempty_or_uncertain" in s or s.startswith("(B)"):
        return "nonempty_or_uncertain"
    return s


def normalize_usability(x: str) -> str:
    s = str(x)
    if "insufficient" in s or s.startswith("(A)"):
        return "insufficient_for_review_decision"
    if "sufficient" in s or s.startswith("(B)"):
        return "sufficient_for_review_decision"
    return s


def normalize_count(x: str) -> str:
    s = str(x)
    if "3-5" in s or s.startswith("(C)"):
        return "3-5"
    if "6+" in s or s.startswith("(D)"):
        return "6+"
    if "2" in s or s.startswith("(B)"):
        return "2"
    if "1" in s or s.startswith("(A)"):
        return "1"
    return s


def normalize_needs_review(x: str) -> int:
    s = str(x).lower()
    if s in {"0", "routine"} or s.startswith("(a)"):
        return 0
    return 1


def qwen_rows(task: str, path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in data.get("results", []):
        eid = event_id_from(r.get("id"))
        if not eid:
            continue
        if task == "Image usability":
            gold, pred, score = normalize_usability(r.get("gt")), normalize_usability(r.get("pred")), None
        elif task == "Empty-event":
            gold, pred, score = normalize_empty(r.get("gt")), normalize_empty(r.get("pred")), None
        elif task == "Count-bin":
            gold, pred, score = normalize_count(r.get("gt")), normalize_count(r.get("pred")), None
        elif task == "Needs-review":
            gold, pred = normalize_needs_review(r.get("gt_taxon") or r.get("gt")), normalize_needs_review(r.get("pred_taxon") or r.get("pred"))
            probs = r.get("choice_probs") or {}
            score = float(probs.get("(B)", 0.0)) + float(probs.get("(C)", 0.0))
        else:
            gold, pred, score = str(r.get("gt_taxon") or r.get("gt")), str(r.get("pred_taxon") or r.get("pred")), None
        rows.append({"event_id": eid, "gold": gold, "pred": pred, "score": score})
    return rows


def jsonl_rows(task: str, path: Path) -> list[dict]:
    rows = []
    for r in read_jsonl(path):
        eid = event_id_from(r.get("event_id") or r.get("event_key") or r.get("id") or r.get("sample_id"))
        if not eid:
            continue
        if task == "Image usability":
            gold, pred, score = normalize_usability(r.get("gold")), normalize_usability(r.get("pred")), None
        elif task == "Empty-event":
            gold, pred, score = normalize_empty(r.get("gold")), normalize_empty(r.get("pred")), r.get("detector_target_confidence")
        elif task == "Count-bin":
            gold, pred, score = normalize_count(r.get("gold")), normalize_count(r.get("pred")), None
        elif task == "Needs-review":
            gold = int(r.get("gold_needs_review"))
            score = float(r.get("score_needs_review"))
            pred = int(r.get("pred_at_0.5", score >= 0.5))
        else:
            gold, pred, score = str(r.get("gold")), str(r.get("pred")), None
        rows.append({"event_id": eid, "gold": gold, "pred": pred, "score": score})
    return rows


def p_at_k(y: list[int], score: list[float], k: int = 50) -> float | None:
    if not y or any(s is None for s in score):
        return None
    kk = min(k, len(y))
    order = np.argsort(-np.asarray(score))[:kk]
    return float(np.asarray(y)[order].sum() / kk)


def summarize(rows: list[dict]) -> dict:
    y = [r["gold"] for r in rows]
    p = [r["pred"] for r in rows]
    out = {"N": len(rows)}
    if not rows:
        return {**out, "Accuracy": None, "Macro-F1": None, "AUPRC": None, "P@50": None}
    out["Accuracy"] = float(accuracy_score(y, p))
    out["Macro-F1"] = float(f1_score(y, p, average="macro", zero_division=0))
    if set(y) <= {0, 1}:
        scores = [r.get("score") for r in rows]
        out["P@50"] = p_at_k([int(v) for v in y], scores, 50)
        out["AUPRC"] = float(average_precision_score([int(v) for v in y], scores)) if all(s is not None for s in scores) and len(set(y)) == 2 else None
    else:
        out["P@50"] = None
        out["AUPRC"] = None
    return out


def fmt(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def attach(rows: list[dict], meta: dict[str, dict], audit_modes: dict[str, str]) -> list[dict]:
    out = []
    for r in rows:
        m = meta.get(str(r["event_id"]), {})
        rr = dict(r)
        rr["source"] = m.get("dataset_name", "unknown")
        rr["imaging_mode"] = imaging_mode(m, audit_modes)
        out.append(rr)
    return out


def group_summary(records: list[dict], group_key: str) -> list[dict]:
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["task"], r["model"], r[group_key])].append(r)
    out = []
    for (task, model, group), rows in sorted(grouped.items()):
        out.append({"task": task, "model": model, group_key: group, **summarize(rows)})
    return out


def table_lines(rows: list[dict], key: str, limit_tasks: set[str]) -> str:
    keep = [r for r in rows if r["task"] in limit_tasks and r["N"] >= 20]
    lines = []
    for r in keep:
        metric = fmt(r["AUPRC"] if r["task"] == "Needs-review" else r["Macro-F1"])
        name = r[key].replace("_", "\\_")
        lines.append(f"{name} & {r['task']} & {r['model']} & {r['N']} & {fmt(r['Accuracy'])} & {fmt(r['Macro-F1'])} & {fmt(r['P@50'])} & {metric} \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_catalog()
    audit_modes = load_audit_modes()
    records = []
    for task, models in TASKS.items():
        for model, path in models.items():
            if not path.exists():
                continue
            rows = qwen_rows(task, path) if path.suffix == ".json" and model == "qwen" else jsonl_rows(task, path)
            model_name = {"qwen": "Qwen3-VL-8B", "clip": "CLIP linear probe", "clip/context": "CLIP/context ranker", "megadetector": "MegaDetector threshold"}[model]
            for row in attach(rows, meta, audit_modes):
                row["task"] = task
                row["model"] = model_name
                records.append(row)
    imaging = group_summary(records, "imaging_mode")
    source = group_summary(records, "source")
    count_sources = Counter(r["source"] for r in records if r["task"] == "Count-bin")
    payload = {
        "source_files": {task: {m: str(p.relative_to(ROOT)) for m, p in models.items() if p.exists()} for task, models in TASKS.items()},
        "imaging_mode_summary": imaging,
        "source_summary": source,
        "count_bin_source_counts": dict(count_sources),
        "imaging_mode_method": "Uses existing sensor_mode_fine_v2 labels where available; otherwise maps Caltech/Idaho to night_ir and WCS/Snapshot Serengeti to night_color. This is a diagnostic grouping, not a new visual classifier.",
    }
    (OUT / "imaging_mode_summary.json").write_text(json.dumps(payload["imaging_mode_summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "source_summary.json").write_text(json.dumps(payload["source_summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "robustness_v08_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (TABLE_DIR / "table_imaging_mode_v08.tex").write_text(table_lines(imaging, "imaging_mode", {"Empty-event", "Needs-review"}), encoding="utf-8")
    (TABLE_DIR / "table_source_breakdown_v08.tex").write_text(table_lines(source, "source", {"Count-bin", "Needs-review"}), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "tables": ["table_imaging_mode_v08.tex", "table_source_breakdown_v08.tex"]}, indent=2))


if __name__ == "__main__":
    main()
