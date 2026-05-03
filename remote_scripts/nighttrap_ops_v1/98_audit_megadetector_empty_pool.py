#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("<ANIMALLAMA_ROOT>")
PRED = ROOT / "results/nighttrap_supervised_baselines/megadetector_empty_threshold_v1/test_predictions.jsonl"
CATALOG = ROOT / "results/nighttrap_ops_v1_build/00_catalog/night_event_catalog.jsonl"
OUT = ROOT / "results/nighttrap_diagnostics/megadetector_audit_v08"
TABLE_DIR = ROOT / "results/nighttrap_tables"
AUDIT_CANDIDATES = [
    ROOT / "results/route_v08_detector_pools/empty_candidate_review_200/audits/route_review_sample.csv",
    ROOT / "results/empty_disagreement_review_bundle/audits/route_review_sample.csv",
    ROOT / "results/empty_false_trigger_review_bundle/audits/route_review_sample.csv",
    ROOT / "results/empty_rebuild_review_bundle/audits/route_review_sample.csv",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eid_from_id(s: str) -> str:
    return str(s).split("__")[-1]


def load_meta() -> dict[str, dict]:
    out = {}
    for r in read_jsonl(CATALOG):
        out[str(r["event_key"])] = r
    return out


def confidence_bin(x: float) -> str:
    if x < 0.01:
        return "[0,0.01)"
    if x < 0.05:
        return "[0.01,0.05)"
    if x < 0.25:
        return "[0.05,0.25)"
    if x < 0.50:
        return "[0.25,0.50)"
    if x < 0.75:
        return "[0.50,0.75)"
    return "[0.75,1.00]"


def mode_from_existing_audits() -> dict[str, str]:
    modes = {}
    for path in AUDIT_CANDIDATES:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                eid = str(r.get("event_key") or "")
                mode = r.get("sensor_mode_fine_v2") or ""
                if eid and mode:
                    modes[eid] = mode
    return modes


def row_bucket(row: dict) -> str:
    gold, pred, conf = row["gold"], row["pred"], row["detector_confidence"]
    if gold == "empty_false_trigger" and pred == "empty_false_trigger":
        return "empty_false_trigger"
    if gold == "nonempty_or_uncertain" and pred == "empty_false_trigger":
        return "low_confidence_nonempty"
    if gold == "empty_false_trigger" and pred == "nonempty_or_uncertain":
        return "high_confidence_empty"
    if conf < 0.05:
        return "low_confidence_nonempty"
    return "nonempty_or_uncertain"


def stratum(row: dict) -> str:
    parts = [row["confidence_bin"], row["audit_bucket"], row["imaging_mode"]]
    return "|".join(parts)


def sample_manifest(rows: list[dict], target: int = 300) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[stratum(r)].append(r)
    selected = []
    per = max(1, target // max(1, len(groups)))
    for _, items in sorted(groups.items()):
        items = sorted(items, key=lambda r: (r["event_id"], r["detector_confidence"]))
        selected.extend(items[:per])
    if len(selected) < target:
        used = {r["event_id"] for r in selected}
        rest = [r for r in sorted(rows, key=lambda r: (r["confidence_bin"], r["event_id"])) if r["event_id"] not in used]
        selected.extend(rest[: target - len(selected)])
    return selected[:target]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    audit_modes = mode_from_existing_audits()
    rows = []
    for r in read_jsonl(PRED):
        eid = eid_from_id(r["id"])
        m = meta.get(eid, {})
        conf = float(r.get("detector_target_confidence") or 0.0)
        gold = str(r.get("gold"))
        pred = str(r.get("pred"))
        row = {
            "event_id": eid,
            "dataset_name": m.get("dataset_name", "unknown"),
            "site_key": m.get("site_key", "unknown"),
            "sample_image_path": m.get("sample_image_path", ""),
            "gold_label": gold,
            "detector_prediction": pred,
            "detector_confidence": conf,
            "confidence_bin": confidence_bin(conf),
            "imaging_mode": audit_modes.get(eid, "unknown"),
            "audit_bucket": "",
            "manual_review_label": "",
            "manual_review_notes": "",
        }
        row["audit_bucket"] = row_bucket({"gold": gold, "pred": pred, "detector_confidence": conf})
        rows.append(row)
    manifest = sample_manifest(rows, 300)
    fields = [
        "event_id",
        "dataset_name",
        "site_key",
        "sample_image_path",
        "gold_label",
        "detector_prediction",
        "detector_confidence",
        "confidence_bin",
        "imaging_mode",
        "audit_bucket",
        "manual_review_label",
        "manual_review_notes",
    ]
    with (OUT / "audit_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)
    summary = {
        "prediction_file": str(PRED.relative_to(ROOT)),
        "human_review_results_available": False,
        "manifest_n": len(manifest),
        "candidate_pool_n": len(rows),
        "confidence_bins": dict(Counter(r["confidence_bin"] for r in manifest)),
        "audit_buckets": dict(Counter(r["audit_bucket"] for r in manifest)),
        "imaging_modes_from_existing_audit_files": dict(Counter(r["imaging_mode"] for r in manifest)),
        "note": "No manual labels are summarized here; this file is a stratified manifest for subsequent human review.",
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = []
    for bucket, n in sorted(summary["audit_buckets"].items()):
        bucket_tex = bucket.replace("_", "\\_")
        lines.append(f"{bucket_tex} & {n} & manifest only & -- & -- \\\\")
    (TABLE_DIR / "table_megadetector_audit_v08.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "manifest_n": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
