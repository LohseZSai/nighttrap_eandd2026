#!/usr/bin/env python3
"""Prepare human-annotation batches for NightTrap audit tasks.

This script does not copy or redistribute raw images. It keeps the original
image paths so that the generated CSV/JSON can be uploaded to the annotation
server that has access to the source datasets.
"""

from __future__ import annotations

import csv
import html
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MEGA_IN = ROOT / "results/nighttrap_megadetector_independent_audit_v1/annotation_sheet.csv"
LABEL_IN = ROOT / "results/nighttrap_label_audit_v1/audit_manifest.csv"
OUT = ROOT / "results/nighttrap_manual_annotation_tasks_v1"


MANUAL_LABELS = "empty | nonempty | uncertain"
ERROR_TYPES = (
    "no_error | animal_visible | no_animal_visible | uncertain_low_quality | "
    "partial_or_occluded | overexposed_or_near_black | image_missing | other"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def stratified_order(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("gold_label", ""), row.get("confidence_bucket", ""), row.get("source", ""))
        buckets[key].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda r: (r.get("event_id", ""), r.get("image_path", "")))

    ordered: list[dict[str, str]] = []
    keys = sorted(buckets, key=lambda k: (k[0], k[1], k[2]))
    while keys:
        next_keys = []
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def to_label_studio_json(rows: list[dict[str, str]], out_path: Path) -> None:
    tasks = []
    for row in rows:
        tasks.append(
            {
                "data": {
                    "image": row["image_path"],
                    "event_id": row["event_id"],
                    "source": row.get("source", ""),
                    "mode": row.get("mode", ""),
                    "detector_score": row.get("detector_score", ""),
                    "pred_label": row.get("pred_label", ""),
                    "confidence_bucket": row.get("confidence_bucket", ""),
                    "gold_label_for_audit": row.get("gold_label", ""),
                }
            }
        )
    out_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")


def html_page(rows: list[dict[str, str]], title: str) -> str:
    cards = []
    for i, row in enumerate(rows, 1):
        image_path = html.escape(row["image_path"])
        event_id = html.escape(row["event_id"])
        meta = " | ".join(
            html.escape(str(row.get(k, "")))
            for k in ["source", "mode", "pred_label", "detector_score", "confidence_bucket"]
        )
        cards.append(
            f"""
            <section class="card">
              <div class="idx">#{i}</div>
              <div class="meta"><b>event_id:</b> {event_id}<br>{meta}</div>
              <img src="file://{image_path}" alt="event {event_id}">
              <div class="path">{image_path}</div>
              <div class="fields">
                manual_label: <b>{MANUAL_LABELS}</b><br>
                error_type: <b>{ERROR_TYPES}</b><br>
                notes: optional short reason
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
h1 {{ margin-bottom: 4px; }}
.note {{ background: #f7f7f7; border: 1px solid #d7dce3; padding: 12px; margin: 12px 0 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(360px, 1fr)); gap: 14px; }}
.card {{ border: 1px solid #cfd6e3; border-radius: 8px; padding: 10px; break-inside: avoid; }}
.idx {{ font-size: 18px; font-weight: 700; color: #0b2a6f; }}
.meta {{ font-size: 13px; line-height: 1.35; margin: 4px 0 8px; }}
img {{ width: 100%; max-height: 420px; object-fit: contain; background: #111; border-radius: 4px; }}
.path {{ font-size: 11px; color: #4b5563; word-break: break-all; margin-top: 6px; }}
.fields {{ font-size: 13px; background: #fffdf2; border: 1px solid #eee1a0; padding: 8px; margin-top: 8px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="note">
Label <b>manual_label</b> from the image only. Use <b>uncertain</b> when the image is too dark,
blurred, occluded, or otherwise not safely judgeable. Fill the returned CSV; this HTML is only a
visual checklist. Images load only on a machine/server where the listed paths are mounted.
</div>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""


def prepare_megadetector_batches() -> None:
    rows = stratified_order(read_csv(MEGA_IN))
    fieldnames = [
        "batch_id",
        "task_index",
        "event_id",
        "image_path",
        "source",
        "mode",
        "gold_label",
        "detector_score",
        "pred_label",
        "confidence_bucket",
        "manual_label",
        "error_type",
        "notes",
    ]
    mega_dir = OUT / "megadetector_empty_audit"
    mega_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 100
    manifest_rows = []
    for b in range(math.ceil(len(rows) / batch_size)):
        batch_id = f"megadetector_batch_{b+1:03d}"
        batch = rows[b * batch_size : (b + 1) * batch_size]
        out_rows = []
        for i, row in enumerate(batch, 1):
            new = dict(row)
            new["batch_id"] = batch_id
            new["task_index"] = str(i)
            out_rows.append(new)
        batch_dir = mega_dir / batch_id
        write_csv(batch_dir / f"{batch_id}.csv", out_rows, fieldnames)
        to_label_studio_json(out_rows, batch_dir / f"{batch_id}_label_studio.json")
        (batch_dir / f"{batch_id}.html").write_text(
            html_page(out_rows, f"NightTrap MegaDetector Audit {batch_id}"),
            encoding="utf-8",
        )
        with zipfile.ZipFile(mega_dir / f"{batch_id}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in batch_dir.iterdir():
                z.write(p, p.relative_to(mega_dir))

        counts = Counter(r["gold_label"] for r in out_rows)
        manifest_rows.append(
            {
                "batch_id": batch_id,
                "n": str(len(out_rows)),
                "empty_false_trigger": str(counts.get("empty_false_trigger", 0)),
                "nonempty_or_uncertain": str(counts.get("nonempty_or_uncertain", 0)),
                "csv": str((batch_dir / f"{batch_id}.csv").relative_to(ROOT)),
                "label_studio_json": str((batch_dir / f"{batch_id}_label_studio.json").relative_to(ROOT)),
                "html": str((batch_dir / f"{batch_id}.html").relative_to(ROOT)),
                "zip": str((mega_dir / f"{batch_id}.zip").relative_to(ROOT)),
            }
        )
    write_csv(
        mega_dir / "batch_manifest.csv",
        manifest_rows,
        ["batch_id", "n", "empty_false_trigger", "nonempty_or_uncertain", "csv", "label_studio_json", "html", "zip"],
    )


def prepare_needs_review_batch() -> None:
    rows = read_csv(LABEL_IN)
    # Keep a small, balanced rulebook audit starter: 25 routine, 25 review-like positive,
    # plus whatever boundary rows are already present in the manifest.
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("gold_label", "")].append(row)
    selected = []
    for label, n in [("routine", 25), ("review", 25), ("boundary", 20)]:
        selected.extend(grouped.get(label, [])[:n])
    if not selected:
        selected = rows[:60]

    out_dir = OUT / "needs_review_rulebook_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(out_dir / "needs_review_rulebook_batch_001.csv", selected, fieldnames)

    readme = """# Needs-review rulebook audit batch 001

Purpose: check whether the binary `routine` vs `review` assignment is plausible under the written
NightTrap rulebook. This is not an ecological anomaly audit.

Fill these fields:
- `manual_label`: routine | review | uncertain
- `manual_binary_label`: 0 | 1 | uncertain
- `image_support`: yes | no | uncertain
- `context_support`: yes | no | uncertain
- `error_type`: no_error | label_too_strong | label_too_weak | ambiguous_evidence | context_insufficient | image_unusable | other
- `notes`: short reason

Do not report IAA or plausibility precision until completed human labels are summarized.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def write_instructions() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = f"""# NightTrap Manual Annotation Push v1

Generated local task package for human audit. Raw images are not copied or redistributed; all task
files keep source image paths.

## P0 task to start now

Start with:

`results/nighttrap_manual_annotation_tasks_v1/megadetector_empty_audit/megadetector_batch_001/megadetector_batch_001.csv`

or upload:

`results/nighttrap_manual_annotation_tasks_v1/megadetector_empty_audit/megadetector_batch_001.zip`

Batch 1 contains 100 stratified MegaDetector audit items. Fill:

- `manual_label`: {MANUAL_LABELS}
- `error_type`: {ERROR_TYPES}
- `notes`: optional short reason

Recommended decision rule:

- `empty`: no animal or valid animal subject is visible.
- `nonempty`: an animal or likely animal subject is visible.
- `uncertain`: too dark, blurred, occluded, cropped, or otherwise not safely judgeable.

After Batch 1 is returned, compute precision/recall only on rows with `manual_label` in
`empty` or `nonempty`; keep `uncertain` as an audit bucket and report it separately.

## Batches

See `megadetector_empty_audit/batch_manifest.csv`.

## Needs-review rulebook audit

A starter binary rulebook audit batch is in:

`results/nighttrap_manual_annotation_tasks_v1/needs_review_rulebook_audit/needs_review_rulebook_batch_001.csv`

This is secondary to the MegaDetector audit unless reviewer response time allows both.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    prepare_megadetector_batches()
    prepare_needs_review_batch()
    write_instructions()


if __name__ == "__main__":
    main()
