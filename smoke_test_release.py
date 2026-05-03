#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


RAW_MEDIA_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> None:
    root = Path(os.environ.get("NIGHTTRAP_DATASET_ROOT", "../dataset_repo")).resolve()
    require(root / "README.md")
    require(root / "LICENSE")
    require(root / "data/catalog/night_event_catalog.jsonl")
    require(root / "data/catalog/night_event_catalog_summary.json")
    require(root / "data/tasks/empty_event_filtering/summary.json")
    require(root / "data/tasks/species_classification/summary.json")
    require(root / "data/tasks/count_bin_classification/summary.json")
    require(root / "data/tasks/needs_review_recommendation/strict_983/summary.json")

    raw_media = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in RAW_MEDIA_EXTENSIONS
    ]
    if raw_media:
        raise RuntimeError("Raw media-like files found: " + ", ".join(raw_media[:10]))

    catalog_rows = sum(1 for _ in (root / "data/catalog/night_event_catalog.jsonl").open())
    stats = json.loads((root / "data/frozen_results/dataset_figures/nighttrap_dataset_overview_stats.json").read_text())
    checks = {
        "night_events": stats.get("night_events") == 68187,
        "night_frames": stats.get("night_frames") == 115617,
        "camera_sites": stats.get("camera_sites") == 2902,
        "species": stats.get("species") == 283,
        "catalog_rows": catalog_rows == 68187,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Dataset consistency checks failed: {failed}")
    print("NightTrap release smoke test passed.")


if __name__ == "__main__":
    main()
