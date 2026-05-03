#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path("<ANIMALLAMA_ROOT>")
OUT = Path("/tmp/nighttrap_qa_examples_v09")


def load(rel: str) -> list[dict]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def select(rows: list[dict], pred):
    for row in rows:
        if pred(row):
            return row
    raise RuntimeError("no matching row")


def answer_text(row: dict) -> str | None:
    idx = row.get("answer")
    choices = row.get("choices") or []
    if isinstance(idx, int) and 0 <= idx < len(choices):
        return choices[idx].strip()
    return None


def image_exists(row: dict) -> bool:
    path = row.get("image")
    return bool(path and Path(path).exists())


def safe_image_name(src: str, prefix: str) -> str:
    ext = Path(src).suffix.lower() or ".jpg"
    return f"{prefix}{ext}"


def copy_image(src: str, dst: Path) -> None:
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src_path, dst)


def write_record(slug: str, title: str, row: dict) -> dict:
    out_dir = OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "slug": slug,
        "title": title,
        "id": row.get("id"),
        "event_key": row.get("event_key") or (row.get("meta") or {}).get("event_id"),
        "site_key": row.get("site_key") or (row.get("meta") or {}).get("site_key"),
        "question_type": row.get("question_type"),
        "variant": row.get("variant"),
        "question": row.get("question"),
        "choices": row.get("choices"),
        "answer_index": row.get("answer"),
        "answer_text": answer_text(row),
        "meta": row.get("meta") or {},
        "source_images": row.get("images") or [row.get("image")],
        "copied_images": [],
    }
    if slug == "task5_needs_review":
        for idx, src in enumerate((row.get("images") or [])[:3], 1):
            name = safe_image_name(src, f"frame_{idx}")
            copy_image(src, out_dir / name)
            record["copied_images"].append(str(Path(slug) / name))
    else:
        src = row.get("image") or (row.get("images") or [None])[0]
        name = safe_image_name(src, "image")
        copy_image(src, out_dir / name)
        record["copied_images"].append(str(Path(slug) / name))
    (out_dir / "qa.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    task1 = select(
        load("results/nighttrap_ops_v1_track_d_build/track_d_v15_taskwise_hardsets/evidence_hard/qwen3vl8b_eval_questions/track_d_evidence_full_input.json"),
        lambda r: r.get("answer") == 0 and image_exists(r),
    )
    task2 = select(
        load("results/nighttrap_ops_v1_build/track_a_empty/test.json"),
        lambda r: r.get("answer") == 0 and image_exists(r),
    )
    task3 = select(
        load("results/nighttrap_ops_v1_build/track_b_species/test.json"),
        image_exists,
    )
    task4 = select(
        load("results/nighttrap_ops_v1_build/track_c_count/test.json"),
        lambda r: r.get("answer") in {2, 3} and image_exists(r),
    )
    task5 = None
    task5_rows = load("results/nighttrap_ops_v1_track_d_build/track_d_expanded_v4/routine_stress_6000/context_lite_questions_strict_v4/track_d_priority_full_context_lite.json")
    for desired_answer in [2, 1, 0]:
        for row in task5_rows:
            images = (row.get("images") or [])[:3]
            if (
                row.get("answer") == desired_answer
                and len(images) == 3
                and len(set(images)) == 3
                and all(Path(p).exists() for p in images)
            ):
                task5 = row
                break
        if task5 is not None:
            break
    if task5 is None:
        raise RuntimeError("no task5 row with three unique images")

    records = [
        write_record("task1_image_usability", "Task 1: Image usability filtering", task1),
        write_record("task2_empty_event", "Task 2: Empty-event filtering", task2),
        write_record("task3_species", "Task 3: Species classification", task3),
        write_record("task4_count_bin", "Task 4: Count-bin classification", task4),
        write_record("task5_needs_review", "Task 5: Needs-review recommendation", task5),
    ]
    (OUT / "qa_examples_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# NightTrap QA Examples v0.9", ""]
    for rec in records:
        lines.extend(
            [
                f"## {rec['title']}",
                f"- ID: `{rec['id']}`",
                f"- Event: `{rec['event_key']}`",
                f"- Site: `{rec['site_key']}`",
                f"- Answer: `{rec['answer_text']}`",
                "- Copied image(s): " + ", ".join(f"`{p}`" for p in rec["copied_images"]),
                "",
                "**Question**",
                "",
                rec["question"] or "",
                "",
                "**Choices**",
            ]
        )
        for choice in rec.get("choices") or []:
            lines.append(f"- {choice.strip()}")
        lines.append("")
    (OUT / "qa_examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "selected": [(r["slug"], r["id"], r["answer_text"], r["copied_images"]) for r in records]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
