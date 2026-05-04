#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPECIES_TEST = ROOT / "release/_remote_cache/AnimaLLaMA/results/nighttrap_ops_v1_build/track_b_species/test.json"
OUT_DIR = ROOT / "results/nighttrap_species_candidate_alignment_v1"
TABLE_PATH = ROOT / "results/nighttrap_tables/table_species_candidate_alignment_v1.tex"


SUPERVISED_ROWS = [
    (
        "DINOv2 full-frame linear probe",
        ROOT / "results/nighttrap_supervised_baselines/fullframe_dinov2_linear_species_v1/species_predictions.csv",
    ),
    (
        "CLIP full-frame linear probe",
        ROOT / "results/nighttrap_supervised_baselines/fullframe_clip_linear_species_v1/predictions.csv",
    ),
    (
        "ConvNeXt-Tiny fine-tune, 8 epochs",
        ROOT / "results/nighttrap_supervised_baselines/convnext_tiny_finetune_species_v2_e8/predictions.csv",
    ),
    (
        "ViT-B/16 fine-tune, 2 epochs",
        ROOT / "results/nighttrap_supervised_baselines/vit_b16_finetune_species_v1/predictions.csv",
    ),
]

QWEN_ROW = (
    "Qwen3-VL-8B multiple-choice",
    ROOT / "results/nighttrap_vlm_ci_inputs/qwen3vl8b_full_freeze_v1/qwen3_species_predictions.csv",
)


def normalize_species(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^\([a-d]\)\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_choice(value: str) -> str:
    value = (value or "").strip().upper()
    match = re.search(r"\(([A-D])\)", value)
    return match.group(1) if match else value


def load_species_test() -> dict[str, dict[str, Any]]:
    with SPECIES_TEST.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        choices = {}
        for idx, choice in enumerate(row["choices"]):
            choices[normalize_species(choice)] = "ABCD"[idx]
        row["_choice_by_species"] = choices
        row["_answer_choice"] = "ABCD"[int(row["answer"])]
        index[str(row["id"])] = row
    return index


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def score_qwen(test_index: dict[str, dict[str, Any]], label: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_rows = read_csv(path)
    scored = []
    correct = 0
    for pred_row in pred_rows:
        sample_id = pred_row["id"]
        test_row = test_index[sample_id]
        gold_choice = test_row["_answer_choice"]
        pred_choice = normalize_choice(pred_row.get("pred", ""))
        is_correct = pred_choice == gold_choice
        correct += int(is_correct)
        scored.append(
            {
                "id": sample_id,
                "gold_species": normalize_species(pred_row.get("gt_taxon", "")),
                "pred_species": normalize_species(pred_row.get("pred_taxon", "")),
                "gold_choice": gold_choice,
                "projected_choice": pred_choice,
                "outcome": "correct" if is_correct else "wrong_choice",
                "correct": is_correct,
            }
        )
    n = len(scored)
    summary = {
        "model": label,
        "setting": "VLM A/B/C/D species selection",
        "n": n,
        "choice_accuracy": correct / n,
        "candidate_coverage": 1.0,
        "outside_option_rate": 0.0,
        "wrong_candidate_rate": (n - correct) / n,
        "closed_set_accuracy": None,
    }
    return summary, scored


def score_projected_supervised(
    test_index: dict[str, dict[str, Any]], label: str, path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_rows = read_csv(path)
    scored = []
    correct = 0
    in_options = 0
    outside = 0
    wrong_candidate = 0
    closed_correct = 0
    gold_missing = 0
    for pred_row in pred_rows:
        sample_id = pred_row["id"]
        test_row = test_index[sample_id]
        choices = test_row["_choice_by_species"]
        gold_species = normalize_species(pred_row.get("gold", ""))
        pred_species = normalize_species(pred_row.get("pred", ""))
        gold_choice = test_row["_answer_choice"]
        if gold_species not in choices:
            gold_missing += 1
        closed_is_correct = str(pred_row.get("correct", "")).lower() == "true"
        closed_correct += int(closed_is_correct)
        if pred_species in choices:
            in_options += 1
            projected_choice = choices[pred_species]
            if projected_choice == gold_choice:
                outcome = "correct"
                correct += 1
            else:
                outcome = "wrong_candidate"
                wrong_candidate += 1
        else:
            outside += 1
            projected_choice = "outside"
            outcome = "outside_candidate_set"
        scored.append(
            {
                "id": sample_id,
                "gold_species": gold_species,
                "pred_species": pred_species,
                "gold_choice": gold_choice,
                "projected_choice": projected_choice,
                "outcome": outcome,
                "correct": outcome == "correct",
                "closed_set_correct": closed_is_correct,
            }
        )
    n = len(scored)
    summary = {
        "model": label,
        "setting": "closed-set prediction projected to A/B/C/D choices",
        "n": n,
        "choice_accuracy": correct / n,
        "candidate_coverage": in_options / n,
        "outside_option_rate": outside / n,
        "wrong_candidate_rate": wrong_candidate / n,
        "closed_set_accuracy": closed_correct / n,
        "gold_species_missing_from_choices": gold_missing,
    }
    return summary, scored


def write_scored_rows(name: str, rows: list[dict[str, Any]]) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    path = OUT_DIR / f"{safe}.csv"
    if not rows:
        return str(path)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def write_table(summaries: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabularx}{\\textwidth}{YYrrrr}",
        "\\toprule",
        "Model & Species setting & N & Choice Acc. & In-choice pred. & Outside pred. \\\\",
        "\\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{latex_escape(row['model'])} & {latex_escape(row['setting'])} & "
            f"{row['n']:,} & {pct(row['choice_accuracy'])} & "
            f"{pct(row['candidate_coverage'])} & {pct(row['outside_option_rate'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\caption{Species candidate-set alignment diagnostic. Qwen3-VL-8B is evaluated in the released A/B/C/D species-selection setting. Supervised vision rows are closed-set species predictions projected onto the same four candidate choices after inference; predictions outside the four shown choices are counted as outside-option misses. This diagnostic clarifies the evaluation interface and should not be read as a probability-renormalized multiple-choice supervised baseline.}",
        "\\label{tab:species_candidate_alignment}",
        "\\end{table*}",
    ]
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_index = load_species_test()
    summaries: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}

    qwen_summary, qwen_rows = score_qwen(test_index, *QWEN_ROW)
    outputs[qwen_summary["model"]] = write_scored_rows(qwen_summary["model"], qwen_rows)
    summaries.append(qwen_summary)

    for label, path in SUPERVISED_ROWS:
        summary, rows = score_projected_supervised(test_index, label, path)
        outputs[label] = write_scored_rows(label, rows)
        summaries.append(summary)

    write_table(summaries)
    manifest = {
        "species_test": str(SPECIES_TEST),
        "table": str(TABLE_PATH),
        "outputs": outputs,
        "summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
