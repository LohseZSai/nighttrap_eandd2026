from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/nighttrap_tables"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260504)
BOOT = 1000


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def ci(values: list[float], scale: float = 100.0) -> str:
    arr = np.asarray(values, dtype=float) * scale
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return f"[{lo:.2f}, {hi:.2f}]"


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    positives = y_sorted.sum()
    if positives == 0:
        return float("nan")
    cum_pos = np.cumsum(y_sorted)
    precision = cum_pos / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / positives)


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    pos_rank_sum = ranks[y == 1].sum()
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def p_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-score, kind="mergesort")[:k]
    return float(y[order].mean())


def ndcg_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-score, kind="mergesort")[:k]
    gains = y[order].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(y)[::-1][:k].astype(float)
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg else 0.0


def auto_pass_at_recall(y: np.ndarray, score: np.ndarray, target: float = 0.95) -> float:
    total_pos = int(y.sum())
    if total_pos == 0:
        return 0.0
    max_pos_passed = int(np.floor((1.0 - target) * total_pos + 1e-12))
    order = np.argsort(score, kind="mergesort")
    passed_pos = 0
    passed = 0
    for idx in order:
        if y[idx] == 1 and passed_pos + 1 > max_pos_passed:
            break
        passed += 1
        passed_pos += int(y[idx] == 1)
    return passed / len(y)


def load_labels(path: str | None) -> list[str] | None:
    if path is None:
        return None
    rows = read_csv(ROOT / path)
    if not rows:
        return []
    first_key = "class" if "class" in rows[0] else next(iter(rows[0]))
    return [row[first_key] for row in rows]


def choice_text(question: dict, choice: str) -> str:
    if not choice or len(choice) < 2:
        return "missing"
    idx = ord(choice[1].upper()) - ord("A")
    choices = question.get("choices") or []
    if idx < 0 or idx >= len(choices):
        return "missing"
    return str(choices[idx]).strip()


def classification_row(
    name: str,
    task: str,
    path: str,
    label_mode: str,
    labels_path: str | None = None,
    questions_path: str | None = None,
) -> dict[str, str]:
    rows = read_csv(ROOT / path)
    if label_mode == "taxon":
        gold = [r["gt_taxon"] for r in rows]
        pred = [r["pred_taxon"] for r in rows]
    elif label_mode == "question_choice":
        if questions_path is None:
            raise ValueError("questions_path is required for question_choice labels")
        questions = {row["id"]: row for row in json.load(open(ROOT / questions_path, encoding="utf-8"))}
        gold = [choice_text(questions[r["id"]], r["gt"]) for r in rows]
        pred = [choice_text(questions[r["id"]], r["pred"]) for r in rows]
    else:
        gold = [r["gold"] if "gold" in r else r["gt"] for r in rows]
        pred = [r["pred"] for r in rows]
    labels = load_labels(labels_path) or sorted(set(gold) | set(pred))
    n = len(rows)
    acc = accuracy(gold, pred)
    mf1 = macro_f1(gold, pred, labels)
    acc_boot, f1_boot = [], []
    strata: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(gold):
        strata[label].append(i)
    strata_arrays = [np.asarray(v, dtype=int) for v in strata.values()]
    for _ in range(BOOT):
        sample = np.concatenate([RNG.choice(part, size=len(part), replace=True) for part in strata_arrays])
        bg = [gold[i] for i in sample]
        bp = [pred[i] for i in sample]
        acc_boot.append(accuracy(bg, bp))
        f1_boot.append(macro_f1(bg, bp, labels))
    return {
        "model": name,
        "task": task,
        "n": f"{n:,}",
        "acc": pct(acc),
        "acc_ci": ci(acc_boot),
        "f1": pct(mf1),
        "f1_ci": ci(f1_boot),
    }


def task5_rows_from(path: str, settings: list[tuple[str, str]]) -> list[dict[str, str]]:
    rows = read_csv(ROOT / path)
    by_setting: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_setting[row["setting"]].append(row)
    out = []
    for setting, label in settings:
        out.append(task5_row(label, by_setting[setting]))
    return out


def task5_row(label: str, rows: list[dict[str, str]]) -> dict[str, str]:
    y = np.asarray([int(r["gold_needs_review"]) for r in rows], dtype=int)
    score = np.asarray([float(r["score_needs_review"]) for r in rows], dtype=float)
    pred_key = "pred_needs_review" if "pred_needs_review" in rows[0] else "pred_at_0.5"
    pred = [str(int(float(r[pred_key]))) for r in rows]
    gold = [str(v) for v in y.tolist()]
    labels = ["0", "1"]
    n = len(rows)
    base = float(y.mean())
    metrics = {
        "auroc": auroc(y, score),
        "auprc": average_precision(y, score),
        "macro_f1": macro_f1(gold, pred, labels),
        "p50": p_at_k(y, score, 50),
        "ndcg100": ndcg_at_k(y, score, 100),
        "ap95": auto_pass_at_recall(y, score, 0.95),
    }
    boot = {k: [] for k in metrics}
    idx = np.arange(n)
    for _ in range(BOOT):
        sample = RNG.choice(idx, size=n, replace=True)
        by = y[sample]
        bs = score[sample]
        bg = [str(v) for v in by.tolist()]
        bp = [pred[i] for i in sample]
        vals = {
            "auroc": auroc(by, bs),
            "auprc": average_precision(by, bs),
            "macro_f1": macro_f1(bg, bp, labels),
            "p50": p_at_k(by, bs, 50),
            "ndcg100": ndcg_at_k(by, bs, 100),
            "ap95": auto_pass_at_recall(by, bs, 0.95),
        }
        for key, value in vals.items():
            if not np.isnan(value):
                boot[key].append(value)
    return {
        "model": label,
        "n": f"{n:,}",
        "base": pct(base),
        "auroc": pct(metrics["auroc"]),
        "auroc_ci": ci(boot["auroc"]),
        "auprc": pct(metrics["auprc"]),
        "auprc_ci": ci(boot["auprc"]),
        "f1": pct(metrics["macro_f1"]),
        "f1_ci": ci(boot["macro_f1"]),
        "p50": pct(metrics["p50"]),
        "p50_ci": ci(boot["p50"]),
        "ndcg": pct(metrics["ndcg100"]),
        "ndcg_ci": ci(boot["ndcg100"]),
        "ap95": pct(metrics["ap95"]),
        "ap95_ci": ci(boot["ap95"]),
    }


def write_classification_table(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\caption{Bootstrap confidence intervals for selected main classification rows. Intervals are empirical 95\% bootstrap intervals over test events; they are diagnostic intervals, not a replacement for source-wise robustness checks.}",
        r"\label{tab:main_classification_ci}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Model & Task & N & Accuracy [95\% CI] & Macro-F1 [95\% CI] \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['task']} & {row['n']} & "
            f"{row['acc']} {row['acc_ci']} & {row['f1']} {row['f1_ci']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (OUT / "table_main_classification_ci_v1.tex").write_text("\n".join(lines), encoding="utf-8")


def write_task5_table(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\caption{Bootstrap confidence intervals for selected needs-review queue metrics. Intervals are empirical 95\% bootstrap intervals over events within each frozen split; rows from different splits are not a single leaderboard.}",
        r"\label{tab:task5_bootstrap_ci}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model / split & N & Base & AUROC [95\% CI] & AUPRC [95\% CI] & Macro-F1 [95\% CI] & P@50 [95\% CI] & AP@95R [95\% CI] \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['n']} & {row['base']} & "
            f"{row['auroc']} {row['auroc_ci']} & {row['auprc']} {row['auprc_ci']} & "
            f"{row['f1']} {row['f1_ci']} & {row['p50']} {row['p50_ci']} & "
            f"{row['ap95']} {row['ap95_ci']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]
    (OUT / "table_task5_bootstrap_ci_v1.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    classification = [
        classification_row(
            "Qwen3-VL-8B",
            "Image usability",
            "results/nighttrap_vlm_ci_inputs/qwen3vl8b_full_freeze_v1/qwen3_usability_predictions.csv",
            "choice",
        ),
        classification_row(
            "Qwen3-VL-8B",
            "Empty-event",
            "results/nighttrap_vlm_ci_inputs/qwen3vl8b_full_freeze_v1/qwen3_empty_predictions.csv",
            "choice",
        ),
        classification_row(
            "Qwen3-VL-8B",
            "Species",
            "results/nighttrap_vlm_ci_inputs/qwen3vl8b_full_freeze_v1/qwen3_species_predictions.csv",
            "question_choice",
            questions_path="release/_remote_cache/AnimaLLaMA/results/nighttrap_ops_v1_build/track_b_species/test.json",
        ),
        classification_row(
            "Qwen3-VL-8B",
            "Count-bin",
            "results/nighttrap_vlm_ci_inputs/qwen3vl8b_full_freeze_v1/qwen3_count_predictions.csv",
            "choice",
        ),
        classification_row(
            "Best supervised row",
            "Image usability",
            "results/nighttrap_supervised_baselines/vit_b16_finetune_usability_v1/predictions.csv",
            "choice",
        ),
        classification_row(
            "Best supervised row",
            "Empty-event",
            "results/nighttrap_supervised_baselines/convnext_tiny_finetune_empty_v1/predictions.csv",
            "choice",
        ),
        classification_row(
            "Best supervised row",
            "Species",
            "results/nighttrap_supervised_baselines/fullframe_dinov2_linear_species_v1/species_predictions.csv",
            "choice",
            labels_path="results/nighttrap_supervised_baselines/fullframe_dinov2_linear_species_v1/per_class_f1.csv",
        ),
        classification_row(
            "Best supervised row",
            "Count-bin",
            "results/nighttrap_supervised_baselines/convnext_tiny_finetune_count_v1/predictions.csv",
            "choice",
        ),
    ]
    write_classification_table(classification)

    task5 = [
        task5_row(
            "Qwen3-VL-8B routine-stress",
            read_csv(ROOT / "results/nighttrap_qwen3_score_v1/qwen3_routine_stress_predictions.csv"),
        )
    ]
    task5 += task5_rows_from(
        "results/nighttrap_supervised_baselines/needs_review_ranker_variants_v1/predictions.csv",
        [("CLIP+context-lite logistic", "CLIP/context-lite logistic, 913 split")],
    )
    task5 += task5_rows_from(
        "results/nighttrap_supervised_baselines/needs_review_tree_ranker_variants_clean_v1/predictions.csv",
        [
            ("xgboost clean-context-lite-only", "XGBoost clean context-lite, 913 split"),
            ("xgboost CLIP+clean-context-lite", "XGBoost CLIP+clean context, 913 split"),
            ("lightgbm clean-context-lite-only", "LightGBM clean context-lite, 913 split"),
            ("lightgbm CLIP+clean-context-lite", "LightGBM CLIP+clean context, 913 split"),
        ],
    )
    write_task5_table(task5)

    manifest = {
        "bootstrap_replicates": BOOT,
        "seed": 20260504,
        "outputs": [
            "results/nighttrap_tables/table_main_classification_ci_v1.tex",
            "results/nighttrap_tables/table_task5_bootstrap_ci_v1.tex",
        ],
    }
    (OUT / "reviewer_response_metrics_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
