#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("<ANIMALLAMA_ROOT>")
SPLIT_DIR = ROOT / "results/nighttrap_supervised_baselines/needs_review_clip_context_ranker_v1"
OUT = ROOT / "results/nighttrap_diagnostics/event_representation_ablation_v08"
TABLE_DIR = ROOT / "results/nighttrap_tables"
LABELS = [0, 1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label(row: dict) -> int:
    return 0 if row.get("review_priority") == "routine" else 1


def paths_for(row: dict, mode: str) -> list[str]:
    images = row.get("images") or {}
    ordered = [images.get("first"), images.get("middle"), images.get("last")]
    if mode == "first":
        ordered = ordered[:1]
    elif mode == "middle":
        ordered = ordered[1:2]
    elif mode == "last":
        ordered = ordered[2:3]
    out, seen = [], set()
    for path in ordered:
        if path and path not in seen:
            seen.add(path)
            out.append(str(path))
    return out


def is_unique(row: dict) -> bool:
    return len(set(paths_for(row, "mean"))) > 1


def featurize(rows: list[dict], embeddings: dict[str, np.ndarray], mode: str):
    xs, ys, kept = [], [], []
    for row in rows:
        feats = [embeddings[p] for p in paths_for(row, mode) if p in embeddings]
        if not feats:
            continue
        xs.append(np.stack(feats, axis=0).mean(axis=0))
        ys.append(label(row))
        kept.append(row)
    return np.stack(xs).astype("float32"), np.array(ys, dtype=np.int64), kept


def dcg(rels: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-score)[:k]
    ideal = sorted(y.tolist(), reverse=True)[:k]
    denom = dcg(ideal)
    return dcg([int(y[i]) for i in order]) / denom if denom else 0.0


def rank_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    base = float(y.mean()) if len(y) else 0.0
    order = np.argsort(-score)
    out = {
        "n": int(len(y)),
        "base_rate": base,
        "auroc": float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else None,
        "auprc": float(average_precision_score(y, score)) if len(set(y.tolist())) == 2 else None,
        "ndcg@100": float(ndcg_at(y, score, min(100, len(y)))) if len(y) else None,
    }
    for k in [50, 100]:
        kk = min(k, len(y))
        hits = int(y[order[:kk]].sum()) if kk else 0
        precision = hits / kk if kk else 0.0
        out[f"precision@{k}"] = precision
        out[f"enrichment@{k}"] = precision / base if base else None
    return out


def safe_auto_pass(y: np.ndarray, score: np.ndarray, target: float = 0.95) -> dict:
    order = np.argsort(score)
    total_pos = int(y.sum())
    best = {"auto_pass_n": 0, "auto_pass_rate": 0.0, "safe_auto_pass": 0, "unsafe_auto_pass": 0, "needs_review_recall": 1.0}
    for n_pass in range(0, len(y) + 1):
        passed = order[:n_pass]
        missed = int(y[passed].sum()) if n_pass else 0
        recall = (total_pos - missed) / total_pos if total_pos else 0.0
        if recall >= target:
            best = {
                "auto_pass_n": n_pass,
                "auto_pass_rate": n_pass / len(y) if len(y) else 0.0,
                "safe_auto_pass": int(n_pass - missed),
                "unsafe_auto_pass": missed,
                "needs_review_recall": recall,
            }
        else:
            break
    return best


def tune_and_eval(train_rows, dev_rows, test_rows, embeddings, mode: str, subset: str | None = None) -> dict:
    x_train, y_train, _ = featurize(train_rows, embeddings, mode)
    x_dev, y_dev, _ = featurize(dev_rows, embeddings, mode)
    if subset == "unique":
        test_rows = [row for row in test_rows if is_unique(row)]
    elif subset == "duplicate":
        test_rows = [row for row in test_rows if not is_unique(row)]
    x_test, y_test, kept = featurize(test_rows, embeddings, mode)
    best = None
    tuning = []
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced", solver="lbfgs"))
        clf.fit(x_train, y_train)
        dev_score = clf.predict_proba(x_dev)[:, 1]
        row = {"C": c, "dev_auprc": float(average_precision_score(y_dev, dev_score))}
        tuning.append(row)
        if best is None or row["dev_auprc"] > best[0]["dev_auprc"]:
            best = (row, clf)
    selected, clf = best
    score = clf.predict_proba(x_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    return {
        "representation": subset or mode,
        "mode": mode,
        "subset": subset,
        "selected": selected,
        "label_distribution": dict(Counter(y_test.tolist())),
        "macro_f1": float(f1_score(y_test, pred, labels=LABELS, average="macro")),
        **rank_metrics(y_test, score),
        "auto_pass@95R": safe_auto_pass(y_test, score, 0.95),
    }


def fmt(x):
    return "--" if x is None else f"{x:.3f}"


def pct(x):
    return "--" if x is None else f"{100 * x:.2f}\\%"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    splits = {name: read_jsonl(SPLIT_DIR / f"{name}.jsonl") for name in ["train", "dev", "test"]}
    data = np.load(SPLIT_DIR / "clip_image_embeddings.npz", allow_pickle=True)
    embeddings = {str(k): v.astype("float32") for k, v in data.items()}
    configs = [
        ("first-frame-only", "first", None),
        ("middle-frame-only", "middle", None),
        ("last-frame-only", "last", None),
        ("three-slot mean", "mean", None),
        ("unique-frame-only subset", "mean", "unique"),
        ("duplicate-only subset", "mean", "duplicate"),
    ]
    results = []
    for name, mode, subset in configs:
        row = tune_and_eval(splits["train"], splits["dev"], splits["test"], embeddings, mode, subset)
        row["representation"] = name
        results.append(row)
    summary = {
        "task": "needs-review recommendation",
        "label_mapping": "routine -> 0; review + priority_review -> 1",
        "model": "CLIP logistic probe over event image embeddings",
        "source_split_dir": str(SPLIT_DIR.relative_to(ROOT)),
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = []
    for row in results:
        ap = row["auto_pass@95R"]
        lines.append(
            f"{row['representation']} & {row['n']} & {fmt(row['base_rate'])} & {fmt(row['auroc'])} & "
            f"{fmt(row['auprc'])} & {fmt(row['macro_f1'])} & {fmt(row['precision@50'])} & "
            f"{fmt(row['precision@100'])} & {fmt(row['enrichment@50'])}$\\times$ & "
            f"{fmt(row['enrichment@100'])}$\\times$ & {pct(ap['auto_pass_rate'])} \\\\"
        )
    (TABLE_DIR / "table_event_representation_v08.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "table": str(TABLE_DIR / "table_event_representation_v08.tex")}, indent=2))


if __name__ == "__main__":
    main()
