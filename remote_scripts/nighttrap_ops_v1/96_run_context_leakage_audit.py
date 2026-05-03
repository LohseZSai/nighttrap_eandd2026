#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("<ANIMALLAMA_ROOT>")
SPLIT_DIR = ROOT / "results/nighttrap_supervised_baselines/needs_review_clip_context_ranker_v1"
OUT = ROOT / "results/nighttrap_diagnostics/context_leakage_audit_v08"
TABLE_DIR = ROOT / "results/nighttrap_tables"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label(row: dict) -> int:
    return 0 if row.get("review_priority") == "routine" else 1


def image_paths(row: dict) -> list[str]:
    images = row.get("images") or {}
    vals = [images.get("first"), images.get("middle"), images.get("last")]
    out, seen = [], set()
    for path in vals:
        if path and path not in seen:
            seen.add(path)
            out.append(str(path))
    return out


def image_xy(rows: list[dict], embeddings: dict[str, np.ndarray]):
    xs, ys, kept = [], [], []
    for row in rows:
        feats = [embeddings[p] for p in image_paths(row) if p in embeddings]
        if not feats:
            continue
        xs.append(np.stack(feats, axis=0).mean(axis=0))
        ys.append(label(row))
        kept.append(row)
    return np.stack(xs).astype("float32"), np.array(ys, dtype=np.int64), kept


def context(row: dict, mode: str = "full") -> dict:
    ref = row.get("reference") or {}
    feats = {}
    if mode in {"full", "shuffled"}:
        feats[f"season={row.get('season') or 'unknown'}"] = 1
        feats["reference_available"] = int(bool(ref.get("available")))
        common = ref.get("site_common_species_names") or []
        feats["site_common_species_count"] = float(len(common))
        for name in common[:10]:
            feats[f"site_common_species={name}"] = 1
        feats["used_sqlite_fallback"] = int(bool(ref.get("used_sqlite_fallback")))
    elif mode == "site_frequency":
        feats["reference_available"] = int(bool(ref.get("available")))
        feats["site_common_species_count"] = float(len(ref.get("site_common_species_names") or []))
        feats["reference_event_count"] = float(ref.get("event_count") or 0)
        feats["site_species_count"] = float(ref.get("site_species_count") or 0)
    return feats


def context_matrix(train, dev, test, mode: str):
    vec = DictVectorizer(sparse=False)
    x_train = vec.fit_transform([context(row, mode) for row in train]).astype("float32")
    x_dev = vec.transform([context(row, mode) for row in dev]).astype("float32")
    x_test = vec.transform([context(row, mode) for row in test]).astype("float32")
    return x_train, x_dev, x_test


def dcg(rels):
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at(y, score, k):
    order = np.argsort(-score)[:k]
    denom = dcg(sorted(y.tolist(), reverse=True)[:k])
    return dcg([int(y[i]) for i in order]) / denom if denom else None


def metrics(y, score, pred):
    base = float(y.mean())
    order = np.argsort(-score)
    out = {
        "n": int(len(y)),
        "base_rate": base,
        "auroc": float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else None,
        "auprc": float(average_precision_score(y, score)) if len(set(y.tolist())) == 2 else None,
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro")),
        "ndcg@100": float(ndcg_at(y, score, min(100, len(y)))) if len(y) else None,
    }
    for k in [50, 100]:
        kk = min(k, len(y))
        precision = float(y[order[:kk]].sum() / kk) if kk else 0.0
        out[f"precision@{k}"] = precision
        out[f"enrichment@{k}"] = precision / base if base else None
    return out


def tune(x_train, y_train, x_dev, y_dev):
    best = None
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced", solver="lbfgs"))
        clf.fit(x_train, y_train)
        score = clf.predict_proba(x_dev)[:, 1]
        ap = average_precision_score(y_dev, score)
        if best is None or ap > best[0]:
            best = (ap, c, clf)
    return best[1], best[2]


def run_model(name, x_train, y_train, x_dev, y_dev, x_test, y_test):
    c, clf = tune(x_train, y_train, x_dev, y_dev)
    score = clf.predict_proba(x_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    row = {"setting": name, "selected_C": c, **metrics(y_test, score, pred)}
    return row


def fmt(x):
    return "--" if x is None else f"{x:.3f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    train, dev, test = [read_jsonl(SPLIT_DIR / f"{name}.jsonl") for name in ["train", "dev", "test"]]
    data = np.load(SPLIT_DIR / "clip_image_embeddings.npz", allow_pickle=True)
    embeddings = {str(k): v.astype("float32") for k, v in data.items()}
    x_img_train, y_train, train = image_xy(train, embeddings)
    x_img_dev, y_dev, dev = image_xy(dev, embeddings)
    x_img_test, y_test, test = image_xy(test, embeddings)
    ctx_train, ctx_dev, ctx_test = context_matrix(train, dev, test, "full")
    sf_train, sf_dev, sf_test = context_matrix(train, dev, test, "site_frequency")

    rng = np.random.default_rng(20260503)
    shuffled = ctx_test.copy()
    rng.shuffle(shuffled, axis=0)

    results = [
        run_model("image-only", x_img_train, y_train, x_img_dev, y_dev, x_img_test, y_test),
        run_model("context-lite-only", ctx_train, y_train, ctx_dev, y_dev, ctx_test, y_test),
        run_model("full context-lite", np.concatenate([x_img_train, ctx_train], axis=1), y_train, np.concatenate([x_img_dev, ctx_dev], axis=1), y_dev, np.concatenate([x_img_test, ctx_test], axis=1), y_test),
        run_model("shuffled-context", np.concatenate([x_img_train, ctx_train], axis=1), y_train, np.concatenate([x_img_dev, ctx_dev], axis=1), y_dev, np.concatenate([x_img_test, shuffled], axis=1), y_test),
        run_model("site-frequency-only", sf_train, y_train, sf_dev, y_dev, sf_test, y_test),
    ]

    # Simple non-learned rule: missing or very sparse reference history is more likely to need review.
    rule_score = np.array([1.0 - min(float((row.get("reference") or {}).get("event_count") or 0), 100.0) / 100.0 for row in test], dtype="float32")
    rule_pred = (rule_score >= 0.5).astype(int)
    results.append({"setting": "rule-only", "selected_C": None, **metrics(y_test, rule_score, rule_pred)})

    summary = {
        "task": "needs-review recommendation",
        "label_mapping": "routine -> 0; review + priority_review -> 1",
        "source_split_dir": str(SPLIT_DIR.relative_to(ROOT)),
        "results": results,
        "interpretation": {
            "context_lite_only_solves_task": "compare context-lite-only with full context-lite and image-only",
            "shuffled_context_test": "uses full model with test context rows permuted deterministically",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    for row in results:
        lines.append(
            f"{row['setting']} & {row['n']} & {fmt(row['auroc'])} & {fmt(row['auprc'])} & "
            f"{fmt(row['macro_f1'])} & {fmt(row['precision@50'])} & {fmt(row['precision@100'])} & "
            f"{fmt(row['enrichment@50'])}$\\times$ & {fmt(row['ndcg@100'])} \\\\"
        )
    (TABLE_DIR / "table_context_leakage_v08.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "table": str(TABLE_DIR / "table_context_leakage_v08.tex")}, indent=2))


if __name__ == "__main__":
    main()
