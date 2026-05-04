#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path("<ANIMALLAMA_ROOT>")

EXCLUDED_CONTEXT_FIELDS = [
    "audit_flags",
    "reference.species_ref_count_audit_only",
    "reference.species_ref_ratio_audit_only",
    "reference.month_neighbor_support_audit_only",
    "reference.site_common_species_names[] by default",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label(row: dict[str, Any]) -> int:
    return 0 if row.get("review_priority") == "routine" else 1


def image_paths(row: dict[str, Any]) -> list[str]:
    images = row.get("images") or {}
    out, seen = [], set()
    for key in ["first", "middle", "last"]:
        path = images.get(key)
        if path and path not in seen:
            seen.add(path)
            out.append(str(path))
    return out


def image_xy(rows: list[dict[str, Any]], embeddings: dict[str, np.ndarray]):
    xs, ys, kept = [], [], []
    for row in rows:
        feats = [embeddings[p] for p in image_paths(row) if p in embeddings]
        if not feats:
            continue
        xs.append(np.stack(feats, axis=0).mean(axis=0))
        ys.append(label(row))
        kept.append(row)
    return np.stack(xs).astype("float32"), np.array(ys, dtype=np.int64), kept


def clean_context(row: dict[str, Any], include_site_species_names: bool = False) -> dict[str, float]:
    ref = row.get("reference") or {}
    feats: dict[str, float] = {
        f"dataset={row.get('dataset_key') or 'unknown'}": 1.0,
        f"season={row.get('season') or 'unknown'}": 1.0,
        "reference_available": float(bool(ref.get("available"))),
        "reference_event_count": float(ref.get("event_count") or 0),
        "site_species_count": float(ref.get("site_species_count") or 0),
        "site_common_species_count": float(len(ref.get("site_common_species_names") or [])),
        "used_sqlite_fallback": float(bool(ref.get("used_sqlite_fallback"))),
    }
    if include_site_species_names:
        for name in (ref.get("site_common_species_names") or [])[:12]:
            feats[f"site_common_species={name}"] = 1.0
    return feats


def context_matrix(train, dev, test, include_site_species_names: bool):
    vec = DictVectorizer(sparse=False)
    x_train = vec.fit_transform([clean_context(row, include_site_species_names) for row in train]).astype("float32")
    x_dev = vec.transform([clean_context(row, include_site_species_names) for row in dev]).astype("float32")
    x_test = vec.transform([clean_context(row, include_site_species_names) for row in test]).astype("float32")
    return x_train, x_dev, x_test, vec


def dcg(rels):
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at(y: np.ndarray, score: np.ndarray, k: int) -> float | None:
    order = np.argsort(-score)[:k]
    denom = dcg(sorted(y.tolist(), reverse=True)[:k])
    return dcg([int(y[i]) for i in order]) / denom if denom else None


def auto_pass_at_recall(y: np.ndarray, score: np.ndarray, target_recall: float = 0.95) -> dict[str, Any]:
    positives = int(y.sum())
    best = {"auto_pass_n": 0, "auto_pass_rate": 0.0, "needs_review_recall": 1.0}
    if positives == 0:
        return best
    for threshold in np.unique(score):
        auto = score < threshold
        reviewed = ~auto
        recall = float(y[reviewed].sum() / positives)
        if recall >= target_recall and int(auto.sum()) > best["auto_pass_n"]:
            best = {"auto_pass_n": int(auto.sum()), "auto_pass_rate": float(auto.mean()), "needs_review_recall": recall}
    return best


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    base = float(y.mean())
    order = np.argsort(-score)
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": base,
        "auroc": float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else None,
        "auprc": float(average_precision_score(y, score)) if len(set(y.tolist())) == 2 else None,
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "ndcg@100": float(ndcg_at(y, score, min(100, len(y)))) if len(y) else None,
    }
    for k in [50, 100]:
        kk = min(k, len(y))
        precision = float(y[order[:kk]].sum() / kk) if kk else 0.0
        out[f"precision@{k}"] = precision
        out[f"enrichment@{k}"] = precision / base if base else None
    out["auto_pass@95R"] = auto_pass_at_recall(y, score, 0.95)["auto_pass_rate"]
    return out


def make_xgb(params: dict[str, Any]):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=8,
    )


def make_lgbm(params: dict[str, Any]):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        random_state=42,
        n_jobs=8,
        verbose=-1,
    )


def tune_and_eval(model_name: str, x_train, y_train, x_dev, y_dev, x_test, y_test):
    grid = []
    best = None
    for params in [
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.03},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.03},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    ]:
        clf = make_xgb(params) if model_name == "xgboost" else make_lgbm(params)
        clf.fit(x_train, y_train)
        dev_score = clf.predict_proba(x_dev)[:, 1]
        row = {**params, "dev_auprc": float(average_precision_score(y_dev, dev_score))}
        grid.append(row)
        if best is None or row["dev_auprc"] > best[0]["dev_auprc"]:
            best = (row, clf)
    assert best is not None
    score = best[1].predict_proba(x_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    return best[0], grid, score, pred


def fmt_pct(x: Any) -> str:
    return "--" if x is None else f"{100 * float(x):.2f}"


def fmt_enrich(x: Any) -> str:
    return "--" if x is None else f"{float(x):.2f}$\\times$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="results/nighttrap_supervised_baselines/needs_review_clip_context_ranker_v1")
    parser.add_argument("--out-dir", default="results/nighttrap_supervised_baselines/needs_review_tree_ranker_variants_clean_v1")
    parser.add_argument("--table", default="results/nighttrap_tables/table_needs_review_tree_rankers_clean_v1.tex")
    parser.add_argument("--include-site-species-names", action="store_true")
    args = parser.parse_args()

    split_dir = ROOT / args.split_dir
    out_dir = ROOT / args.out_dir
    table_path = ROOT / args.table
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    train0, dev0, test0 = [read_jsonl(split_dir / f"{name}.jsonl") for name in ["train", "dev", "test"]]
    data = np.load(split_dir / "clip_image_embeddings.npz", allow_pickle=True)
    embeddings = {str(k): v.astype("float32") for k, v in data.items()}
    x_img_train, y_train, train = image_xy(train0, embeddings)
    x_img_dev, y_dev, dev = image_xy(dev0, embeddings)
    x_img_test, y_test, test = image_xy(test0, embeddings)
    x_ctx_train, x_ctx_dev, x_ctx_test, vectorizer = context_matrix(train, dev, test, args.include_site_species_names)

    scaler = StandardScaler()
    x_img_train = scaler.fit_transform(x_img_train).astype("float32")
    x_img_dev = scaler.transform(x_img_dev).astype("float32")
    x_img_test = scaler.transform(x_img_test).astype("float32")
    variants = {
        "image-only": (x_img_train, x_img_dev, x_img_test),
        "clean-context-lite-only": (x_ctx_train, x_ctx_dev, x_ctx_test),
        "CLIP+clean-context-lite": (
            np.concatenate([x_img_train, x_ctx_train], axis=1),
            np.concatenate([x_img_dev, x_ctx_dev], axis=1),
            np.concatenate([x_img_test, x_ctx_test], axis=1),
        ),
    }

    results = []
    pred_rows = []
    unavailable = []
    for model_name in ["xgboost", "lightgbm"]:
        try:
            __import__("xgboost" if model_name == "xgboost" else "lightgbm")
        except Exception as exc:
            unavailable.append({"model": model_name, "error": repr(exc)})
            continue
        for setting, (x_train, x_dev, x_test) in variants.items():
            selected, grid, score, pred = tune_and_eval(model_name, x_train, y_train, x_dev, y_dev, x_test, y_test)
            row = {
                "setting": f"{model_name} {setting}",
                "model": model_name,
                "features": setting,
                "selected": selected,
                "grid": grid,
                **metrics(y_test, score, pred),
            }
            results.append(row)
            for src, s, p in zip(test, score.tolist(), pred.tolist()):
                pred_rows.append(
                    {
                        "setting": row["setting"],
                        "event_id": src.get("event_id"),
                        "site_key": src.get("site_key"),
                        "review_priority": src.get("review_priority"),
                        "gold_needs_review": label(src),
                        "score_needs_review": f"{s:.8f}",
                        "pred_at_0.5": int(p),
                    }
                )

    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["setting", "event_id", "site_key", "review_priority", "gold_needs_review", "score_needs_review", "pred_at_0.5"],
        )
        writer.writeheader()
        writer.writerows(pred_rows)

    summary = {
        "task": "needs-review recommendation",
        "split": "frozen CLIP/context-lite 913-event test split",
        "label_mapping": "routine -> 0; review + priority_review -> 1",
        "score_direction": "higher score means more likely needs_review",
        "source_split_dir": args.split_dir,
        "n_test": int(len(y_test)),
        "base_rate": float(y_test.mean()),
        "context_policy": {
            "name": "clean context-lite tree features",
            "included": vectorizer.get_feature_names_out().tolist(),
            "excluded": EXCLUDED_CONTEXT_FIELDS,
            "include_site_species_names": bool(args.include_site_species_names),
        },
        "unavailable": unavailable,
        "variants": results,
        "outputs": {"predictions": str(out_dir / "predictions.csv"), "table": args.table},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "\\begin{table*}[h]",
        "\\centering",
        "\\caption{Clean tree-based needs-review ranker variants on the frozen 913-event CLIP/context-lite split. Clean context-lite excludes audit flags, audit-only current-species support fields, and site common-species name expansion by default. Values are percentages except enrichment.}",
        "\\label{tab:needs_review_tree_rankers_clean}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Setting & N & AUROC & AUPRC & Macro-F1 & P@50 & P@100 & E@50 & AP@95R \\\\",
        "\\midrule",
    ]
    for row in results:
        lines.append(
            f"{row['setting']} & {row['n']} & {fmt_pct(row['auroc'])} & {fmt_pct(row['auprc'])} & "
            f"{fmt_pct(row['macro_f1'])} & {fmt_pct(row['precision@50'])} & {fmt_pct(row['precision@100'])} & "
            f"{fmt_enrich(row['enrichment@50'])} & {fmt_pct(row['auto_pass@95R'])} \\\\"
        )
    if unavailable:
        for row in unavailable:
            lines.append(f"{row['model']} unavailable & -- & -- & -- & -- & -- & -- & -- & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "table": str(table_path), "n_results": len(results), "unavailable": unavailable}, indent=2))


if __name__ == "__main__":
    main()
