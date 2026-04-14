"""Subgroup fairness audit: per-race and per-gender AUC, FPR, FNR, and prevalence."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.pipeline import load_splits

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "champion.pkl"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
PLOT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
MIN_GROUP_SIZE = 200
DEFAULT_THRESHOLD = 0.30


def subgroup_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    if y_true.nunique() < 2:
        return {"n": int(len(y_true)), "note": "single-class subgroup — AUC undefined"}

    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else None,
        "precision_at_thr": float(tp / (tp + fp)) if (tp + fp) > 0 else None,
        "recall_at_thr": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
    }


def audit_group(df: pd.DataFrame, y: pd.Series, proba: np.ndarray, col: str, threshold: float) -> dict:
    out = {}
    for val in df[col].dropna().unique():
        mask = df[col] == val
        if mask.sum() < MIN_GROUP_SIZE:
            continue
        out[str(val)] = subgroup_metrics(y[mask], proba[mask.values], threshold)
    return out


def plot_subgroup_aucs(report: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, group in zip(axes, ["race", "gender"]):
        groups = report[group]
        labels = [k for k, v in groups.items() if "roc_auc" in v]
        aucs = [groups[k]["roc_auc"] for k in labels]
        ns = [groups[k]["n"] for k in labels]
        order = np.argsort(aucs)
        bars = ax.barh([labels[i] for i in order], [aucs[i] for i in order])
        for bar, i in zip(bars, order):
            ax.text(aucs[i] + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"n={ns[i]:,}", va="center", fontsize=9)
        ax.set_xlim(0.55, 0.80)
        ax.set_xlabel("ROC-AUC")
        ax.set_title(f"ROC-AUC by {group}")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python -m src.export_model` first.")

    pipeline = joblib.load(MODEL_PATH)
    X_train, X_test, y_train, y_test, _, _ = load_splits()
    proba = pipeline.predict_proba(X_test)[:, 1]

    threshold_path = MODEL_PATH.parent / "threshold.json"
    thr = DEFAULT_THRESHOLD
    if threshold_path.exists():
        thr = json.loads(threshold_path.read_text()).get("best_threshold_f1", DEFAULT_THRESHOLD)
    print(f"Using decision threshold = {thr}")

    overall = subgroup_metrics(y_test, proba, thr)
    report = {
        "threshold_used": thr,
        "overall": overall,
        "race": audit_group(X_test, y_test, proba, "race", thr),
        "gender": audit_group(X_test, y_test, proba, "gender", thr),
    }

    def spread(group_dict: dict, key: str) -> dict:
        vals = [v[key] for v in group_dict.values() if key in v and v[key] is not None]
        return {
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
            "gap": round(max(vals) - min(vals), 4) if len(vals) >= 2 else None,
        }

    report["fairness_gaps"] = {
        "race": {k: spread(report["race"], k) for k in ["roc_auc", "fpr", "fnr"]},
        "gender": {k: spread(report["gender"], k) for k in ["roc_auc", "fpr", "fnr"]},
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "fairness_report.json").write_text(json.dumps(report, indent=2))
    plot_subgroup_aucs(report, PLOT_DIR / "fairness_subgroup_auc.png")

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {REPORTS_DIR / 'fairness_report.json'}")
    print(f"Saved: {PLOT_DIR / 'fairness_subgroup_auc.png'}")


if __name__ == "__main__":
    main()
