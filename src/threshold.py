"""Sweep decision thresholds on a held-out validation set and pick the best operating point.

Saves two thresholds:
  - best_f1: maximizes F1 (symmetric cost)
  - best_cost: maximizes recall-weighted F-beta where a missed readmission costs
    FN_COST_RATIO times more than a false alarm (clinically realistic)
The API can optionally use these for a binary prediction field; probabilities are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.pipeline import RANDOM_STATE, load_splits

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "champion.pkl"
OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
PLOT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
FN_COST_RATIO = 5.0


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python -m src.export_model` first to produce champion.pkl")

    pipeline = joblib.load(MODEL_PATH)
    X_train, X_test, y_train, y_test, _, _ = load_splits()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
    )
    proba_val = pipeline.predict_proba(X_val)[:, 1]
    proba_test = pipeline.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 181)
    beta = float(np.sqrt(FN_COST_RATIO))

    rows = []
    for t in thresholds:
        preds = (proba_val >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_val, preds, zero_division=0)),
            "recall": float(recall_score(y_val, preds, zero_division=0)),
            "f1": float(f1_score(y_val, preds, zero_division=0)),
            "fbeta": float(fbeta_score(y_val, preds, beta=beta, zero_division=0)),
        })

    best_f1 = max(rows, key=lambda r: r["f1"])
    best_cost = max(rows, key=lambda r: r["fbeta"])

    preds_at_best = (proba_test >= best_f1["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_at_best).ravel()
    preds_at_default = (proba_test >= 0.5).astype(int)
    tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, preds_at_default).ravel()

    result = {
        "best_threshold_f1": round(best_f1["threshold"], 4),
        "best_threshold_cost_weighted": round(best_cost["threshold"], 4),
        "fn_cost_ratio": FN_COST_RATIO,
        "default_threshold_0_5": {
            "precision": float(precision_score(y_test, preds_at_default, zero_division=0)),
            "recall": float(recall_score(y_test, preds_at_default, zero_division=0)),
            "f1": float(f1_score(y_test, preds_at_default, zero_division=0)),
            "confusion": {"tn": int(tn_d), "fp": int(fp_d), "fn": int(fn_d), "tp": int(tp_d)},
        },
        "tuned_threshold_f1": {
            "precision": float(precision_score(y_test, preds_at_best, zero_division=0)),
            "recall": float(recall_score(y_test, preds_at_best, zero_division=0)),
            "f1": float(f1_score(y_test, preds_at_best, zero_division=0)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "threshold.json").write_text(json.dumps(result, indent=2))

    ts = [r["threshold"] for r in rows]
    plt.figure(figsize=(9, 5))
    plt.plot(ts, [r["precision"] for r in rows], label="Precision")
    plt.plot(ts, [r["recall"] for r in rows], label="Recall")
    plt.plot(ts, [r["f1"] for r in rows], label="F1")
    plt.plot(ts, [r["fbeta"] for r in rows], label=f"Fβ (β={beta:.2f})", linestyle="--")
    plt.axvline(best_f1["threshold"], color="red", alpha=0.4, label=f"best F1 @ {best_f1['threshold']:.2f}")
    plt.axvline(best_cost["threshold"], color="purple", alpha=0.4, linestyle=":",
                label=f"best Fβ @ {best_cost['threshold']:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Threshold sweep on validation set")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = PLOT_DIR / "threshold_sweep.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {OUT_DIR / 'threshold.json'}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
