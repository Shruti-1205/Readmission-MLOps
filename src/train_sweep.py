"""Train 5 candidate models on the same splits and log each to MLflow for comparison."""
from __future__ import annotations

import time

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.pipeline import (
    RANDOM_STATE,
    build_preprocessor,
    compute_scale_pos_weight,
    eval_metrics,
    load_splits,
    safe_input_example,
)

EXPERIMENT_NAME = "readmission-sweep"


def model_zoo(scale_pos_weight: float) -> dict:
    """5-model lineup. Tree models get scale_pos_weight; linear/MLP use class_weight='balanced'."""
    return {
        "logreg": (
            LogisticRegression(
                max_iter=3000, C=1.0, class_weight="balanced",
                solver="lbfgs", n_jobs=-1, random_state=RANDOM_STATE,
            ),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=400, max_depth=18, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
            ),
            False,
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight, tree_method="hist",
                eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1,
            ),
            False,
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=500, num_leaves=63, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            ),
            False,
        ),
        "mlp": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=40,
                early_stopping=True, random_state=RANDOM_STATE,
            ),
            True,
        ),
    }


def main() -> None:
    X_train, X_test, y_train, y_test, cat_cols, num_cols = load_splits()
    spw = compute_scale_pos_weight(y_train)

    mlflow.set_experiment(EXPERIMENT_NAME)
    results = []

    for name, (estimator, scale_numeric) in model_zoo(spw).items():
        with mlflow.start_run(run_name=name) as run:
            pre = build_preprocessor(cat_cols, num_cols, scale_numeric=scale_numeric)
            pipe = Pipeline([("pre", pre), ("clf", estimator)])

            t0 = time.time()
            pipe.fit(X_train, y_train)
            fit_secs = time.time() - t0

            proba = pipe.predict_proba(X_test)[:, 1]
            metrics = eval_metrics(y_test, proba)
            metrics["fit_seconds"] = round(fit_secs, 2)

            mlflow.log_params({
                "model": name,
                "scale_pos_weight": round(spw, 4),
                "rows_train": len(X_train),
                "rows_test": len(X_test),
            })
            mlflow.log_metrics(metrics)

            sample = safe_input_example(X_train, n=5)
            signature = infer_signature(sample, pipe.predict_proba(sample)[:, 1])
            mlflow.sklearn.log_model(
                pipe, artifact_path="model",
                signature=signature, input_example=sample,
            )

            results.append((name, metrics, run.info.run_id))
            print(f"[{name:>14}] roc_auc={metrics['roc_auc']:.4f}  "
                  f"pr_auc={metrics['pr_auc']:.4f}  f1={metrics['f1']:.4f}  "
                  f"fit={fit_secs:.1f}s")

    print("\n=== Leaderboard (by ROC-AUC) ===")
    for name, m, rid in sorted(results, key=lambda r: -r[1]["roc_auc"]):
        print(f"  {name:>14}  roc_auc={m['roc_auc']:.4f}  pr_auc={m['pr_auc']:.4f}  run_id={rid[:8]}")


if __name__ == "__main__":
    main()
