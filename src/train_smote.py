"""SMOTE ablation: does oversampling beat scale_pos_weight for XGBoost on this dataset?

Applies SMOTE AFTER preprocessing (OHE) but BEFORE the classifier, so the test set
is never oversampled. Logged head-to-head with the XGBoost baseline in MLflow.
"""
from __future__ import annotations

import time

import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.models import infer_signature
from xgboost import XGBClassifier

from src.pipeline import (
    RANDOM_STATE,
    build_preprocessor,
    eval_metrics,
    load_splits,
    safe_input_example,
)

EXPERIMENT_NAME = "readmission-sweep"


def main() -> None:
    X_train, X_test, y_train, y_test, cat_cols, num_cols = load_splits()

    pre = build_preprocessor(cat_cols, num_cols)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", eval_metric="auc",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    pipe = ImbPipeline([("pre", pre), ("smote", smote), ("clf", clf)])

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgboost-smote"):
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_secs = round(time.time() - t0, 1)

        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = eval_metrics(y_test, proba)
        metrics["fit_seconds"] = fit_secs

        mlflow.log_params({
            "model": "xgboost-smote",
            "imbalance_strategy": "SMOTE(k_neighbors=5)",
            "rows_train": len(X_train),
        })
        mlflow.log_metrics(metrics)

        sample = safe_input_example(X_train)
        signature = infer_signature(sample, pipe.predict_proba(sample)[:, 1])
        mlflow.sklearn.log_model(
            pipe, artifact_path="model",
            signature=signature, input_example=sample,
        )

        print(f"[xgboost-smote] roc_auc={metrics['roc_auc']:.4f}  "
              f"pr_auc={metrics['pr_auc']:.4f}  f1={metrics['f1']:.4f}  fit={fit_secs}s")
        print("\nCompare to the xgboost baseline in the `readmission-sweep` experiment.")


if __name__ == "__main__":
    main()
