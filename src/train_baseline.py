"""Train an XGBoost baseline on the UCI Diabetes dataset and log to MLflow."""
from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data_loader import load_raw
from src.features import build_feature_matrix, clean, engineer

EXPERIMENT_NAME = "readmission-baseline"
RANDOM_STATE = 42


def build_pipeline(cat_cols, num_cols, scale_pos_weight: float) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    df = load_raw()
    df = clean(df)
    df = engineer(df)
    X, y, cat_cols, num_cols = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pos_rate = float(y_train.mean())
    scale_pos_weight = (1 - pos_rate) / pos_rate

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgb-baseline") as run:
        mlflow.log_params(
            {
                "rows_total": len(df),
                "rows_train": len(X_train),
                "rows_test": len(X_test),
                "pos_rate_train": round(pos_rate, 4),
                "scale_pos_weight": round(scale_pos_weight, 4),
                "num_numeric_features": len(num_cols),
                "num_categorical_features": len(cat_cols),
                "model": "XGBClassifier",
            }
        )

        pipe = build_pipeline(cat_cols, num_cols, scale_pos_weight)
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "f1": f1_score(y_test, preds),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print("\n=== Baseline results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification report (threshold=0.5):")
        print(classification_report(y_test, preds, digits=3))
        print(f"MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
