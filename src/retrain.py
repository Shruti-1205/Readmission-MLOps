"""Retrain on reference + current data, then promote the new model only if it beats production.

Run this after `src.drift` reports drift. The script guarantees no silent regressions:
the new model must exceed the current Production champion's test ROC-AUC by a safety
margin before it becomes the new Production alias.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import build_feature_matrix
from src.pipeline import (
    RANDOM_STATE,
    build_preprocessor,
    compute_scale_pos_weight,
    eval_metrics,
    safe_input_example,
)
from src.register import MODEL_NAME, build_model_card

ALIAS = "production"
EXPERIMENT_NAME = "readmission-retrain"
IMPROVEMENT_MARGIN = 0.001

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def load_combined() -> pd.DataFrame:
    ref = pd.read_parquet(DATA_DIR / "reference.parquet")
    cur = pd.read_parquet(DATA_DIR / "current.parquet")
    return pd.concat([ref, cur], ignore_index=True)


def load_production_params() -> dict:
    """Pull the tuned hyperparameters from the current Production run."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    run = client.get_run(mv.run_id)
    params = run.data.params
    keep = {
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "reg_lambda", "min_child_weight",
    }
    out = {}
    for k in keep:
        if k in params:
            v = params[k]
            out[k] = int(v) if k in {"n_estimators", "max_depth", "min_child_weight"} else float(v)
    return out


def score_production(X_test, y_test) -> float:
    """Evaluate the current Production champion on the new held-out test set."""
    pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    proba = pipe.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, proba))


def main() -> None:
    df = load_combined()
    X, y, cat_cols, num_cols = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    spw = compute_scale_pos_weight(y_train)
    params = load_production_params()
    print(f"Retraining with inherited hyperparameters: {params}")

    pre = build_preprocessor(cat_cols, num_cols)
    clf = XGBClassifier(
        **params, scale_pos_weight=spw, tree_method="hist",
        eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    candidate_metrics = eval_metrics(y_test, proba)
    prod_auc = score_production(X_test, y_test)
    promoted = candidate_metrics["roc_auc"] >= prod_auc + IMPROVEMENT_MARGIN

    print(f"\nCandidate ROC-AUC: {candidate_metrics['roc_auc']:.4f}")
    print(f"Production ROC-AUC (on same test set): {prod_auc:.4f}")
    print(f"Promotion decision: {'PROMOTE' if promoted else 'REJECT'} (margin={IMPROVEMENT_MARGIN})")

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="retrain-candidate") as run:
        mlflow.log_params({**params, "model": "xgboost-retrain", "rows_train": len(X_train)})
        mlflow.log_metrics({
            **candidate_metrics,
            "production_roc_auc": prod_auc,
            "promoted": int(promoted),
        })
        sample = safe_input_example(X_train)
        mlflow.sklearn.log_model(
            pipe, artifact_path="model",
            signature=infer_signature(sample, pipe.predict_proba(sample)[:, 1]),
            input_example=sample,
        )

        client = MlflowClient()
        if promoted:
            mv = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=MODEL_NAME)
            client.update_model_version(
                name=MODEL_NAME, version=mv.version,
                description=build_model_card(run.info.run_id,
                                             {k: round(v, 4) for k, v in candidate_metrics.items()}),
            )
            client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS, version=mv.version)
            print(f"Promoted {MODEL_NAME} v{mv.version} -> alias '{ALIAS}'")
        else:
            print("Candidate did not beat production; alias unchanged.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "retrain_verdict.json").write_text(json.dumps({
        "promoted": promoted,
        "candidate_roc_auc": round(candidate_metrics["roc_auc"], 4),
        "production_roc_auc": round(prod_auc, 4),
        "improvement_margin": IMPROVEMENT_MARGIN,
        "run_id": run.info.run_id,
    }, indent=2))


if __name__ == "__main__":
    main()
