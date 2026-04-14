"""Optuna hyperparameter search on the XGBoost champion, logged to MLflow."""
from __future__ import annotations

import mlflow
import mlflow.sklearn
import optuna
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

EXPERIMENT_NAME = "readmission-tuning"
N_TRIALS = 25
N_FOLDS = 3


def main() -> None:
    X_train, X_test, y_train, y_test, cat_cols, num_cols = load_splits()
    spw = compute_scale_pos_weight(y_train)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_experiment(EXPERIMENT_NAME)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        fold_scores = []
        for fold_idx, (tr, va) in enumerate(skf.split(X_train, y_train)):
            pre = build_preprocessor(cat_cols, num_cols)
            clf = XGBClassifier(
                **params, scale_pos_weight=spw, tree_method="hist",
                eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1,
            )
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
            proba = pipe.predict_proba(X_train.iloc[va])[:, 1]
            fold_scores.append(roc_auc_score(y_train.iloc[va], proba))

        return sum(fold_scores) / len(fold_scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest CV ROC-AUC: {study.best_value:.4f}")
    print("Best params:", study.best_params)

    with mlflow.start_run(run_name="xgb-tuned") as run:
        pre = build_preprocessor(cat_cols, num_cols)
        clf = XGBClassifier(
            **study.best_params, scale_pos_weight=spw, tree_method="hist",
            eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1,
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = eval_metrics(y_test, proba)
        metrics["cv_roc_auc"] = float(study.best_value)

        mlflow.log_params({**study.best_params, "model": "xgboost-tuned", "n_trials": N_TRIALS})
        mlflow.log_metrics(metrics)

        sample = safe_input_example(X_train, n=5)
        signature = infer_signature(sample, pipe.predict_proba(sample)[:, 1])
        mlflow.sklearn.log_model(
            pipe, artifact_path="model",
            signature=signature, input_example=sample,
        )

        print(f"\nTest metrics: {metrics}")
        print(f"MLflow run_id: {run.info.run_id}")
        print("\nTo register this model, run:")
        print(f"  python -m src.register {run.info.run_id}")


if __name__ == "__main__":
    main()
