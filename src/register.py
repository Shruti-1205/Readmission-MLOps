"""Register a trained MLflow run as the Production champion with a model card."""
from __future__ import annotations

import sys
from textwrap import dedent

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "readmission-risk-champion"


def build_model_card(run_id: str, metrics: dict) -> str:
    return dedent(
        f"""
        # Readmission Risk Champion — Model Card

        **Source run:** {run_id}

        ## Intended use
        Predicts probability of 30-day hospital readmission for diabetic patients at
        discharge. For decision *support* only — must not be used as sole basis for
        clinical action.

        ## Training data
        UCI Diabetes 130-US Hospitals, 1999-2008 (101,766 encounters, 50 features).
        Encounters ending in death or hospice are excluded. Binary target: readmit within 30 days.

        ## Metrics (held-out test set, 20%)
        - ROC-AUC: {metrics.get('roc_auc', 'n/a')}
        - PR-AUC:  {metrics.get('pr_auc', 'n/a')}
        - F1 @ 0.5: {metrics.get('f1', 'n/a')}

        ## Known limitations
        - Data is from 130 US hospitals, 1999-2008 — distribution shift likely for current populations.
        - Class imbalance (~11% positive); threshold tuning recommended per deployment.
        - No ethnicity fairness audit performed yet — flagged for Phase 4.
        """
    ).strip()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.register <run_id>")
        sys.exit(1)
    run_id = sys.argv[1]

    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {mv.version}")

    client.update_model_version(
        name=MODEL_NAME, version=mv.version,
        description=build_model_card(run_id, metrics),
    )
    client.set_model_version_tag(name=MODEL_NAME, version=mv.version, key="stage", value="Production")
    client.set_registered_model_alias(name=MODEL_NAME, alias="production", version=mv.version)

    print(f"Set alias 'production' -> {MODEL_NAME} v{mv.version}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
