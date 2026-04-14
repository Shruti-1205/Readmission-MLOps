"""Export the registered Production model to a portable pickle for container serving.

The container image doesn't need access to the MLflow tracking server — we freeze
the sklearn Pipeline artefact at build time and load it directly in FastAPI.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "readmission-risk-champion"
ALIAS = "production"
OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    print(f"Loading {MODEL_NAME} v{mv.version} (run {mv.run_id[:8]})")

    pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    pkl_path = OUT_DIR / "champion.pkl"
    joblib.dump(pipeline, pkl_path)

    run = client.get_run(mv.run_id)
    metadata = {
        "model_name": MODEL_NAME,
        "version": mv.version,
        "run_id": mv.run_id,
        "alias": ALIAS,
        "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
    }
    meta_path = OUT_DIR / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {pkl_path} ({pkl_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
