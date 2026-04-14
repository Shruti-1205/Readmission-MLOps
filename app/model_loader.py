"""Load the exported champion pipeline + warm a SHAP explainer at startup."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import shap

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "champion.pkl"
DEFAULT_META_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model_metadata.json"


@dataclass
class LoadedModel:
    pipeline: object
    feature_names: list[str]
    explainer: shap.TreeExplainer
    metadata: dict


def load() -> LoadedModel:
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    meta_path = Path(os.getenv("MODEL_METADATA_PATH", DEFAULT_META_PATH))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model pickle not found at {model_path}. "
            "Run `python -m src.export_model` first."
        )

    pipeline = joblib.load(model_path)
    feature_names = list(pipeline.named_steps["pre"].get_feature_names_out())
    explainer = shap.TreeExplainer(pipeline.named_steps["clf"])

    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    return LoadedModel(
        pipeline=pipeline,
        feature_names=feature_names,
        explainer=explainer,
        metadata=metadata,
    )


def risk_category(prob: float) -> str:
    if prob >= 0.30:
        return "high"
    if prob >= 0.15:
        return "medium"
    return "low"


def top_shap_features(
    shap_values: np.ndarray, feature_names: list[str], k: int = 5
) -> list[dict]:
    """Return the k features with largest |SHAP value| for one prediction."""
    flat = np.asarray(shap_values).ravel()
    order = np.argsort(np.abs(flat))[::-1][:k]
    return [
        {
            "feature": feature_names[i],
            "shap_value": float(flat[i]),
            "direction": "increases_risk" if flat[i] > 0 else "decreases_risk",
        }
        for i in order
    ]
