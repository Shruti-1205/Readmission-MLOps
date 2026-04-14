"""FastAPI service for 30-day readmission risk prediction."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import LoadedModel, load, risk_category, top_shap_features
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PatientRecord,
    PredictionResponse,
)
from src.features import engineer

EXTRA_MED_COLS = [
    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def prepare_inference_df(payload: dict) -> pd.DataFrame:
    """Turn a PatientRecord payload into the column layout the pipeline expects.

    The pipeline was fit on clean+engineered data (~54 cols). The pydantic schema
    exposes only the core clinical fields (~23), so we fill defaults for the
    remaining medication columns and run the same feature engineering the
    training pipeline saw.
    """
    df = pd.DataFrame([payload])
    for col in EXTRA_MED_COLS:
        if col not in df.columns:
            df[col] = "No"
    return engineer(df)

state: dict[str, LoadedModel | None] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load()
    yield
    state["model"] = None


app = FastAPI(
    title="Readmission Risk API",
    description="Predicts 30-day hospital readmission risk for diabetic patients.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    m = state["model"]
    if m is None:
        return HealthResponse(status="ok", model_loaded=False)
    meta = m.metadata or {}
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=meta.get("model_name"),
        model_version=str(meta.get("version", "")),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    m = state["model"]
    if m is None or not m.metadata:
        raise HTTPException(status_code=503, detail="Model metadata unavailable")
    meta = m.metadata
    return ModelInfoResponse(
        model_name=meta["model_name"],
        version=str(meta["version"]),
        run_id=meta["run_id"],
        alias=meta["alias"],
        metrics=meta.get("metrics", {}),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord) -> PredictionResponse:
    m = state["model"]
    if m is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    payload = record.model_dump(by_alias=True)
    df = prepare_inference_df(payload)

    proba = float(m.pipeline.predict_proba(df)[0, 1])

    X_tx = m.pipeline.named_steps["pre"].transform(df)
    if hasattr(X_tx, "toarray"):
        X_tx = X_tx.toarray()
    shap_values = m.explainer.shap_values(X_tx)[0]
    top = top_shap_features(shap_values, m.feature_names, k=5)

    meta = m.metadata or {}
    return PredictionResponse(
        readmit_probability=proba,
        risk_category=risk_category(proba),
        top_features=top,
        model_name=meta.get("model_name", "unknown"),
        model_version=str(meta.get("version", "unknown")),
    )
