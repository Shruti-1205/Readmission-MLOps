"""Pydantic request/response models for the readmission risk API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    """One encounter, matching UCI Diabetes 130-US schema.

    Declares the core clinical fields with validation; accepts the remaining
    optional medication columns via `extra='allow'` so we don't need 40+ field defs.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    race: Optional[str] = None
    gender: str = Field(..., examples=["Male", "Female"])
    age: str = Field(..., examples=["[70-80)"], description="Age bracket, 10-year bins")

    admission_type_id: int = Field(..., ge=1, le=8)
    discharge_disposition_id: int = Field(..., ge=1, le=30)
    admission_source_id: int = Field(..., ge=1, le=30)

    time_in_hospital: int = Field(..., ge=1, le=14)
    num_lab_procedures: int = Field(..., ge=0, le=200)
    num_procedures: int = Field(..., ge=0, le=10)
    num_medications: int = Field(..., ge=0, le=100)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    number_diagnoses: int = Field(..., ge=1, le=20)

    diag_1: Optional[str] = None
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None

    max_glu_serum: Optional[str] = None
    A1Cresult: Optional[str] = None

    insulin: Optional[str] = "No"
    metformin: Optional[str] = "No"
    change: Optional[str] = "No"
    diabetesMed: Optional[str] = "No"


class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    readmit_probability: float = Field(..., ge=0.0, le=1.0)
    risk_category: Literal["low", "medium", "high"]
    top_features: list[FeatureAttribution]
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    version: str
    run_id: str
    alias: str
    metrics: dict
