"""Integration tests for the FastAPI service using TestClient."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "champion.pkl"
pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="Run `python -m src.export_model` to produce artifacts/champion.pkl first.",
)


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 5,
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 13,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 9,
    "diag_1": "428",
    "diag_2": "427",
    "diag_3": "250.02",
    "max_glu_serum": None,
    "A1Cresult": ">8",
    "insulin": "Up",
    "metformin": "No",
    "change": "Ch",
    "diabetesMed": "Yes",
}


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "readmission-risk-champion"
    assert "metrics" in body


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["readmit_probability"] <= 1.0
    assert body["risk_category"] in {"low", "medium", "high"}
    assert len(body["top_features"]) == 5
    for feat in body["top_features"]:
        assert feat["direction"] in {"increases_risk", "decreases_risk"}


def test_predict_invalid_payload_returns_422(client):
    bad = dict(VALID_PAYLOAD)
    bad["time_in_hospital"] = 99
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_required_field_returns_422(client):
    bad = dict(VALID_PAYLOAD)
    bad.pop("gender")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
