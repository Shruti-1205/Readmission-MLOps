"""Unit tests for feature-engineering helpers."""
import numpy as np
import pandas as pd

from src.features import binarize_target, clean, engineer


def test_binarize_target_maps_only_under_30():
    s = pd.Series(["<30", ">30", "NO", "<30"])
    out = binarize_target(s)
    assert out.tolist() == [1, 0, 0, 1]


def test_clean_replaces_question_marks_and_drops_death_rows():
    df = pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4],
            "weight": ["?", "?", "?", "?"],
            "race": ["Caucasian", "?", "AfricanAmerican", "?"],
            "discharge_disposition_id": [1, 3, 11, 2],
            "readmitted": ["NO", "<30", ">30", "NO"],
        }
    )
    out = clean(df)
    assert "weight" not in out.columns
    assert "encounter_id" not in out.columns
    assert len(out) == 3
    assert out["race"].isna().sum() == 2


def test_engineer_creates_med_count_features():
    df = pd.DataFrame(
        {
            "metformin": ["Steady", "No", "Up"],
            "insulin": ["No", "Down", "Steady"],
            "change": ["No", "Ch", "Ch"],
            "diabetesMed": ["Yes", "Yes", "No"],
            "number_outpatient": [0, 1, 2],
            "number_emergency": [0, 0, 1],
            "number_inpatient": [1, 0, 0],
        }
    )
    out = engineer(df)
    assert out["num_active_meds"].tolist() == [1, 1, 2]
    assert out["num_meds_changed"].tolist() == [0, 1, 1]
    assert out["change_flag"].tolist() == [0, 1, 1]
    assert out["diabetes_med_flag"].tolist() == [1, 1, 0]
    assert out["prior_visits_total"].tolist() == [1, 1, 3]
