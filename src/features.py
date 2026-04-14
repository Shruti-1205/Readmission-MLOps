"""Preprocess the raw UCI Diabetes dataset into a modelling-ready frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

DROP_COLS = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "medical_specialty",
]

DISCHARGE_DEATH_OR_HOSPICE = {11, 13, 14, 19, 20, 21}


def binarize_target(series: pd.Series) -> pd.Series:
    """Map readmitted labels to binary: 1 if readmitted within 30 days else 0."""
    return (series == "<30").astype(int)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop high-missing cols, death/hospice rows, replace '?' with NaN."""
    df = df.replace("?", np.nan).copy()

    if "discharge_disposition_id" in df.columns:
        df = df[~df["discharge_disposition_id"].isin(DISCHARGE_DEATH_OR_HOSPICE)]

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df.reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add high-signal engineered features discussed in the project brief."""
    df = df.copy()

    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    present_med_cols = [c for c in med_cols if c in df.columns]

    med_active = df[present_med_cols].apply(lambda s: s.isin(["Steady", "Up", "Down"]))
    med_changed = df[present_med_cols].apply(lambda s: s.isin(["Up", "Down"]))

    df["num_active_meds"] = med_active.sum(axis=1).astype(int)
    df["num_meds_changed"] = med_changed.sum(axis=1).astype(int)
    df["polypharmacy_flag"] = (df["num_active_meds"] >= 10).astype(int)

    if {"number_outpatient", "number_emergency", "number_inpatient"}.issubset(df.columns):
        df["prior_visits_total"] = (
            df["number_outpatient"].fillna(0)
            + df["number_emergency"].fillna(0)
            + df["number_inpatient"].fillna(0)
        )

    if "change" in df.columns:
        df["change_flag"] = (df["change"] == "Ch").astype(int)
    if "diabetesMed" in df.columns:
        df["diabetes_med_flag"] = (df["diabetesMed"] == "Yes").astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame):
    """Return X, y ready for an sklearn pipeline. y is binary readmit-within-30."""
    y = binarize_target(df["readmitted"])
    X = df.drop(columns=["readmitted"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return X, y, cat_cols, num_cols
