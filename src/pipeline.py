"""Shared data/preprocessing helpers used by training, tuning, and explainability scripts."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_raw
from src.features import build_feature_matrix, clean, engineer

RANDOM_STATE = 42


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list, list]:
    """Return train/test split and column lists. Same seed = comparable runs."""
    df = load_raw()
    df = clean(df)
    df = engineer(df)
    X, y, cat_cols, num_cols = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, cat_cols, num_cols


def build_preprocessor(cat_cols: list, num_cols: list, scale_numeric: bool = False) -> ColumnTransformer:
    """OHE for cats, median-impute for nums. Optionally scale numerics (LogReg/MLP)."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def eval_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, preds)),
    }


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    pos_rate = float(y_train.mean())
    return (1 - pos_rate) / pos_rate


def safe_input_example(X: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return n rows suitable for MLflow signature inference.

    Two fixups so the inferred schema is robust at serve time:
    1. If a column is entirely null in the sample, substitute the first non-null
       value from the full frame (otherwise MLflow can't infer the column type).
    2. Cast integer columns to float64 so the schema tolerates NaN at inference —
       Python ints can't represent missing, and pandas silently upcasts, which
       triggers MLflow schema enforcement errors in FastAPI serving.
    """
    sample = X.head(n).copy()
    for col in sample.columns:
        if sample[col].isna().all():
            non_null = X[col].dropna()
            if len(non_null) > 0:
                sample[col] = non_null.iloc[0]

    int_cols = sample.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns
    if len(int_cols) > 0:
        sample[int_cols] = sample[int_cols].astype("float64")
    return sample
