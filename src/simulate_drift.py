"""Split the UCI dataset into reference (training) and current (drifted) slices.

Simulates a real-world drift scenario without waiting weeks for production data.
The `current` slice is perturbed in plausible ways (aging population, higher
polypharmacy, more aggressive insulin regimens) so Evidently will detect drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_raw
from src.features import clean, engineer

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REFERENCE_FRACTION = 0.70
CURRENT_FRACTION = 0.20
RANDOM_STATE = 42


def perturb_current(df: pd.DataFrame) -> pd.DataFrame:
    """Apply realistic, drift-inducing edits to the current slice."""
    df = df.copy()
    rng = np.random.default_rng(RANDOM_STATE)

    if "age" in df.columns:
        young_mask = df["age"].isin(["[0-10)", "[10-20)", "[20-30)", "[30-40)"])
        young_idx = df.index[young_mask]
        reassigned = rng.choice(["[70-80)", "[80-90)", "[90-100)"], size=len(young_idx))
        df.loc[young_idx, "age"] = reassigned

    if "num_medications" in df.columns:
        df["num_medications"] = (df["num_medications"] + 3).clip(upper=100)

    if "num_lab_procedures" in df.columns:
        df["num_lab_procedures"] = (df["num_lab_procedures"] * 1.2).round().clip(upper=200).astype(int)

    if "insulin" in df.columns:
        no_idx = df.index[df["insulin"] == "No"]
        flip_n = int(len(no_idx) * 0.30)
        flip_idx = rng.choice(no_idx, size=flip_n, replace=False)
        df.loc[flip_idx, "insulin"] = "Up"

    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    df = clean(df)
    df = engineer(df)

    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    n = len(df)
    n_ref = int(n * REFERENCE_FRACTION)
    n_cur = int(n * CURRENT_FRACTION)

    reference = df.iloc[:n_ref].copy()
    current = perturb_current(df.iloc[n_ref : n_ref + n_cur])

    ref_path = OUT_DIR / "reference.parquet"
    cur_path = OUT_DIR / "current.parquet"
    reference.to_parquet(ref_path, index=False)
    current.to_parquet(cur_path, index=False)

    print(f"Wrote {ref_path} ({len(reference):,} rows)")
    print(f"Wrote {cur_path} ({len(current):,} rows)")
    print("\nDrift injected on: age (younger → older), num_medications (+3), "
          "num_lab_procedures (×1.2), insulin (30% No → Up)")


if __name__ == "__main__":
    main()
