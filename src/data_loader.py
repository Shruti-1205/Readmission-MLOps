"""Load and cache the UCI Diabetes 130-US Hospitals dataset."""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_FILE = DATA_DIR / "diabetes_130us.parquet"


def load_raw(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the UCI dataset (id=296), cache to parquet, return combined frame."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and not force_refresh:
        return pd.read_parquet(RAW_FILE)

    print("Downloading UCI Diabetes 130-US dataset (id=296)...")
    repo = fetch_ucirepo(id=296)
    X = repo.data.features
    y = repo.data.targets
    df = pd.concat([X, y], axis=1)
    df.to_parquet(RAW_FILE, index=False)
    print(f"Saved {len(df):,} rows, {df.shape[1]} columns to {RAW_FILE}")
    return df


if __name__ == "__main__":
    df = load_raw()
    print("\nShape:", df.shape)
    print("\nTarget distribution (readmitted):")
    print(df["readmitted"].value_counts())
    print("\nColumn dtypes:")
    print(df.dtypes.value_counts())
    print("\nMissing-value columns (showing '?' as missing):")
    missing = (df == "?").sum()
    print(missing[missing > 0].sort_values(ascending=False))
