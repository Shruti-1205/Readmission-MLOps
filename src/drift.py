"""Run an Evidently DataDriftPreset on reference vs current and emit a CI-gated verdict."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REF_PATH = DATA_DIR / "reference.parquet"
CUR_PATH = DATA_DIR / "current.parquet"

DRIFT_SHARE_THRESHOLD = 0.30
INDIVIDUAL_SCORE_THRESHOLD = 0.20
DROP_COLS = ["readmitted"]


def _extract_metric(as_dict: dict, metric_name: str) -> dict | None:
    for m in as_dict.get("metrics", []):
        if m.get("metric") == metric_name:
            return m.get("result")
    return None


def main() -> int:
    if not REF_PATH.exists() or not CUR_PATH.exists():
        print("Reference/current not found. Run `python -m src.simulate_drift` first.")
        return 2

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(REF_PATH)
    current = pd.read_parquet(CUR_PATH)
    for col in DROP_COLS:
        reference = reference.drop(columns=col, errors="ignore")
        current = current.drop(columns=col, errors="ignore")

    print(f"Reference: {reference.shape}, Current: {current.shape}")
    print("Running Evidently DataDriftPreset...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    html_path = REPORTS_DIR / "drift_report.html"
    report.save_html(str(html_path))

    as_dict = report.as_dict()
    dataset_metric = _extract_metric(as_dict, "DatasetDriftMetric") or {}
    table_metric = _extract_metric(as_dict, "DataDriftTable") or {}

    drifted_cols = []
    for col, info in (table_metric.get("drift_by_columns") or {}).items():
        if info.get("drift_detected"):
            drifted_cols.append({
                "feature": col,
                "stattest": info.get("stattest_name"),
                "drift_score": round(float(info.get("drift_score", 0.0)), 4),
            })

    drift_share = float(dataset_metric.get("share_of_drifted_columns", 0.0))
    dataset_drift = bool(dataset_metric.get("dataset_drift", False))
    any_above_thresh = any(
        c["drift_score"] >= INDIVIDUAL_SCORE_THRESHOLD for c in drifted_cols
    )

    should_retrain = dataset_drift or drift_share >= DRIFT_SHARE_THRESHOLD or any_above_thresh

    verdict = {
        "dataset_drift": dataset_drift,
        "drift_share": round(drift_share, 4),
        "drifted_column_count": len(drifted_cols),
        "drifted_columns": sorted(drifted_cols, key=lambda c: -c["drift_score"])[:10],
        "thresholds": {
            "drift_share": DRIFT_SHARE_THRESHOLD,
            "individual_score": INDIVIDUAL_SCORE_THRESHOLD,
        },
        "should_retrain": should_retrain,
    }
    verdict_path = REPORTS_DIR / "drift_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2))

    print("\n=== Drift verdict ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nHTML report: {html_path}")
    print(f"JSON verdict: {verdict_path}")
    return 1 if should_retrain else 0


if __name__ == "__main__":
    sys.exit(main())
