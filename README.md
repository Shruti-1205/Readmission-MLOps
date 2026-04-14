# Readmission Risk MLOps

A hospital readmission risk model for diabetic patients, wrapped in the sort of operational scaffolding you would actually run in production: experiment tracking, a containerised API, automated drift checks, a promote-if-better retraining gate, a subgroup fairness audit, and a small Streamlit dashboard on top.

Built on the UCI Diabetes 130-US dataset (101,766 inpatient encounters, 50 clinical features, CC-BY). The model predicts whether a discharged patient will be readmitted within 30 days.

## What's in the box

Training and modeling:

- Five candidate algorithms compared on the same split (Logistic Regression, Random Forest, XGBoost, LightGBM, MLP).
- Optuna TPE hyperparameter search (25 trials, 3-fold stratified CV) on the winning model.
- SHAP explainability (global bar and beeswarm, per-patient force plots).
- SMOTE vs. class-weighting ablation, with the winning strategy justified.

Operational pieces:

- MLflow for experiment tracking and a Model Registry with a `production` alias.
- FastAPI service with Pydantic v2 validation, a `/predict` endpoint that returns SHAP attributions, plus `/health` and `/model-info`. OpenAPI docs at `/docs`.
- Multi-stage Docker build, non-root user, built-in HEALTHCHECK.
- Pytest: feature-engineering unit tests plus integration tests through the FastAPI TestClient.

Monitoring and retraining:

- Evidently data drift report across all 50 features, producing an HTML artifact and a JSON verdict the CI can read.
- Automated retraining script that inherits the Production hyperparameters, evaluates the candidate against the current Production model on a shared test set, and only promotes if ROC-AUC improves by a safety margin.
- Subgroup fairness audit covering ROC-AUC, FPR, and FNR per race and gender.

Presentation:

- Streamlit dashboard with a home page, patient scorer, performance view, fairness view, and drift monitor.
- GitHub Actions workflows for lint, test, model quality gate, Docker build, and a scheduled drift check + retrain.

## Results on the held-out test set

| Metric                     | Value  |
| -------------------------- | ------ |
| ROC-AUC                    | 0.6776 |
| PR-AUC                     | 0.2344 |
| F1 at 0.5                  | 0.2785 |
| Base rate (positive class) | 11.4%  |
| PR-AUC lift over base rate | 2.06x  |

F1 looks low because the default 0.5 threshold is poorly matched to an 11% positive class. The `src.threshold` script sweeps the decision threshold and writes a tuned operating point that you can plug into the API.

## Running it locally

Python 3.10, a virtual environment, and about 4 GB of disk.

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Unix:     source venv/bin/activate
pip install -r requirements.txt
```

### Train the champion

```bash
python -m src.data_loader        # downloads and caches UCI dataset
python -m src.train_sweep        # 5-model comparison
python -m src.tune_champion      # Optuna on the winner
python -m src.register <run_id>  # register in MLflow as Production
```

### Serve it

```bash
python -m src.export_model                     # freezes champion to artifacts/champion.pkl
pytest -v                                      # 8 tests
uvicorn app.main:app --reload                  # http://127.0.0.1:8000/docs
docker build -t readmission-api:1.0 .
docker run --rm -p 8000:8000 readmission-api:1.0
```

### Drift, retrain, fairness, threshold

```bash
python -m src.simulate_drift   # produces reference and drifted current parquet files
python -m src.drift            # Evidently report and verdict JSON
python -m src.retrain          # retrains if drift detected, promotes only if better
python -m src.threshold        # picks operating point
python -m src.fairness         # subgroup audit
python -m src.train_smote      # SMOTE ablation logged alongside baselines
```

### Dashboard

```bash
streamlit run streamlit_app.py   # http://127.0.0.1:8501
```

## Repo layout

```text
src/                 training, features, drift, retrain, fairness, threshold
app/                 FastAPI service (main.py, schemas.py, model_loader.py)
tests/               pytest unit and integration tests
streamlit_app.py     multi-page dashboard
Dockerfile           multi-stage build, non-root user, HEALTHCHECK
.github/workflows/   CI (lint, test, gate, build) and weekly drift + retrain
artifacts/           exported champion.pkl, threshold.json, metadata
reports/             drift and fairness verdicts, Evidently HTML report
docs/screenshots/    SHAP plots, threshold sweep, fairness chart, UI captures
```

## Notes and caveats

The model sits around 0.67 to 0.68 ROC-AUC because much of what drives readmission (medication adherence, post-discharge care, social support) is not in the dataset. Grinding for another point of AUC was not the goal; the goal was end-to-end operational plumbing.

SMOTE was tested against `scale_pos_weight`. They matched on ROC-AUC but SMOTE took 7.5x longer to train and broke probability calibration (F1 fell from 0.28 to 0.03 at a 0.5 threshold). Class-weighting stayed.

The retrain safety gate blocked a drift-triggered candidate from replacing Production because the candidate did not outperform on a shared test set. That is the intended behaviour; a noisy drift event should not silently regress a working model.

The fairness audit found a 0.14 gap in ROC-AUC between the best and worst race subgroups, concentrated in the smallest and most underrepresented groups. A real deployment response would be targeted data collection for those groups rather than a model patch.

## License

MIT.
