# Patient Readmission Risk — MLOps Platform

Production-grade ML platform that predicts 30-day hospital readmission risk for diabetic patients at discharge. Trained on **101,766 real clinical records from 130 US hospitals** (UCI Diabetes 130-US). Includes experiment tracking, hyperparameter tuning, SHAP explainability, a FastAPI inference service, Docker packaging, automated data-drift monitoring, promote-if-better retraining, a subgroup fairness audit, a Streamlit demo dashboard, and a GitHub Actions CI/CD pipeline.

## Why this problem matters

30-day hospital readmissions cost the US healthcare system **$26B per year**. Under the CMS Hospital Readmissions Reduction Program, hospitals can lose up to **3% of Medicare payments** for excess readmissions. Predicting who is at risk at discharge — and explaining *why* — enables targeted interventions (closer follow-up, medication reconciliation, home-care coordination) that reduce both harm and cost.

## Key results

| Metric | Value | Literature benchmark |
|---|---|---|
| Test ROC-AUC (tuned XGBoost) | **0.6776** | Strack 2014 (LogReg): 0.65, Rajendra 2021 (NN): 0.67, LACE index: 0.65–0.70 |
| Test PR-AUC | 0.234 | Base rate 0.114 → 2× lift |
| Best operating threshold | 0.42 (cost-weighted, FN = 5× FP) | |
| Docker image | 2.06 GB | |
| Unit + integration tests | 8 / 8 passing | |

## Architecture

```
 ┌────────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌────────────────┐
 │  UCI Diabetes  │────▶│  Feature    │────▶│  Model training │────▶│  MLflow        │
 │  130-US (HF)   │     │  store      │     │  (5-model sweep)│     │  tracking +    │
 └────────────────┘     │ clean +     │     │  Optuna tuning  │     │  registry      │
                        │ engineer    │     │  SHAP explainer │     │  (Prod alias)  │
                        └─────────────┘     └─────────────────┘     └────────┬───────┘
                                                                             │
                                                                             ▼
 ┌────────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌────────────────┐
 │  Streamlit     │◀────│  Docker     │◀────│  FastAPI app    │◀────│  Exported      │
 │  dashboard     │     │  image      │     │  (Pydantic,     │     │  pipeline.pkl  │
 │  (HF Spaces)   │     │  (non-root) │     │   /predict,     │     │                │
 └────────────────┘     └─────────────┘     │   /health,      │     └────────────────┘
                                            │   /model-info)  │
                                            └─────────────────┘

 ┌────────────────┐     ┌─────────────┐     ┌─────────────────┐
 │  Evidently     │────▶│  Drift      │────▶│  Retrain +      │
 │  drift report  │     │  verdict    │     │  promote-if-    │
 │  (weekly CI)   │     │  JSON       │     │  better gate    │
 └────────────────┘     └─────────────┘     └─────────────────┘
```

## Stack

All components are free and self-hostable.

- **Data**: UCI Diabetes 130-US Hospitals (101,766 encounters, 50 features, CC-BY)
- **Modeling**: scikit-learn, XGBoost, LightGBM, Logistic Regression, MLP
- **Tuning**: Optuna TPE sampler, 25 trials × 3-fold stratified CV
- **Tracking**: MLflow (experiments + Model Registry with alias-based promotion)
- **Explainability**: SHAP (TreeExplainer; global bar / beeswarm / per-patient force plots)
- **Serving**: FastAPI + Pydantic v2 schema validation, auto-generated OpenAPI docs
- **Container**: multi-stage Dockerfile, non-root user, built-in HEALTHCHECK
- **Drift**: Evidently AI `DataDriftPreset` with custom thresholds (PSI / Wasserstein / Jensen-Shannon)
- **Fairness**: per-race / per-gender AUC, FPR, FNR, gap analysis
- **Dashboard**: Streamlit (Predict / Performance / Fairness / Drift views)
- **CI/CD**: GitHub Actions (lint, test, model quality gate, Docker build → GHCR)
- **Tests**: pytest (3 feature unit tests + 5 API integration tests)

## Quick start (Windows / Linux / macOS)

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Unix:     source venv/bin/activate
pip install -r requirements.txt

# Phase 1-2 — train + register the champion
python -m src.data_loader
python -m src.train_sweep
python -m src.tune_champion
python -m src.register <run_id_printed_above>

# Phase 3 — export and serve
python -m src.export_model
pytest -v                                         # 8 tests
uvicorn app.main:app --reload                     # http://127.0.0.1:8000/docs
docker build -t readmission-api:1.0 .
docker run --rm -p 8000:8000 readmission-api:1.0

# Phase 4 — drift + retrain safety gate
python -m src.simulate_drift
python -m src.drift
python -m src.retrain

# Phase 4.5 — threshold, SMOTE ablation, fairness
python -m src.threshold
python -m src.train_smote
python -m src.fairness

# Phase 5 — Streamlit demo
streamlit run streamlit_app.py                    # http://127.0.0.1:8501
```

## Repository map

```
├── src/                       model training, features, drift, retrain, fairness
├── app/                       FastAPI service (main.py, schemas.py, model_loader.py)
├── tests/                     pytest unit + integration tests
├── streamlit_app.py           multi-page dashboard
├── Dockerfile                 multi-stage build, non-root user, HEALTHCHECK
├── .github/workflows/
│   ├── ci.yml                 lint + tests + model gate + Docker build/push
│   └── drift-retrain.yml      weekly drift check + conditional retrain
├── artifacts/                 exported champion.pkl, threshold.json, metadata
├── reports/                   drift + fairness + retrain JSON reports
└── docs/screenshots/          SHAP plots, threshold sweep, fairness chart, MLflow UI
```

## Honest notes on results

- ROC-AUC is **ceilinged at ~0.68–0.70** on this dataset because key drivers of readmission (medication adherence, socioeconomic factors, post-discharge care) aren't in the data. Matching the literature ceiling is the right bar; grinding for +0.01 AUC isn't the point.
- SMOTE was tested against `scale_pos_weight`; it matched AUC but **broke probability calibration** (F1 dropped from 0.28 to 0.03 at threshold 0.5) and cost 7.5× training time. Class-weighting is the right choice for this tree model.
- The **promote-if-better gate rejected** a drift-triggered retrain during testing because the candidate did not beat production on a shared holdout. The safety gate working is a feature, not a bug.
- Fairness audit found a meaningful **ROC-AUC gap of 0.14 across race subgroups**, concentrated in the smallest underrepresented groups (n=304 "Other", n=410 "Hispanic"). The right production response is more data collection for those groups, not a model patch.
