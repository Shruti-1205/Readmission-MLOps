"""Streamlit dashboard for the Readmission Risk MLOps platform.

Sidebar navigation has five views:
  Home              high-level overview of the platform
  Predict risk      single-patient prediction with SHAP explanation
  Model performance headline metrics, threshold sweep, confusion matrices
  Fairness audit    per-subgroup metrics and gap analysis
  Drift monitoring  latest drift check and retrain decision
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
SCREENSHOTS = ROOT / "docs" / "screenshots"

EXTRA_MED_COLS = [
    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

ALGORITHM_LABELS = {
    "XGBClassifier": "XGBoost (gradient boosting)",
    "LGBMClassifier": "LightGBM (gradient boosting)",
    "RandomForestClassifier": "Random Forest",
    "LogisticRegression": "Logistic Regression",
    "MLPClassifier": "Neural Network (MLP)",
}

st.set_page_config(page_title="Readmission Risk", layout="wide")

STYLE = """
<style>
.card {
    background: #f7f9fc;
    border: 1px solid #e3e8ef;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.card .label {
    color: #5b6b80;
    font-size: 0.82rem;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.card .value {
    color: #1f2a44;
    font-size: 1.35rem;
    font-weight: 600;
    line-height: 1.3;
}
.card .hint {
    color: #6b7891;
    font-size: 0.8rem;
    margin-top: 4px;
}
.section-intro {
    color: #4a5568;
    background: #f2f5fa;
    border-left: 3px solid #7c93b7;
    padding: 10px 14px;
    border-radius: 4px;
    margin-bottom: 16px;
    font-size: 0.9rem;
}
.hero {
    background: linear-gradient(135deg, #f7f9fc 0%, #eef2f8 100%);
    border: 1px solid #e3e8ef;
    border-radius: 10px;
    padding: 28px 30px;
    margin-bottom: 18px;
}
.hero h1 {
    color: #1f2a44;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 6px 0;
}
.hero p {
    color: #4a5568;
    font-size: 0.98rem;
    margin: 0;
    max-width: 780px;
}
.feature-tile {
    background: #ffffff;
    border: 1px solid #e3e8ef;
    border-radius: 8px;
    padding: 16px 18px;
    height: 100%;
}
.feature-tile h4 {
    color: #1f2a44;
    font-size: 1rem;
    margin: 0 0 6px 0;
}
.feature-tile p {
    color: #5b6b80;
    font-size: 0.88rem;
    margin: 0;
}
.footnote {
    color: #6b7891;
    font-size: 0.82rem;
    padding-top: 8px;
    border-top: 1px solid #e3e8ef;
    margin-top: 24px;
}
</style>
"""


def card(label: str, value: str, hint: str | None = None) -> str:
    hint_html = f"<div class='hint'>{hint}</div>" if hint else ""
    return (
        f"<div class='card'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>{hint_html}</div>"
    )


def feature_tile(title: str, body: str) -> str:
    return f"<div class='feature-tile'><h4>{title}</h4><p>{body}</p></div>"


@st.cache_resource
def load_assets():
    pipeline = joblib.load(ARTIFACTS / "champion.pkl")
    metadata = json.loads((ARTIFACTS / "model_metadata.json").read_text())
    threshold_path = ARTIFACTS / "threshold.json"
    thresholds = json.loads(threshold_path.read_text()) if threshold_path.exists() else {}
    feature_names = list(pipeline.named_steps["pre"].get_feature_names_out())
    explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
    clf_class = type(pipeline.named_steps["clf"]).__name__
    algorithm = ALGORITHM_LABELS.get(clf_class, clf_class)
    return pipeline, metadata, thresholds, feature_names, explainer, algorithm


def sidebar_header(algorithm: str) -> None:
    st.sidebar.title("Readmission Risk")
    st.sidebar.caption("Clinical decision support prototype.")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Algorithm:** {algorithm}")
    st.sidebar.markdown("**Dataset:** UCI Diabetes 130-US")
    st.sidebar.markdown("**Records:** 101,766 encounters")


def page_home(metadata: dict, algorithm: str) -> None:
    metrics = metadata.get("metrics", {})
    st.markdown(
        "<div class='hero'>"
        "<h1>Readmission Risk Intelligence</h1>"
        "<p>Predicts the probability of a diabetic patient being readmitted "
        "within 30 days of discharge. Use the sidebar to score a patient, review "
        "model performance, check subgroup fairness, or inspect the data drift "
        "monitor.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Training encounters", "101,766"), unsafe_allow_html=True)
    c2.markdown(card("Features", "50 clinical variables"), unsafe_allow_html=True)
    c3.markdown(card("Algorithm", algorithm), unsafe_allow_html=True)
    c4.markdown(
        card("Test ROC-AUC", str(metrics.get("roc_auc", "n/a")), "held-out 20% sample"),
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("What this platform does")
    r1c1, r1c2 = st.columns(2)
    r1c1.markdown(
        feature_tile(
            "Patient risk prediction",
            "Enter a patient's admission, diagnosis, and medication details. Get a "
            "calibrated probability, a clinical decision flag, and the top features "
            "driving the prediction.",
        ),
        unsafe_allow_html=True,
    )
    r1c2.markdown(
        feature_tile(
            "Model performance",
            "Headline metrics from the held-out test set, a precision-recall "
            "tradeoff curve across decision thresholds, and confusion matrices at "
            "both the default and tuned operating points.",
        ),
        unsafe_allow_html=True,
    )

    r2c1, r2c2 = st.columns(2)
    r2c1.markdown(
        feature_tile(
            "Subgroup fairness",
            "ROC-AUC, false-positive rate, and false-negative rate broken out by "
            "race and gender. Identifies subgroups where the model under-performs "
            "so they can be prioritised for data collection.",
        ),
        unsafe_allow_html=True,
    )
    r2c2.markdown(
        feature_tile(
            "Drift monitoring",
            "Compares live data distributions against the training reference. "
            "When drift is detected, an automated retrain runs and only replaces "
            "the production model if it passes a held-out quality gate.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='footnote'>Predictions are decision support only. They "
        "are not a substitute for clinical judgement.</div>",
        unsafe_allow_html=True,
    )


def page_predict(pipeline, feature_names, explainer, thresholds) -> None:
    st.header("Predict readmission risk")
    st.markdown(
        "<div class='section-intro'>Enter a patient's clinical details and click "
        "Predict. You will receive the readmission probability, a decision flag at "
        "the calibrated clinical threshold, and the ten features contributing most "
        "to the prediction.</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox(
            "Age bracket",
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
            index=7,
        )
        admission_type_id = st.number_input("Admission type id", 1, 8, 1)
        discharge_disposition_id = st.number_input("Discharge disposition id", 1, 30, 1)
        admission_source_id = st.number_input("Admission source id", 1, 30, 7)

    with col2:
        time_in_hospital = st.slider("Days in hospital", 1, 14, 5)
        num_lab_procedures = st.slider("Lab procedures", 0, 200, 41)
        num_procedures = st.slider("Medical procedures", 0, 10, 0)
        num_medications = st.slider("Medications", 0, 100, 13)
        number_outpatient = st.number_input("Prior outpatient visits (1 yr)", 0, 50, 0)
        number_emergency = st.number_input("Prior emergency visits (1 yr)", 0, 50, 0)
        number_inpatient = st.number_input("Prior inpatient visits (1 yr)", 0, 50, 1)
        number_diagnoses = st.slider("Number of diagnoses", 1, 20, 9)

    with col3:
        diag_1 = st.text_input("Primary diagnosis (ICD-9)", "428")
        diag_2 = st.text_input("Secondary diagnosis", "427")
        diag_3 = st.text_input("Tertiary diagnosis", "250.02")
        max_glu_serum = st.selectbox("Glucose serum", ["None", "Norm", ">200", ">300"], index=0)
        A1Cresult = st.selectbox("A1C result", ["None", "Norm", ">7", ">8"], index=3)
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"], index=2)
        metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"], index=0)
        change = st.selectbox("Med change this visit", ["No", "Ch"], index=1)
        diabetesMed = st.selectbox("Diabetes medication prescribed", ["No", "Yes"], index=1)

    st.write("")
    if st.button("Predict", type="secondary"):
        payload = {
            "race": race, "gender": gender, "age": age,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures, "num_medications": num_medications,
            "number_outpatient": number_outpatient, "number_emergency": number_emergency,
            "number_inpatient": number_inpatient, "number_diagnoses": number_diagnoses,
            "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
            "max_glu_serum": None if max_glu_serum == "None" else max_glu_serum,
            "A1Cresult": None if A1Cresult == "None" else A1Cresult,
            "insulin": insulin, "metformin": metformin,
            "change": change, "diabetesMed": diabetesMed,
        }
        from src.features import engineer
        df = pd.DataFrame([payload])
        for col in EXTRA_MED_COLS:
            df[col] = "No"
        df = engineer(df)

        proba = float(pipeline.predict_proba(df)[0, 1])
        thr = thresholds.get("best_threshold_cost_weighted", 0.42)
        decision = "High risk" if proba >= thr else "Low risk"
        tier = "High" if proba >= 0.30 else "Medium" if proba >= 0.15 else "Low"

        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.markdown(card("Readmit probability", f"{proba:.1%}"), unsafe_allow_html=True)
        c2.markdown(
            card("Clinical decision", decision, f"threshold = {thr:.2f}"),
            unsafe_allow_html=True,
        )
        c3.markdown(card("Risk tier", tier), unsafe_allow_html=True)

        X_tx = pipeline.named_steps["pre"].transform(df)
        if hasattr(X_tx, "toarray"):
            X_tx = X_tx.toarray()
        shap_values = explainer.shap_values(X_tx)[0]
        order = np.argsort(np.abs(shap_values))[::-1][:10]
        top = pd.DataFrame({
            "feature": [feature_names[i] for i in order],
            "shap_value": [float(shap_values[i]) for i in order],
            "direction": ["increases_risk" if shap_values[i] > 0 else "decreases_risk"
                          for i in order],
        })
        st.write("")
        st.subheader("Top 10 features driving this prediction")
        st.caption(
            "SHAP values quantify each feature's contribution to the model's output. "
            "Red bars push the prediction toward readmission, green bars push it away."
        )

        fig, ax = plt.subplots(figsize=(8, 4.2))
        colors = ["#c44e4e" if d == "increases_risk" else "#2f855a" for d in top["direction"]]
        ax.barh(top["feature"][::-1], top["shap_value"][::-1], color=colors[::-1])
        ax.set_xlabel("SHAP value")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(top, use_container_width=True, hide_index=True)


def page_performance(metadata, thresholds) -> None:
    st.header("Model performance")
    st.markdown(
        "<div class='section-intro'>Headline metrics from the tuned champion model "
        "on the held-out test set. The threshold sweep shows how precision, recall, "
        "and F1 trade off across decision thresholds. Confusion matrices compare the "
        "default cutoff of 0.5 against the tuned operating point.</div>",
        unsafe_allow_html=True,
    )

    metrics = metadata.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("ROC-AUC", f"{metrics.get('roc_auc', 'n/a')}"), unsafe_allow_html=True)
    c2.markdown(card("PR-AUC", f"{metrics.get('pr_auc', 'n/a')}"), unsafe_allow_html=True)
    c3.markdown(card("F1 at 0.5", f"{metrics.get('f1', 'n/a')}"), unsafe_allow_html=True)
    c4.markdown(
        card("CV ROC-AUC", f"{metrics.get('cv_roc_auc', 'n/a')}", "3-fold stratified"),
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Threshold sweep (validation set)")
    img = SCREENSHOTS / "threshold_sweep.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.info("Run `python -m src.threshold` to generate this chart.")

    if thresholds:
        st.subheader("Confusion matrices")
        st.caption(
            "Left: default threshold of 0.5 (higher recall, lower precision). "
            "Right: tuned threshold that maximises F1 on the validation set."
        )
        a, b = st.columns(2)
        for col, key, label in [
            (a, "default_threshold_0_5", "Default threshold = 0.5"),
            (b, "tuned_threshold_f1", f"Tuned threshold = {thresholds.get('best_threshold_f1')}"),
        ]:
            info = thresholds.get(key, {})
            cm = info.get("confusion", {})
            col.markdown(f"**{label}**")
            col.markdown(
                f"Precision: `{info.get('precision', 0):.3f}`  \n"
                f"Recall: `{info.get('recall', 0):.3f}`  \n"
                f"F1: `{info.get('f1', 0):.3f}`"
            )
            if cm:
                col.dataframe(
                    pd.DataFrame(
                        [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]],
                        index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"],
                    )
                )


def page_fairness() -> None:
    st.header("Subgroup fairness audit")
    st.markdown(
        "<div class='section-intro'>We audit whether the model performs comparably "
        "across race and gender subgroups. Large gaps in AUC, false-positive rate, "
        "or false-negative rate indicate the model serves some groups better than "
        "others. Small subgroups have wider confidence intervals.</div>",
        unsafe_allow_html=True,
    )

    path = REPORTS / "fairness_report.json"
    if not path.exists():
        st.warning("Run `python -m src.fairness` to produce the audit.")
        return

    report = json.loads(path.read_text())
    st.caption(f"Evaluated at decision threshold = {report['threshold_used']}")

    for group in ["race", "gender"]:
        st.subheader(f"By {group}")
        rows = []
        for name, m in report[group].items():
            rows.append({
                "subgroup": name, "n": m["n"],
                "positive_rate": round(m["positive_rate"], 3),
                "ROC-AUC": round(m["roc_auc"], 3),
                "FPR": round(m["fpr"], 3) if m.get("fpr") is not None else None,
                "FNR": round(m["fnr"], 3) if m.get("fnr") is not None else None,
            })
        st.dataframe(
            pd.DataFrame(rows).sort_values("n", ascending=False),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Fairness gaps")
    st.caption(
        "Gap = max minus min across subgroups. A gap of 0.10 on AUC or FNR is "
        "generally considered meaningful."
    )
    st.json(report["fairness_gaps"])

    img = SCREENSHOTS / "fairness_subgroup_auc.png"
    if img.exists():
        st.image(str(img), caption="ROC-AUC per subgroup", use_container_width=True)


def page_drift() -> None:
    st.header("Drift monitoring")
    st.markdown(
        "<div class='section-intro'>This dashboard runs an Evidently data drift "
        "check comparing incoming data against the training reference. If drift "
        "crosses configured thresholds, an automated retraining job runs. The new "
        "candidate model only becomes Production if it outperforms the current "
        "model on a held-out test set (the promote-if-better gate).</div>",
        unsafe_allow_html=True,
    )

    verdict_path = REPORTS / "drift_verdict.json"
    if not verdict_path.exists():
        st.warning("Run `python -m src.drift` first.")
        return

    verdict = json.loads(verdict_path.read_text())
    thresh = verdict["thresholds"]

    c1, c2, c3 = st.columns(3)
    drift_val = "Yes" if verdict["dataset_drift"] else "No"
    c1.markdown(
        card("Dataset drift", drift_val, "Evidently default rule (over 50% of cols)"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        card(
            "Columns drifted",
            f"{verdict['drifted_column_count']} of 50",
            f"share = {verdict['drift_share']:.2%}",
        ),
        unsafe_allow_html=True,
    )
    retrain_val = "Yes" if verdict["should_retrain"] else "No"
    c3.markdown(
        card(
            "Retrain triggered",
            retrain_val,
            f"if share >= {thresh['drift_share']} or any score >= {thresh['individual_score']}",
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Top drifted features")
    st.caption(
        "Each feature is tested for distribution shift between reference and current "
        "data. A higher drift score means a larger distributional change."
    )
    if verdict["drifted_columns"]:
        st.dataframe(
            pd.DataFrame(verdict["drifted_columns"]),
            use_container_width=True, hide_index=True,
        )

    retrain_path = REPORTS / "retrain_verdict.json"
    if retrain_path.exists():
        r = json.loads(retrain_path.read_text())
        st.subheader("Latest retraining cycle")
        st.caption(
            "When drift is detected, the platform automatically retrains a candidate "
            "model on the combined reference plus current data. The candidate is "
            "only promoted to Production if it improves ROC-AUC on a held-out test "
            "set. This prevents a drift-triggered retrain from silently regressing "
            "model quality."
        )
        cc1, cc2, cc3 = st.columns(3)
        cc1.markdown(
            card("Candidate ROC-AUC", f"{r['candidate_roc_auc']}"),
            unsafe_allow_html=True,
        )
        cc2.markdown(
            card("Production ROC-AUC", f"{r['production_roc_auc']}", "on the same test set"),
            unsafe_allow_html=True,
        )
        promoted_val = "Promoted" if r["promoted"] else "Rejected"
        hint = "new model is now Production" if r["promoted"] else "safety gate blocked regression"
        cc3.markdown(
            card("Decision", promoted_val, hint),
            unsafe_allow_html=True,
        )
        if not r["promoted"]:
            st.info(
                "The candidate did not beat the current Production model. "
                "The promote-if-better gate prevented a silent regression."
            )


def main() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    pipeline, metadata, thresholds, feature_names, explainer, algorithm = load_assets()
    sidebar_header(algorithm)
    page = st.sidebar.radio(
        "View",
        ["Home", "Predict risk", "Model performance", "Fairness audit", "Drift monitoring"],
    )
    if page == "Home":
        page_home(metadata, algorithm)
    elif page == "Predict risk":
        page_predict(pipeline, feature_names, explainer, thresholds)
    elif page == "Model performance":
        page_performance(metadata, thresholds)
    elif page == "Fairness audit":
        page_fairness()
    else:
        page_drift()


if __name__ == "__main__":
    main()
