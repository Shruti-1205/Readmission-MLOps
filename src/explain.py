"""Generate SHAP explainability plots for the tuned XGBoost model."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.pipeline import (
    RANDOM_STATE,
    build_preprocessor,
    compute_scale_pos_weight,
    load_splits,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
SAMPLE_SIZE = 2000


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, cat_cols, num_cols = load_splits()
    spw = compute_scale_pos_weight(y_train)

    pre = build_preprocessor(cat_cols, num_cols)
    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=spw, tree_method="hist", eval_metric="auc",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    X_test_sample = X_test.sample(n=min(SAMPLE_SIZE, len(X_test)), random_state=RANDOM_STATE)
    X_test_tx = pipe.named_steps["pre"].transform(X_test_sample)
    if hasattr(X_test_tx, "toarray"):
        X_test_tx = X_test_tx.toarray()

    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_test_tx)

    plt.figure()
    shap.summary_plot(
        shap_values, X_test_tx, feature_names=feature_names,
        plot_type="bar", max_display=15, show=False,
    )
    plt.tight_layout()
    bar_path = OUT_DIR / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values, X_test_tx, feature_names=feature_names,
        max_display=15, show=False,
    )
    plt.tight_layout()
    beeswarm_path = OUT_DIR / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    proba = pipe.predict_proba(X_test_sample)[:, 1]
    highest_idx = int(np.argmax(proba))
    plt.figure()
    shap.force_plot(
        explainer.expected_value, shap_values[highest_idx], X_test_tx[highest_idx],
        feature_names=feature_names, matplotlib=True, show=False,
    )
    force_path = OUT_DIR / "shap_force_high_risk_patient.png"
    plt.savefig(force_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP plots to {OUT_DIR}:")
    print(f"  - {bar_path.name}  (global feature importance bar chart)")
    print(f"  - {beeswarm_path.name}  (directional feature impact beeswarm)")
    print(f"  - {force_path.name}  (highest-risk patient, p={proba[highest_idx]:.3f})")


if __name__ == "__main__":
    main()
