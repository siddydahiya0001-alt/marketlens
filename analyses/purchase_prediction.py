"""Module C: Who is most likely to purchase? (logistic regression under the hood)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve,
)

from services import validation_service as vs
from utils.helpers import infer_positive_value, positive_mask


def _encode_features(df: pd.DataFrame, factor_cols: list) -> pd.DataFrame:
    frame = df[factor_cols].copy()
    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in frame.columns if c not in numeric_cols]

    for col in numeric_cols:
        frame[col] = frame[col].fillna(frame[col].median())
    for col in categorical_cols:
        mode = frame[col].mode()
        frame[col] = frame[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    encoded = pd.get_dummies(frame, columns=categorical_cols, drop_first=True)
    encoded = encoded.loc[:, encoded.nunique(dropna=False) > 1]
    return encoded


def run_analysis(df: pd.DataFrame, outcome_col: str, factor_cols: list, roles: dict, positive_value=None) -> dict:
    warnings = []
    warnings += vs.check_dataset_size(df)

    clean_factors, id_warnings = vs.clean_predictor_list(factor_cols, roles)
    warnings += id_warnings
    warnings += vs.check_date_used_as_number(factor_cols, roles)

    if not outcome_col or outcome_col not in df.columns or not clean_factors:
        return {"ok": False, "reason": "Please select an outcome column and at least one factor.", "limitations": warnings}

    working = df[[outcome_col] + clean_factors].dropna(subset=[outcome_col]).copy()

    if positive_value is None:
        positive_value = infer_positive_value(working[outcome_col])
    y = positive_mask(working[outcome_col], positive_value).astype(int)

    warnings += vs.check_target_outcomes(y.map({1: "Yes", 0: "No"}))
    warnings += vs.check_missing_data(working, clean_factors)

    X = _encode_features(working, clean_factors)
    valid_idx = X.index.intersection(y.index)
    X, y = X.loc[valid_idx], y.loc[valid_idx]

    if X.shape[1] == 0 or len(X) < 10 or y.nunique() < 2:
        return {"ok": False, "reason": "Not enough usable data or outcome variety to run this analysis.", "limitations": warnings}

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=stratify,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_proba_test))
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    except ValueError:
        auc, fpr, tpr = None, None, None
    cm = confusion_matrix(y_test, y_pred)

    warnings += vs.check_model_performance(accuracy, "accuracy", 0.55)

    # Final model on all data for business-facing probabilities
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_scaled, y)
    probabilities = final_model.predict_proba(X_scaled)[:, 1]

    predictions_df = working.loc[valid_idx, [outcome_col]].copy()
    predictions_df["Purchase probability"] = probabilities
    predictions_df["Group"] = pd.cut(
        probabilities, bins=[-0.01, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"],
    )

    coef_series = pd.Series(final_model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    positive_signals = coef_series[coef_series > 0].index.tolist()
    negative_signals = coef_series[coef_series < 0].index.tolist()

    ranked_factors = [
        {
            "name": name,
            "direction": "positive" if value > 0 else "negative",
            "strength": _strength_label(value),
            "score": abs(value),
        }
        for name, value in coef_series.items()
    ]

    return {
        "ok": True,
        "outcome_col": outcome_col,
        "positive_value": positive_value,
        "factor_cols": clean_factors,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "top_positive_signals": positive_signals,
        "top_negative_signals": negative_signals,
        "ranked_factors": ranked_factors,
        "predictions_df": predictions_df,
        "n_high": int((predictions_df["Group"] == "High").sum()),
        "n_medium": int((predictions_df["Group"] == "Medium").sum()),
        "n_low": int((predictions_df["Group"] == "Low").sum()),
        "limitations": warnings,
    }


def _strength_label(value: float) -> str:
    v = abs(value)
    if v >= 1.0:
        return "Strong"
    if v >= 0.4:
        return "Moderate"
    if v >= 0.15:
        return "Weak"
    return "Very weak"


def render(result: dict, df: pd.DataFrame, roles: dict, config: dict, settings: dict):
    import streamlit as st
    from services.interpretation_service import get_interpreter
    from services.recommendation_engine import recommend_from_purchase_prediction
    from services import validation_service as vs
    from visualisations import charts, result_panels

    if not result.get("ok"):
        st.error(result.get("reason", "This analysis could not be run with the selected columns."))
        return

    interpreter = get_interpreter()
    interpretation = interpreter.interpret_classification(result)
    recommendations = recommend_from_purchase_prediction(
        result["top_positive_signals"], settings.get("business_priority"), result.get("accuracy"),
    )

    ranking_fig, ranking_explanation = charts.horizontal_ranking_chart(
        result["ranked_factors"][:10], title="What signals purchase likelihood?",
    )

    better_text = (
        "The model is better at identifying buyers than non-buyers."
        if result["recall"] >= result["precision"] else
        "The model is better at avoiding false alarms than catching every buyer."
    )

    result_panels.render_main_answer(interpretation["main_answer"], interpretation["explanation"] + " " + better_text)
    result_panels.render_confidence(interpretation["confidence"])
    result_panels.render_reasoning_trail(interpretation.get("reasoning_steps", []))

    st.markdown("### Customer probability groups")
    from visualisations.metric_cards import render_metric_row
    render_metric_row([
        {"label": "High probability", "value": result["n_high"]},
        {"label": "Medium probability", "value": result["n_medium"]},
        {"label": "Low probability", "value": result["n_low"]},
    ], columns_per_row=3)

    result_panels.render_factor_ranking(ranking_fig, ranking_explanation)
    result_panels.render_recommendations(recommendations)
    result_panels.render_limitations(
        result.get("limitations", []),
        do_not_conclude=[
            "A high probability score is not a guarantee of purchase - it reflects a pattern, not certainty.",
            vs.correlation_causation_warning(),
        ],
    )

    st.markdown("### Download")
    from services.export_service import dataframe_to_excel_bytes
    from utils.download_helpers import lazy_download_button
    lazy_download_button(
        "Download customer-level predictions",
        lambda: dataframe_to_excel_bytes(result["predictions_df"].reset_index(drop=True), "Predictions"),
        file_name="purchase_predictions.xlsx",
        key="purchase_predictions",
    )

    with st.expander("Show technical details", expanded=settings.get("show_technical", False)):
        if settings.get("show_technical"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{result['accuracy']:.2%}")
            c2.metric("Precision", f"{result['precision']:.2%}")
            c3.metric("Recall", f"{result['recall']:.2%}")
            c4.metric("ROC-AUC", f"{result['auc']:.3f}" if result["auc"] else "N/A")

            cm_fig, cm_expl = charts.confusion_matrix_chart(result["confusion_matrix"], ["No", "Yes"])
            st.plotly_chart(cm_fig, use_container_width=True)
            st.caption(cm_expl)

            if result["fpr"] is not None:
                roc_fig, roc_expl = charts.roc_curve_chart(result["fpr"], result["tpr"], result["auc"])
                st.plotly_chart(roc_fig, use_container_width=True)
                st.caption(roc_expl)
        else:
            st.caption("Turn on 'Show technical details' in the sidebar to see statistical output.")
