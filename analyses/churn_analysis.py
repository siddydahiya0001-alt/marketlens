"""Module F: Which customers may stop buying? (interpretable risk scoring under the hood).

Two modes:
  - Supervised: if a known churn/outcome column is provided, an interpretable
    logistic regression is trained (same technique as purchase prediction,
    reframed as risk).
  - Heuristic: otherwise, a transparent weighted z-score risk index is built
    from the selected behaviour signals (e.g. recency, complaints, declining
    frequency), each with a business-sensible direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from services import validation_service as vs
from utils.helpers import slugify, infer_positive_value

RISK_INCREASES_WITH_VALUE = ["recency", "dayssince", "delay", "complaint", "cost", "churn"]
RISK_DECREASES_WITH_VALUE = ["frequency", "visit", "engagement", "loyalty", "tenure", "spend", "revenue", "purchase"]

FRIENDLY_REASON = {
    "recency": "the customer has not purchased recently",
    "dayssince": "the customer has not purchased recently",
    "delay": "recent orders had delivery delays",
    "complaint": "a recent complaint was recorded",
    "frequency": "purchase frequency has declined",
    "visit": "website engagement has dropped",
    "engagement": "engagement has dropped",
}


def _direction_for_column(col: str) -> int:
    slug = slugify(col)
    for keyword in RISK_INCREASES_WITH_VALUE:
        if keyword in slug:
            return 1
    for keyword in RISK_DECREASES_WITH_VALUE:
        if keyword in slug:
            return -1
    return 1


def _friendly_reason(col: str) -> str:
    slug = slugify(col)
    for keyword, phrase in FRIENDLY_REASON.items():
        if keyword in slug:
            return phrase
    return f"'{col}' is unusual compared to other customers"


def run_analysis(df: pd.DataFrame, risk_signal_cols: list, roles: dict, known_churn_col: str | None = None) -> dict:
    warnings = []
    warnings += vs.check_dataset_size(df)

    clean_cols, id_warnings = vs.clean_predictor_list(risk_signal_cols, roles)
    warnings += id_warnings
    numeric_cols = [c for c in clean_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        return {"ok": False, "reason": "Please select at least one numeric behaviour column (e.g. recency, frequency, complaints).",
                "limitations": warnings}

    working = df.copy()
    for col in numeric_cols:
        working[col] = working[col].fillna(working[col].median())

    if known_churn_col and known_churn_col in df.columns:
        result = _supervised_risk(working, numeric_cols, known_churn_col, warnings)
    else:
        result = _heuristic_risk(working, numeric_cols, warnings)

    return result


def _heuristic_risk(df: pd.DataFrame, numeric_cols: list, warnings: list) -> dict:
    scaler = StandardScaler()
    z = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    directions = {col: _direction_for_column(col) for col in numeric_cols}

    signed = z.copy()
    for col in numeric_cols:
        signed[col] = signed[col] * directions[col]

    raw_score = signed.mean(axis=1)
    risk_score_0_100 = ((raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-9) * 100)

    main_reason_idx = signed.abs().idxmax(axis=1)
    main_reasons = main_reason_idx.map(_friendly_reason)

    risk_df = df.copy()
    risk_df["Risk score"] = risk_score_0_100.round(1)
    risk_df["Main risk reason"] = main_reasons
    risk_df["Risk group"] = pd.cut(risk_score_0_100, bins=[-1, 33, 66, 100], labels=["Low", "Medium", "High"])

    reason_counts = main_reasons.value_counts()
    top_reasons = reason_counts.index.tolist()

    warnings.append(
        "No confirmed churn outcome was provided, so this risk score is based on behaviour patterns "
        "(recency, frequency, complaints, etc.) rather than a trained prediction of actual churn."
    )

    return {
        "ok": True,
        "mode": "heuristic",
        "risk_df": risk_df,
        "risk_signal_cols": numeric_cols,
        "n_high": int((risk_df["Risk group"] == "High").sum()),
        "n_medium": int((risk_df["Risk group"] == "Medium").sum()),
        "n_low": int((risk_df["Risk group"] == "Low").sum()),
        "top_reasons": top_reasons,
        "limitations": warnings,
    }


def _supervised_risk(df: pd.DataFrame, numeric_cols: list, churn_col: str, warnings: list) -> dict:
    positive_value = infer_positive_value(df[churn_col])
    y = (df[churn_col] == positive_value).astype(int)
    warnings += vs.check_target_outcomes(y.map({1: "Yes", 0: "No"}))

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)

    if y.nunique() < 2 or len(df) < 10:
        warnings.append("Not enough outcome variety to train a supervised churn model; falling back to a behaviour-based risk score.")
        return _heuristic_risk(df, numeric_cols, warnings)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]

    coefs = pd.Series(model.coef_[0], index=numeric_cols)
    contributions = X * coefs

    main_reason_idx = contributions.abs().idxmax(axis=1)
    main_reasons = main_reason_idx.map(_friendly_reason)

    risk_df = df.copy()
    risk_df["Risk score"] = (probabilities * 100).round(1)
    risk_df["Main risk reason"] = main_reasons
    risk_df["Risk group"] = pd.cut(probabilities * 100, bins=[-1, 33, 66, 100], labels=["Low", "Medium", "High"])

    top_reasons = main_reasons.value_counts().index.tolist()

    return {
        "ok": True,
        "mode": "supervised",
        "risk_df": risk_df,
        "risk_signal_cols": numeric_cols,
        "n_high": int((risk_df["Risk group"] == "High").sum()),
        "n_medium": int((risk_df["Risk group"] == "Medium").sum()),
        "n_low": int((risk_df["Risk group"] == "Low").sum()),
        "top_reasons": top_reasons,
        "limitations": warnings,
    }


def render(result: dict, df: pd.DataFrame, roles: dict, config: dict, settings: dict):
    import streamlit as st
    from services.recommendation_engine import recommend_from_churn
    from visualisations import result_panels, charts

    if not result.get("ok"):
        st.error(result.get("reason", "This analysis could not be run with the selected columns."))
        return

    risk_df = result["risk_df"]
    top_reasons = result["top_reasons"]

    main_answer = (
        f"{result['n_high']} customers show a high risk of stopping purchases, most often because {top_reasons[0]}."
        if top_reasons else f"{result['n_high']} customers show a high risk of stopping purchases."
    )
    result_panels.render_main_answer(
        main_answer,
        "These customers should be prioritised for retention outreach before they churn.",
    )
    confidence = "High confidence" if result["mode"] == "supervised" else "Moderate confidence"
    result_panels.render_confidence(confidence)

    n_total = result["n_high"] + result["n_medium"] + result["n_low"]
    signal_cols = result.get("risk_signal_cols", [])
    if result["mode"] == "supervised":
        reasoning_steps = [
            f"You provided a column with a known outcome, so we trained a statistical model that estimates "
            f"risk probability (technical name: logistic regression) using {len(signal_cols)} behaviour "
            f"signal{'s' if len(signal_cols) != 1 else ''}.",
            f"We scored all {n_total} customers with that model and converted each score into a 0-100 risk "
            "score.",
            "Because the model was trained against real past outcomes, we rate this Higher confidence than "
            "a behaviour-only estimate.",
        ]
    else:
        reasoning_steps = [
            f"No confirmed outcome column was available, so we built a transparent risk score directly from "
            f"{len(signal_cols)} behaviour signal{'s' if len(signal_cols) != 1 else ''} "
            f"({', '.join(signal_cols)}).",
            "Each signal was put on a common scale, then combined using a direction we assign by its name - "
            "for example, more days since a purchase increases risk, while more frequent purchases decreases it.",
            f"We scored all {n_total} customers this way and converted the result into a 0-100 risk score.",
            "Because this score is based on business logic rather than a trained model checked against real "
            "outcomes, we rate it Moderate rather than High confidence.",
        ]
    result_panels.render_reasoning_trail(reasoning_steps)

    st.markdown("### Risk groups")
    from visualisations.metric_cards import render_metric_row
    render_metric_row([
        {"label": "High risk", "value": result["n_high"]},
        {"label": "Medium risk", "value": result["n_medium"]},
        {"label": "Low risk", "value": result["n_low"]},
    ], columns_per_row=3)

    st.markdown("### At-risk customer list")
    display_cols = [c for c in risk_df.columns if c in ("Risk score", "Main risk reason", "Risk group")]
    id_cols = [c for c, r in roles.items() if r == "Customer ID" and c in risk_df.columns]
    show_cols = (id_cols[:1] if id_cols else []) + display_cols
    st.dataframe(
        risk_df.sort_values("Risk score", ascending=False)[show_cols].head(25),
        use_container_width=True, hide_index=True,
    )

    if len(risk_df):
        example = risk_df.sort_values("Risk score", ascending=False).iloc[0]
        example_id = example[id_cols[0]] if id_cols else example.name
        st.caption(f"Example: Customer {example_id} is at {str(example['Risk group']).lower()} risk because {example['Main risk reason']}.")

    recommendations = recommend_from_churn(top_reasons, settings.get("business_priority"))
    result_panels.render_recommendations(recommendations)
    result_panels.render_limitations(
        result.get("limitations", []),
        do_not_conclude=["A high risk score does not guarantee a customer will leave - it flags a pattern worth investigating."],
    )

    st.markdown("### Download")
    from services.export_service import dataframe_to_excel_bytes
    from utils.download_helpers import lazy_download_button
    lazy_download_button(
        "Download at-risk customer list",
        lambda: dataframe_to_excel_bytes(risk_df, "Churn risk"),
        file_name="churn_risk.xlsx",
        key="churn_risk",
    )
