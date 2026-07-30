"""Analytical guardrails: checks that produce plain-language warnings.

Every analysis module calls into this service before or after modelling to
collect a list of warning strings to display in a "Risks and limitations"
section. Nothing here blocks analysis outright (except truly empty data) -
the goal is honest disclosure, not obstruction.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from utils.constants import ROLE_CUSTOMER_ID, ROLE_TEXT, MIN_ROWS_FOR_ANALYSIS, MIN_ROWS_FOR_FORECAST


def check_dataset_size(df: pd.DataFrame, min_rows: int = MIN_ROWS_FOR_ANALYSIS) -> list:
    warnings = []
    if len(df) < min_rows:
        warnings.append(
            f"The dataset only has {len(df)} rows. With fewer than {min_rows} rows, results are "
            "directional and may change a lot with more data."
        )
    return warnings


def check_target_outcomes(series: pd.Series, min_per_class: int = 10) -> list:
    warnings = []
    counts = series.dropna().value_counts()
    if len(counts) < 2:
        warnings.append("The outcome column only contains one distinct value, so it cannot be predicted.")
        return warnings
    minority = counts.min()
    if minority < min_per_class:
        warnings.append(
            f"The least common outcome only appears {minority} times. Predictions for rare outcomes "
            "are less reliable."
        )
    ratio = counts.min() / counts.max()
    if ratio < 0.15:
        warnings.append(
            "One outcome is far more common than the other. The model may lean toward always "
            "predicting the majority outcome."
        )
    return warnings


def check_missing_data(df: pd.DataFrame, columns: list, threshold: float = 0.3) -> list:
    warnings = []
    for col in columns:
        if col not in df.columns:
            continue
        pct = df[col].isna().mean()
        if pct > threshold:
            warnings.append(
                f"'{col}' is missing in {pct * 100:.0f}% of rows, which may weaken the reliability of the result."
            )
    return warnings


def check_dominant_category(series: pd.Series, threshold: float = 0.9) -> list:
    warnings = []
    non_null = series.dropna()
    if non_null.empty:
        return warnings
    top_share = non_null.value_counts(normalize=True).iloc[0]
    if top_share > threshold:
        top_value = non_null.value_counts().index[0]
        warnings.append(
            f"'{top_value}' makes up {top_share * 100:.0f}% of the '{series.name}' column, leaving little "
            "variation for the analysis to learn from."
        )
    return warnings


def check_model_performance(metric_value: float, metric_name: str, poor_threshold: float) -> list:
    warnings = []
    if metric_value is not None and metric_value < poor_threshold:
        warnings.append(
            f"The model's {metric_name} is lower than usual, so its conclusions should be treated with caution."
        )
    return warnings


def check_multicollinearity(vif_table: pd.DataFrame, threshold: float = 10.0) -> list:
    warnings = []
    if vif_table is None or vif_table.empty:
        return warnings
    high = vif_table[vif_table["VIF"] > threshold]
    if not high.empty:
        names = ", ".join(high["Factor"].tolist())
        warnings.append(
            f"These factors overlap heavily with each other and may be giving similar information: {names}. "
            "Their individual influence is less certain as a result."
        )
    return warnings


def check_outlier_influence(df: pd.DataFrame, numeric_columns: list, outlier_fraction_threshold: float = 0.1) -> list:
    from utils.helpers import count_outliers
    warnings = []
    for col in numeric_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        n_outliers = count_outliers(series)
        if len(series) and n_outliers / len(series) > outlier_fraction_threshold:
            warnings.append(
                f"A notable share of '{col}' values are unusually high or low. Results may be influenced by "
                "a small number of extreme cases."
            )
    return warnings


def check_id_used_as_predictor(selected_columns: list, roles: dict) -> list:
    warnings = []
    id_cols = [c for c in selected_columns if roles.get(c) == ROLE_CUSTOMER_ID]
    if id_cols:
        warnings.append(
            f"{', '.join(id_cols)} looks like a Customer ID, which carries no real business meaning. "
            "It has been excluded from the analysis."
        )
    return warnings


def check_date_used_as_number(selected_columns: list, roles: dict) -> list:
    from utils.constants import ROLE_DATE
    warnings = []
    date_cols = [c for c in selected_columns if roles.get(c) == ROLE_DATE]
    if date_cols:
        warnings.append(
            f"{', '.join(date_cols)} is a date column. Treating dates as ordinary numbers can produce "
            "misleading results, so it has been excluded or converted appropriately."
        )
    return warnings


def check_personal_identifiers(selected_columns: list, roles: dict) -> list:
    warnings = []
    text_like = [c for c in selected_columns if roles.get(c) == ROLE_TEXT or "name" in c.lower()]
    if text_like:
        warnings.append(
            f"{', '.join(text_like)} looks like a name or personal identifier and has been excluded, "
            "since it should not be used as a predictive factor."
        )
    return warnings


def observational_data_warning() -> str:
    return (
        "This comparison is based on observational data, not a controlled experiment. Customers were not "
        "randomly assigned to each group, so other differences between them may explain part of the result."
    )


def correlation_causation_warning() -> str:
    return (
        "This pattern shows association, not proof of cause and effect. For example, it does not prove that "
        "discounts caused higher sales - customers receiving discounts may already have been more likely to buy."
    )


def check_forecast_feasibility(n_periods: int, min_periods: int = MIN_ROWS_FOR_FORECAST) -> list:
    warnings = []
    if n_periods < min_periods:
        warnings.append(
            f"Only {n_periods} time periods are available. A reliable forecast usually needs at least "
            f"{min_periods}, so any forecast shown here should be treated as a rough guide only."
        )
    return warnings


def clean_predictor_list(selected_columns: list, roles: dict) -> tuple[list, list]:
    """Remove ID / date / text columns from a predictor list and return (clean_list, warnings)."""
    warnings = []
    warnings += check_id_used_as_predictor(selected_columns, roles)
    warnings += check_personal_identifiers(selected_columns, roles)

    id_or_text = {c for c in selected_columns if roles.get(c) in (ROLE_CUSTOMER_ID, ROLE_TEXT)}
    cleaned = [c for c in selected_columns if c not in id_or_text]
    return cleaned, warnings
