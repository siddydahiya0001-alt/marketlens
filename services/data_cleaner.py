"""Data-quality findings (plain language) and cleaning actions.

`generate_quality_findings` never mutates the input DataFrame. Cleaning
actions operate on a copy so the originally uploaded data is preserved.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from utils.constants import ROLE_CUSTOMER_ID, ROLE_YES_NO
from utils.helpers import count_outliers, normalise_label


def _missing_finding_text(col: str, pct: float) -> str:
    """Plain-language missing-value finding, with severity-appropriate framing."""
    if pct >= 50:
        return (
            f"{pct:.0f}% of values are missing in the '{col}' column - more than half the data is "
            "missing, so this column may not be reliable to use in analysis."
        )
    if pct >= 30:
        return (
            f"{pct:.0f}% of values are missing in the '{col}' column - this is a lot of missing data; "
            "consider excluding this column or filling the gaps carefully."
        )
    if pct >= 10:
        return (
            f"{pct:.0f}% of values are missing in the '{col}' column - a moderate amount that filling "
            "in with typical values should handle well."
        )
    if pct >= 1:
        return f"{pct:.0f}% of values are missing in the '{col}' column - a small amount, unlikely to affect results much."
    return f"Less than 1% of values are missing in the '{col}' column."


def generate_quality_findings(df: pd.DataFrame, roles: dict) -> list:
    """Return a list of plain-language data-quality findings (strings)."""
    findings = []
    n = len(df)

    for col in df.columns:
        series = df[col]
        missing_pct = series.isna().mean() * 100
        if missing_pct > 0:
            findings.append(_missing_finding_text(col, missing_pct))

        if series.dropna().nunique() == 1 and n > 1:
            findings.append(f"The '{col}' column has only one value and will not help the analysis.")

        role = roles.get(col)
        if role == ROLE_CUSTOMER_ID:
            n_dupes = int(series.duplicated().sum())
            if n_dupes > 0:
                findings.append(f"Customer IDs are repeated in {n_dupes} rows of the '{col}' column.")

        if pd.api.types.is_numeric_dtype(series) and role not in (ROLE_CUSTOMER_ID,):
            n_outliers = count_outliers(series)
            if n_outliers > 0:
                findings.append(f"The '{col}' column contains {n_outliers} unusually high or low values.")

        if role == ROLE_YES_NO or (series.dropna().dtype == object and series.dropna().nunique() <= 8):
            similar = _find_similar_labels(series)
            if similar:
                groups_text = "; ".join(
                    f"{', '.join(group)}" for group in similar
                )
                findings.append(f"The '{col}' column contains similar labels such as {groups_text}.")

    n_dupe_rows = int(df.duplicated().sum())
    if n_dupe_rows > 0:
        findings.append(f"There are {n_dupe_rows} fully duplicate rows in the data.")

    return findings


MISSING_ACTION_KEEP = "Keep missing values"
MISSING_ACTION_REMOVE = "Remove rows with missing values"
MISSING_ACTION_IMPUTE = "Replace missing numbers with median and categories with most common value"
MISSING_ACTION_OPTIONS = [MISSING_ACTION_KEEP, MISSING_ACTION_REMOVE, MISSING_ACTION_IMPUTE]

# A column this empty is rarely worth imputing - excluding it is usually the honest choice.
HIGH_MISSING_THRESHOLD_PCT = 30
# Below this share of affected rows, simply dropping them barely changes the dataset.
SAFE_ROW_LOSS_THRESHOLD_PCT = 5


def recommend_missing_value_action(df: pd.DataFrame, roles: dict | None = None) -> dict:
    """Suggest the best default way to handle missing values, with a plain-language reason.

    Returns a dict with:
      - action: one of MISSING_ACTION_OPTIONS
      - reason: plain-language explanation of why
      - row_loss_pct: % of rows that would be removed if rows with any missing value were dropped
      - high_missing_columns: columns with >= HIGH_MISSING_THRESHOLD_PCT% missing, worth
        considering for exclusion rather than imputation
    """
    n_rows = len(df)
    missing_by_col = df.isna().mean() * 100
    cols_with_missing = missing_by_col[missing_by_col > 0]

    if cols_with_missing.empty or n_rows == 0:
        return {
            "action": MISSING_ACTION_KEEP,
            "reason": "No missing values were found, so no action is needed here.",
            "row_loss_pct": 0.0,
            "high_missing_columns": [],
        }

    rows_with_any_missing = int(df[cols_with_missing.index].isna().any(axis=1).sum())
    row_loss_pct = rows_with_any_missing / n_rows * 100
    high_missing_columns = cols_with_missing[cols_with_missing >= HIGH_MISSING_THRESHOLD_PCT].index.tolist()

    if row_loss_pct <= SAFE_ROW_LOSS_THRESHOLD_PCT:
        return {
            "action": MISSING_ACTION_REMOVE,
            "reason": (
                f"Removing rows with missing values would only affect {row_loss_pct:.0f}% of your data - "
                "a small enough amount that removing them is simple and unlikely to change your results."
            ),
            "row_loss_pct": row_loss_pct,
            "high_missing_columns": high_missing_columns,
        }

    if high_missing_columns:
        cols_text = ", ".join(f"'{c}'" for c in high_missing_columns)
        reason = (
            f"Removing rows would lose {row_loss_pct:.0f}% of your data, which is too much to simply "
            f"drop. Filling the gaps with typical values keeps every row, though {cols_text} "
            "is missing so much data that excluding it may be more honest than filling it in."
        )
    else:
        reason = (
            f"Removing rows would lose {row_loss_pct:.0f}% of your data. Filling the gaps with typical "
            "values (the median for numbers, the most common value for categories) keeps every row "
            "while still handling the missing data."
        )

    return {
        "action": MISSING_ACTION_IMPUTE,
        "reason": reason,
        "row_loss_pct": row_loss_pct,
        "high_missing_columns": high_missing_columns,
    }


def _find_similar_labels(series: pd.Series):
    """Group distinct labels that normalise to the same token, e.g. 'Yes', 'yes', 'Y'."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return []
    buckets: dict[str, set] = {}
    for value in non_null.unique():
        key = normalise_label(value)
        buckets.setdefault(key, set()).add(value)

    # Also merge common Yes/No style shorthand into buckets regardless of exact normalised match
    yes_like = {"yes", "y", "true", "purchased", "1"}
    no_like = {"no", "n", "false", "notpurchased", "0"}
    merged_yes, merged_no = set(), set()
    for value in non_null.unique():
        norm = normalise_label(value)
        if norm in yes_like:
            merged_yes.add(value)
        if norm in no_like:
            merged_no.add(value)

    groups = [g for g in buckets.values() if len(g) > 1]
    if len(merged_yes) > 1:
        groups.append(merged_yes)
    if len(merged_no) > 1:
        groups.append(merged_no)
    return groups


# ---------------------------------------------------------------------------
# Cleaning actions
# ---------------------------------------------------------------------------

def apply_cleaning_actions(df: pd.DataFrame, actions: dict, roles: dict) -> tuple[pd.DataFrame, list]:
    """Apply a set of user-chosen cleaning actions to a copy of df.

    `actions` keys (all optional, default False/None):
      - remove_missing_rows: bool
      - fill_numeric_median: bool
      - fill_categorical_mode: bool
      - remove_duplicates: bool
      - excluded_columns: list[str]
      - standardise_labels: bool
    Returns (cleaned_df, log_messages)
    """
    cleaned = df.copy()
    log = []
    rows_before = len(cleaned)

    excluded = actions.get("excluded_columns") or []
    if excluded:
        cleaned = cleaned.drop(columns=[c for c in excluded if c in cleaned.columns])
        log.append(f"Excluded columns: {', '.join(excluded)}.")

    if actions.get("standardise_labels"):
        for col in cleaned.columns:
            if roles.get(col) == ROLE_YES_NO or (cleaned[col].dtype == object and cleaned[col].nunique() <= 8):
                cleaned[col] = _standardise_series_labels(cleaned[col])
        log.append("Standardised similar category labels.")

    if actions.get("fill_numeric_median"):
        for col in cleaned.select_dtypes(include="number").columns:
            if cleaned[col].isna().any():
                median_val = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(median_val)
        log.append("Replaced missing numeric values with the column median.")

    if actions.get("fill_categorical_mode"):
        for col in cleaned.select_dtypes(include="object").columns:
            if cleaned[col].isna().any() and cleaned[col].notna().any():
                mode_val = cleaned[col].mode(dropna=True)
                if len(mode_val):
                    cleaned[col] = cleaned[col].fillna(mode_val.iloc[0])
        log.append("Replaced missing category values with the most common value.")

    if actions.get("remove_missing_rows"):
        before = len(cleaned)
        cleaned = cleaned.dropna()
        log.append(f"Removed {before - len(cleaned)} rows containing missing values.")

    if actions.get("remove_duplicates"):
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        log.append(f"Removed {before - len(cleaned)} duplicate rows.")

    rows_after = len(cleaned)
    log.append(f"Row count changed from {rows_before} to {rows_after}.")
    return cleaned, log


def _standardise_series_labels(series: pd.Series) -> pd.Series:
    yes_like = {"yes", "y", "true", "purchased", "1"}
    no_like = {"no", "n", "false", "notpurchased", "0"}

    def _map(value):
        if pd.isna(value):
            return value
        norm = normalise_label(value)
        if norm in yes_like:
            return "Yes"
        if norm in no_like:
            return "No"
        return str(value).strip()

    return series.apply(_map)
