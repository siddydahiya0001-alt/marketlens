"""Small reusable helper functions used throughout MarketLens."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd


def format_number(value, number_format: str = "1,234.56", decimals: int = 0) -> str:
    """Format a number according to a user-selected style."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    text = f"{value:,.{decimals}f}"
    if number_format == "1.234,56":
        text = text.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
    elif number_format == "1 234.56":
        text = text.replace(",", " ")
    return text


def format_currency(value, currency: str = "$", number_format: str = "1,234.56") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{currency}{format_number(value, number_format, decimals=0)}"


def format_percent(value, decimals: int = 1) -> str:
    """value is expected as a fraction (0.12) or already a percent (12)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if abs(value) <= 1.0:
        value = value * 100
    return f"{value:.{decimals}f}%"


def safe_divide(numerator, denominator, default=0.0):
    try:
        if denominator in (0, None) or (isinstance(denominator, float) and np.isnan(denominator)):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalise_label(text) -> str:
    """Normalise inconsistent category text: trims, lowers, strips punctuation for comparison."""
    if pd.isna(text):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


def is_probably_id_column(series: pd.Series, name: str) -> bool:
    id_keywords = ["id", "customer_id", "cust_id", "user_id", "uuid", "account_no", "acct"]
    name_lower = slugify(name)
    if any(k in name_lower for k in id_keywords):
        return True
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    uniqueness_ratio = non_null.nunique() / len(non_null)
    return uniqueness_ratio > 0.98 and len(non_null) > 20


def truncate_label(text: str, max_len: int = 24) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def outlier_bounds(series: pd.Series, k: float = 1.5):
    """Return (lower, upper) Tukey fence bounds for outlier detection."""
    clean = series.dropna()
    if len(clean) == 0:
        return None, None
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def count_outliers(series: pd.Series, k: float = 1.5) -> int:
    lower, upper = outlier_bounds(series, k)
    if lower is None:
        return 0
    clean = series.dropna()
    return int(((clean < lower) | (clean > upper)).sum())


POSITIVE_LABEL_TOKENS = {
    "yes", "y", "true", "1", "purchased", "converted", "active", "churned", "bought", "responded",
}


def infer_positive_value(series: pd.Series):
    """Pick the value that should count as the 'positive' outcome (e.g. 'Yes' over 'No').

    Prefers values that look affirmative by name; falls back to the first
    non-null unique value if nothing matches so behaviour stays deterministic.
    """
    uniques = series.dropna().unique().tolist()
    if not uniques:
        return None
    for value in uniques:
        if normalise_label(value) in POSITIVE_LABEL_TOKENS:
            return value
    return uniques[0]


def sample_size_warning(n_rows: int, min_rows: int = 30) -> str | None:
    if n_rows < min_rows:
        return (
            f"Only {n_rows} rows are available. Results based on fewer than {min_rows} rows "
            "should be treated as directional, not conclusive."
        )
    return None
