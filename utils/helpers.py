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

# Normalised spellings that mean the same thing in a Yes/No outcome column. Real
# spreadsheets mix these freely ("Yes", "Y", "yes", "Purchased" in one column), so
# matching a single exact string would silently count most buyers as non-buyers.
# Only unambiguous synonyms live here - domain-specific words like "active" or
# "churned" are deliberately excluded, since they pair with different opposites
# depending on what the column means.
AFFIRMATIVE_LABEL_TOKENS = {
    "yes", "y", "true", "t", "1", "purchased", "purchase", "converted", "bought", "responded",
}
NEGATIVE_LABEL_TOKENS = {
    "no", "n", "false", "f", "0", "notpurchased", "didnotpurchase", "notconverted",
    "notbought", "notresponded", "none",
}


def infer_positive_value(series: pd.Series):
    """Pick the value that should count as the 'positive' outcome (e.g. 'Yes' over 'No').

    Prefers the most common affirmative-looking spelling, so a column mixing
    'Yes', 'Y' and 'Purchased' reports the label the user is most likely to
    recognise. Falls back to the first non-null unique value if nothing looks
    affirmative, so behaviour stays deterministic.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None
    counts = non_null.value_counts()
    affirmative = [v for v in counts.index if normalise_label(v) in POSITIVE_LABEL_TOKENS]
    if affirmative:
        return affirmative[0]
    return non_null.unique().tolist()[0]


def positive_mask(series: pd.Series, positive_value) -> pd.Series:
    """Boolean mask of rows counting as the positive outcome.

    Plain `series == positive_value` is wrong for real spreadsheets: a column
    holding 'Yes', 'yes', 'Y' and 'Purchased' would match only one spelling and
    treat the rest as negatives, corrupting conversion rates and model targets.

    Matching rules, most permissive first:
    - Numeric/boolean columns compare directly (no text normalisation).
    - If every value in the column is a recognisable yes/no token, any
      affirmative spelling matches any other affirmative spelling.
    - Otherwise values are compared after normalising case, whitespace and
      punctuation, so 'Campaign A' still only matches 'Campaign A'.
    """
    if positive_value is None:
        return pd.Series(False, index=series.index)

    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return (series == positive_value).fillna(False)

    not_null = series.notna()
    normalised = series.map(normalise_label)
    target = normalise_label(positive_value)

    distinct = set(normalised[not_null])
    if distinct and distinct.issubset(AFFIRMATIVE_LABEL_TOKENS | NEGATIVE_LABEL_TOKENS):
        if target in AFFIRMATIVE_LABEL_TOKENS:
            return normalised.isin(AFFIRMATIVE_LABEL_TOKENS) & not_null
        if target in NEGATIVE_LABEL_TOKENS:
            return normalised.isin(NEGATIVE_LABEL_TOKENS) & not_null

    return (normalised == target) & not_null


def sample_size_warning(n_rows: int, min_rows: int = 30) -> str | None:
    if n_rows < min_rows:
        return (
            f"Only {n_rows} rows are available. Results based on fewer than {min_rows} rows "
            "should be treated as directional, not conclusive."
        )
    return None
