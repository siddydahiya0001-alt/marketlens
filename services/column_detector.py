"""Automatic column-type and business-role detection.

Uses column names (keyword matching) combined with pandas dtypes and simple
value inspection to suggest a business meaning for every column. All
suggestions remain user-editable in the UI - nothing here is final.
"""

from __future__ import annotations

import re
import pandas as pd

from utils.constants import (
    ROLE_CUSTOMER_ID, ROLE_NUMBER, ROLE_CATEGORY, ROLE_DATE, ROLE_YES_NO,
    ROLE_REVENUE, ROLE_SALES, ROLE_COST, ROLE_PROFIT, ROLE_CAMPAIGN,
    ROLE_BEHAVIOUR, ROLE_TEXT, ROLE_UNKNOWN,
)
from utils.helpers import slugify, normalise_label

# Ordered keyword rules: (regex applied to slugified column name, role)
# Behaviour and date rules are checked before the Yes/No rule so that compound
# names like 'purchase_frequency' or 'days_since_last_purchase' - which merely
# contain the word 'purchase' - aren't misread as a Yes/No outcome column.
NAME_KEYWORD_RULES = [
    (r"(customer.?id|cust.?id|user.?id|client.?id|account.?id|^id$|_id$)", ROLE_CUSTOMER_ID),
    (r"(revenue|sales.?value|amount|order.?value|sale.?price)", ROLE_REVENUE),
    (r"(^sales$|sales.?volume|units.?sold|quantity.?sold)", ROLE_SALES),
    (r"(cost|spend|ad.?spend|expenditure)", ROLE_COST),
    (r"(profit|margin)", ROLE_PROFIT),
    (r"(campaign|channel|source|medium)", ROLE_CAMPAIGN),
    (r"(visit|frequency|recency|engagement|complaint|delay|tenure|clicks|sessions|logins)", ROLE_BEHAVIOUR),
    (r"(date|_dt$|timestamp|month|year)", ROLE_DATE),
    (r"(name|comment|review|feedback|description|notes)", ROLE_TEXT),
]

# Checked as whole underscore-delimited tokens (not substrings) so that a word
# like 'purchase' embedded inside 'purchase_frequency' does not trigger a
# Yes/No classification - only an exact token match (e.g. 'purchased') does.
YES_NO_NAME_TOKENS = {
    "converted", "purchase", "purchased", "bought", "responded", "response",
    "churn", "churned", "active", "subscribed", "clicked", "opened",
}

YES_NO_VALUE_SETS = [
    {"yes", "no"}, {"y", "n"}, {"true", "false"}, {"1", "0"},
    {"purchased", "notpurchased"}, {"converted", "notconverted"}, {"active", "inactive"},
]

# Normalised (lowercase, punctuation-stripped) tokens that count as an
# affirmative / negative Yes-No value. Used so that messy real-world label
# variants (Yes, yes, Y, Purchased) are still recognised as Yes/No data,
# even before the user standardises labels on the Data Quality page.
YES_NO_AFFIRMATIVE_TOKENS = {"yes", "y", "true", "purchased", "1", "converted", "active", "churned"}
YES_NO_NEGATIVE_TOKENS = {"no", "n", "false", "notpurchased", "0", "notconverted", "inactive", "notchurned"}


def _looks_like_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(30)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def _looks_like_yes_no(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_vals = {str(v).strip().lower() for v in non_null.unique()}

    if len(unique_vals) <= 4:
        for value_set in YES_NO_VALUE_SETS:
            if unique_vals.issubset(value_set) or unique_vals == value_set:
                return True
        if len(unique_vals) == 2 and pd.api.types.is_numeric_dtype(series):
            if unique_vals.issubset({"0", "1", "0.0", "1.0"}):
                return True

    # Fall back to normalised affirmative/negative token matching, which
    # tolerates messy label variants (Yes, yes, Y, Purchased) as long as
    # every distinct value maps cleanly to one side or the other.
    normalised_vals = {normalise_label(v) for v in non_null.unique()}
    if normalised_vals and normalised_vals.issubset(YES_NO_AFFIRMATIVE_TOKENS | YES_NO_NEGATIVE_TOKENS):
        has_affirmative = bool(normalised_vals & YES_NO_AFFIRMATIVE_TOKENS)
        has_negative = bool(normalised_vals & YES_NO_NEGATIVE_TOKENS)
        return has_affirmative and has_negative
    return False


def detect_data_type(series: pd.Series) -> str:
    """Low-level pandas-ish type: Number, Date, Category/Text, Yes/No."""
    if _looks_like_date(series):
        return "Date"
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "Number"
    if _looks_like_yes_no(series):
        return "Yes/No"
    non_null = series.dropna()
    if non_null.empty:
        return "Empty"
    avg_len = non_null.astype(str).str.len().mean()
    unique_ratio = non_null.nunique() / len(non_null)
    if avg_len > 25 and unique_ratio > 0.5:
        return "Free text"
    return "Category"


YES_NO_DISQUALIFYING_TOKENS = {
    "frequency", "count", "rate", "score", "amount", "value", "percentage",
    "number", "total", "days", "since", "time", "duration", "delay", "date",
}


def _name_suggests_yes_no(slug: str) -> bool:
    tokens = slug.split("_")
    if not any(token in YES_NO_NAME_TOKENS for token in tokens):
        return False
    return not any(token in YES_NO_DISQUALIFYING_TOKENS for token in tokens)


def suggest_role(name: str, series: pd.Series, data_type: str) -> str:
    slug = slugify(name)

    if _looks_like_date(series):
        return ROLE_DATE

    if _name_suggests_yes_no(slug):
        if _looks_like_yes_no(series) or pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() <= 2:
            return ROLE_YES_NO
        # Name suggests yes/no but the values don't look like it - fall back on shape
        if pd.api.types.is_numeric_dtype(series):
            return ROLE_NUMBER
        if series.dropna().nunique() <= 4:
            return ROLE_CATEGORY

    for pattern, role in NAME_KEYWORD_RULES:
        if re.search(pattern, slug):
            return role

    # No name match - fall back on data shape
    if data_type == "Date":
        return ROLE_DATE
    if data_type == "Yes/No":
        return ROLE_YES_NO
    if data_type == "Number":
        non_null = series.dropna()
        if len(non_null) > 20 and non_null.nunique() / len(non_null) > 0.98:
            return ROLE_CUSTOMER_ID
        return ROLE_NUMBER
    if data_type == "Free text":
        return ROLE_TEXT
    if data_type == "Category":
        return ROLE_CATEGORY
    return ROLE_UNKNOWN


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Build the column-mapping table shown to the user."""
    rows = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        data_type = detect_data_type(series)
        role = suggest_role(col, series, data_type)
        missing_pct = round(series.isna().mean() * 100, 1)
        n_unique = int(series.nunique(dropna=True))
        rows.append({
            "Column": col,
            "Detected data type": data_type,
            "Suggested business meaning": role,
            "Missing %": missing_pct,
            "Unique values": n_unique,
            "User role": role,  # editable copy, starts equal to suggestion
        })
    profile = pd.DataFrame(rows)
    return profile


def roles_from_profile(profile: pd.DataFrame) -> dict:
    return dict(zip(profile["Column"], profile["User role"]))
