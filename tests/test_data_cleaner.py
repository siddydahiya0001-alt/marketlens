import numpy as np
import pandas as pd

from services.data_cleaner import (
    generate_quality_findings, apply_cleaning_actions, recommend_missing_value_action,
    MISSING_ACTION_KEEP, MISSING_ACTION_REMOVE, MISSING_ACTION_IMPUTE,
)
from utils.constants import ROLE_CUSTOMER_ID, ROLE_YES_NO


def _sample_df():
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C3", "C4"],
        "income": [1000, np.nan, 3000, 3000, 500000],
        "purchased": ["Yes", "Y", "No", "no", np.nan],
        "constant_col": [1, 1, 1, 1, 1],
    })


def test_generate_quality_findings_flags_missing():
    df = _sample_df()
    roles = {"customer_id": ROLE_CUSTOMER_ID, "purchased": ROLE_YES_NO}
    findings = generate_quality_findings(df, roles)
    joined = " ".join(findings)
    assert "income" in joined.lower()


def test_generate_quality_findings_flags_duplicate_ids():
    df = _sample_df()
    roles = {"customer_id": ROLE_CUSTOMER_ID}
    findings = generate_quality_findings(df, roles)
    assert any("customer id" in f.lower() for f in findings)


def test_generate_quality_findings_flags_constant_column():
    df = _sample_df()
    findings = generate_quality_findings(df, {})
    assert any("constant_col" in f for f in findings)


def test_generate_quality_findings_flags_similar_labels():
    df = _sample_df()
    roles = {"purchased": ROLE_YES_NO}
    findings = generate_quality_findings(df, roles)
    assert any("similar labels" in f.lower() for f in findings)


def test_apply_cleaning_remove_missing_rows():
    df = _sample_df()
    cleaned, log = apply_cleaning_actions(df, {"remove_missing_rows": True}, {})
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) < len(df)


def test_apply_cleaning_fill_median_and_mode():
    df = _sample_df()
    cleaned, log = apply_cleaning_actions(
        df, {"fill_numeric_median": True, "fill_categorical_mode": True}, {},
    )
    assert cleaned["income"].isna().sum() == 0
    assert cleaned["purchased"].isna().sum() == 0


def test_apply_cleaning_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    cleaned, log = apply_cleaning_actions(df, {"remove_duplicates": True}, {})
    assert len(cleaned) == 2


def test_apply_cleaning_exclude_columns():
    df = _sample_df()
    cleaned, log = apply_cleaning_actions(df, {"excluded_columns": ["constant_col"]}, {})
    assert "constant_col" not in cleaned.columns


def test_apply_cleaning_standardise_labels():
    df = _sample_df()
    roles = {"purchased": ROLE_YES_NO}
    cleaned, log = apply_cleaning_actions(df, {"standardise_labels": True}, roles)
    values = set(cleaned["purchased"].dropna().unique())
    assert values.issubset({"Yes", "No"})


def test_row_counts_preserved_without_row_removal_actions():
    df = _sample_df()
    cleaned, log = apply_cleaning_actions(df, {}, {})
    assert len(cleaned) == len(df)


def test_recommend_missing_value_action_no_missing_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    rec = recommend_missing_value_action(df, {})
    assert rec["action"] == MISSING_ACTION_KEEP
    assert rec["row_loss_pct"] == 0.0
    assert rec["high_missing_columns"] == []


def test_recommend_missing_value_action_low_missing_recommends_removal():
    # 100 rows, only 2 have any missing value (2%) -> safe to just drop them
    n = 100
    values = list(range(n))
    values[0] = np.nan
    values[1] = np.nan
    df = pd.DataFrame({"a": values})
    rec = recommend_missing_value_action(df, {})
    assert rec["action"] == MISSING_ACTION_REMOVE
    assert rec["row_loss_pct"] == 2.0


def test_recommend_missing_value_action_high_missing_recommends_imputation():
    # 100 rows, 40 missing (40%) -> too much to drop, should recommend imputing
    n = 100
    values = [np.nan if i < 40 else i for i in range(n)]
    df = pd.DataFrame({"a": values})
    rec = recommend_missing_value_action(df, {})
    assert rec["action"] == MISSING_ACTION_IMPUTE
    assert rec["row_loss_pct"] == 40.0


def test_recommend_missing_value_action_flags_very_high_missing_column():
    n = 100
    a = [np.nan if i < 60 else i for i in range(n)]  # 60% missing - above the high-missing threshold
    b = list(range(n))
    df = pd.DataFrame({"a": a, "b": b})
    rec = recommend_missing_value_action(df, {})
    assert rec["action"] == MISSING_ACTION_IMPUTE
    assert "a" in rec["high_missing_columns"]
    assert "b" not in rec["high_missing_columns"]
