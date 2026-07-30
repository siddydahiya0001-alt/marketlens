import pandas as pd

from services.column_detector import detect_data_type, suggest_role, profile_columns
from utils.constants import (
    ROLE_CUSTOMER_ID, ROLE_REVENUE, ROLE_YES_NO, ROLE_CAMPAIGN, ROLE_DATE, ROLE_COST, ROLE_PROFIT,
)


def test_detect_customer_id():
    series = pd.Series([f"CUST{i}" for i in range(100)])
    assert suggest_role("customer_id", series, detect_data_type(series)) == ROLE_CUSTOMER_ID


def test_detect_revenue():
    series = pd.Series([100.0, 200.0, 300.0])
    assert suggest_role("revenue", series, detect_data_type(series)) == ROLE_REVENUE
    assert suggest_role("sales_value", series, detect_data_type(series)) == ROLE_REVENUE


def test_detect_purchase_outcome():
    series = pd.Series(["Yes", "No", "Yes", "No"])
    assert suggest_role("converted", series, detect_data_type(series)) == ROLE_YES_NO
    assert suggest_role("purchased", series, detect_data_type(series)) == ROLE_YES_NO


def test_detect_campaign_category():
    series = pd.Series(["Email", "Social", "Search"])
    assert suggest_role("channel", series, detect_data_type(series)) == ROLE_CAMPAIGN
    assert suggest_role("campaign", series, detect_data_type(series)) == ROLE_CAMPAIGN


def test_detect_date_column():
    series = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])
    assert detect_data_type(series) == "Date"
    assert suggest_role("last_purchase", series, detect_data_type(series)) == ROLE_DATE


def test_detect_cost_and_profit():
    series = pd.Series([10.0, 20.0, 30.0])
    assert suggest_role("ad_spend", series, detect_data_type(series)) == ROLE_COST
    assert suggest_role("cost", series, detect_data_type(series)) == ROLE_COST
    assert suggest_role("margin", series, detect_data_type(series)) == ROLE_PROFIT
    assert suggest_role("profit", series, detect_data_type(series)) == ROLE_PROFIT


def test_profile_columns_shape():
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(20)],
        "revenue": range(20),
        "channel": ["Email"] * 10 + ["Search"] * 10,
    })
    profile = profile_columns(df)
    assert list(profile["Column"]) == ["customer_id", "revenue", "channel"]
    assert "Suggested business meaning" in profile.columns
    assert "User role" in profile.columns
    assert profile.loc[profile["Column"] == "customer_id", "Suggested business meaning"].iloc[0] == ROLE_CUSTOMER_ID
