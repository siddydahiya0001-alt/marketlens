"""Regression tests for churn risk direction and scoring.

The heuristic risk score assigns each behaviour signal a direction from its
column name. Keywords were matched against an underscored slug, so a
multi-word keyword like "dayssince" could never match a real column such as
"Days_Since_Last_Purchase" - which then fell through to the risk-*decreasing*
list on the stray word "purchase". The most dormant customers were scored as
the safest. These tests pin the direction of every signal the shipped sample
data uses.
"""

import numpy as np
import pandas as pd
import pytest

from analyses import churn_analysis
from analyses.churn_analysis import (
    NO_STANDOUT_REASON, _direction_for_column, _friendly_reason,
)

from tests.sample_workbook import load_sample_customer_data


# ---------------------------------------------------------------------------
# Direction of each signal
# ---------------------------------------------------------------------------

RISK_INCREASING_COLUMNS = [
    "Days_Since_Last_Purchase",
    "days since last purchase",
    "DaysSinceLastOrder",
    "Months_Since_Last_Visit",
    "Recency",
    "Customer_Service_Complaints",
    "Delivery_Delay_Days",
    "Refund_Count",
    "Cancelled_Orders",
]

RISK_DECREASING_COLUMNS = [
    "Purchase_Frequency",
    "Website_Visits",
    "Average_Order_Value",
    "Total_Spend",
    "Revenue",
    "Income",
    "Loyalty_Points",
    "Tenure_Months",
    "Engagement_Score",
]


@pytest.mark.parametrize("column", RISK_INCREASING_COLUMNS)
def test_higher_value_increases_risk(column):
    assert _direction_for_column(column) == 1, f"{column} should raise churn risk"


@pytest.mark.parametrize("column", RISK_DECREASING_COLUMNS)
def test_higher_value_decreases_risk(column):
    assert _direction_for_column(column) == -1, f"{column} should lower churn risk"


def test_recency_wins_over_the_word_purchase_in_the_same_name():
    """'Days_Since_Last_Purchase' contains 'purchase', but it is a recency
    signal - the recency meaning must take priority."""
    assert _direction_for_column("Days_Since_Last_Purchase") == 1
    assert _direction_for_column("Purchase_Frequency") == -1


def test_recency_columns_get_a_recency_reason_not_the_generic_fallback():
    assert _friendly_reason("Days_Since_Last_Purchase") == "the customer has not purchased recently"
    assert _friendly_reason("Purchase_Frequency") == "purchase frequency has declined"
    assert _friendly_reason("Delivery_Delay_Days") == "recent orders had delivery delays"
    assert _friendly_reason("Customer_Service_Complaints") == "a recent complaint was recorded"


def test_unknown_columns_fall_back_to_a_generic_reason():
    assert "unusual compared to other customers" in _friendly_reason("Some_Unmapped_Column")


# ---------------------------------------------------------------------------
# End-to-end scoring
# ---------------------------------------------------------------------------

def _churn_df():
    """One clearly dormant customer among otherwise identical regulars.

    Deliberately built from recency and frequency only. Adding a complaint
    column would let that signal carry the result and hide an inverted recency
    direction, which is the bug these tests exist to catch.
    """
    return pd.DataFrame({
        "Days_Since_Last_Purchase": [10, 12, 11, 9, 300],
        "Purchase_Frequency": [8, 7, 9, 8, 2],
    })


def test_dormant_customer_scores_higher_risk_than_regulars():
    df = _churn_df()
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    assert result["ok"]

    scores = result["risk_df"]["Risk score"]
    dormant_score = scores.iloc[-1]
    assert dormant_score == scores.max()
    assert dormant_score > scores.iloc[:-1].max()


def test_dormant_customer_lands_in_the_high_risk_group():
    df = _churn_df()
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    assert result["risk_df"]["Risk group"].iloc[-1] == "High"
    assert result["n_high"] >= 1


def test_loyal_customer_is_not_labelled_at_risk():
    df = _churn_df()
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    regulars = result["risk_df"].iloc[:-1]
    assert (regulars["Risk group"] != "High").all()


def test_high_spend_does_not_raise_risk():
    """Average_Order_Value has no risk keyword of its own; it must not default
    to 'higher is riskier' and flag the best customers."""
    df = pd.DataFrame({
        "Average_Order_Value": [50, 55, 52, 48, 500],
        "Days_Since_Last_Purchase": [20, 20, 20, 20, 20],
    })
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    scores = result["risk_df"]["Risk score"]
    assert scores.iloc[-1] == scores.min()


def test_heuristic_mode_discloses_that_no_outcome_was_used():
    df = _churn_df()
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    assert result["mode"] == "heuristic"
    assert any("no confirmed churn outcome" in w.lower() for w in result["limitations"])


def test_supervised_mode_used_when_a_known_outcome_is_supplied():
    rng = np.random.default_rng(11)
    n = 120
    recency = rng.integers(1, 200, n)
    df = pd.DataFrame({
        "Days_Since_Last_Purchase": recency,
        "Purchase_Frequency": rng.integers(1, 12, n),
        "Churned": np.where(recency > 100, "Yes", "No"),
    })
    result = churn_analysis.run_analysis(
        df, ["Days_Since_Last_Purchase", "Purchase_Frequency"], {}, "Churned")
    assert result["ok"]
    assert result["mode"] == "supervised"
    # The churned rows are the dormant ones, so risk must track recency upward.
    assert result["risk_df"]["Risk score"].corr(df["Days_Since_Last_Purchase"]) > 0.5


def test_requires_at_least_one_numeric_signal():
    df = pd.DataFrame({"City": ["Pune", "Delhi"]})
    result = churn_analysis.run_analysis(df, ["City"], {}, None)
    assert not result["ok"]


# ---------------------------------------------------------------------------
# "Main risk reason" must name a risk driver, never a protective factor
# ---------------------------------------------------------------------------

def test_loyal_customer_is_not_told_their_frequency_declined():
    """The strongest signal for this customer is a protective one - buying far
    more often than anyone else. Picking the largest magnitude regardless of
    sign used to report that as the reason they are at risk."""
    df = pd.DataFrame({
        "Days_Since_Last_Purchase": [10, 10, 10, 10, 10],
        "Purchase_Frequency": [50, 3, 3, 3, 3],
    })
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    loyal = result["risk_df"].iloc[0]
    assert loyal["Main risk reason"] != "purchase frequency has declined"
    assert loyal["Main risk reason"] == NO_STANDOUT_REASON


def test_reason_names_the_signal_that_actually_raises_risk():
    df = pd.DataFrame({
        "Days_Since_Last_Purchase": [10, 10, 10, 400],
        "Purchase_Frequency": [8, 8, 8, 8],
        "Customer_Service_Complaints": [0, 0, 0, 0],
    })
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    assert result["risk_df"]["Main risk reason"].iloc[-1] == "the customer has not purchased recently"


def test_complaint_driven_customer_gets_the_complaint_reason():
    df = pd.DataFrame({
        "Days_Since_Last_Purchase": [30, 30, 30, 30],
        "Purchase_Frequency": [8, 8, 8, 8],
        "Customer_Service_Complaints": [0, 0, 0, 5],
    })
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    assert result["risk_df"]["Main risk reason"].iloc[-1] == "a recent complaint was recorded"


def test_customers_with_only_protective_signals_are_told_so_honestly():
    df = pd.DataFrame({
        "Days_Since_Last_Purchase": [5, 200, 200, 200],
        "Purchase_Frequency": [40, 2, 2, 2],
    })
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    assert result["risk_df"]["Main risk reason"].iloc[0] == NO_STANDOUT_REASON


def test_placeholder_reason_never_reaches_the_headline_or_recommendations():
    df = pd.DataFrame({
        "Days_Since_Last_Purchase": [5, 200, 200, 200],
        "Purchase_Frequency": [40, 2, 2, 2],
    })
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    assert NO_STANDOUT_REASON not in result["top_reasons"]


def test_top_reasons_describe_the_high_risk_customers_not_everyone():
    """The headline says at-risk customers are at risk 'most often because
    <top reason>', so the low-risk majority must not decide what it says.

    This fixture is built so the two answers differ: complaints are the most
    common reason across the whole file, but the customers who actually reach
    the high-risk band are there because they have gone quiet.
    """
    df = pd.DataFrame({
        # Three dormant customers, driven by recency.
        "Days_Since_Last_Purchase": [400, 380, 390] + [10] * 30,
        # Thirty ordinary customers, half of them with a single complaint.
        "Customer_Service_Complaints": [0, 0, 0] + [1, 0] * 15,
    })
    result = churn_analysis.run_analysis(df, list(df.columns), {}, None)
    risk_df = result["risk_df"]

    high_risk = risk_df[risk_df["Risk group"] == "High"]
    assert len(high_risk) == 3

    population_mode = risk_df["Main risk reason"].value_counts().index[0]
    assert population_mode == "a recent complaint was recorded"

    # The headline must follow the high-risk group, not the population.
    assert result["top_reasons"][0] == "the customer has not purchased recently"
    assert result["top_reasons"][0] != population_mode


def _supervised_df(seed=23, n=150):
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 300, n)
    return pd.DataFrame({
        "Days_Since_Last_Purchase": recency,
        # Pure noise, so the model's learned sign for it is arbitrary.
        "Purchase_Frequency": rng.integers(1, 15, n),
        "Churned": np.where(recency > 150, "Yes", "No"),
    })


def _run_supervised(df):
    result = churn_analysis.run_analysis(
        df, ["Days_Since_Last_Purchase", "Purchase_Frequency"], {}, "Churned")
    assert result["mode"] == "supervised"
    return result


def test_supervised_reason_matches_the_direction_the_model_learned():
    """The phrase 'purchase frequency has declined' assumes low frequency is the
    risky end. When the model learns the opposite sign from the data, reusing
    that phrase would state the reverse of the finding - so it must not appear
    for customers whose frequency is above average."""
    df = _supervised_df()
    risk_df = _run_supervised(df)["risk_df"]

    blamed_on_frequency = risk_df["Main risk reason"] == "purchase frequency has declined"
    above_average = risk_df["Purchase_Frequency"] > risk_df["Purchase_Frequency"].mean()
    assert not (blamed_on_frequency & above_average).any()


def test_supervised_riskiest_customers_get_a_concrete_reason():
    risk_df = _run_supervised(_supervised_df())["risk_df"]
    riskiest = risk_df.nlargest(10, "Risk score")
    assert (riskiest["Main risk reason"] != NO_STANDOUT_REASON).all()


def test_supervised_dormant_customers_are_blamed_on_recency():
    risk_df = _run_supervised(_supervised_df())["risk_df"]
    dormant = risk_df.nlargest(10, "Days_Since_Last_Purchase")
    assert (dormant["Main risk reason"] == "the customer has not purchased recently").all()


# ---------------------------------------------------------------------------
# The shipped sample workbook
# ---------------------------------------------------------------------------

def test_sample_data_dormant_customers_rank_as_higher_risk():
    df = load_sample_customer_data()
    signals = ["Days_Since_Last_Purchase", "Purchase_Frequency",
               "Customer_Service_Complaints", "Delivery_Delay_Days"]
    result = churn_analysis.run_analysis(df, signals, {}, None)
    assert result["ok"]

    risk_df = result["risk_df"]
    correlation = risk_df["Risk score"].corr(risk_df["Days_Since_Last_Purchase"])
    assert correlation > 0, "risk score must rise with days since last purchase"

    most_dormant = risk_df.nlargest(50, "Days_Since_Last_Purchase")["Risk score"].mean()
    most_recent = risk_df.nsmallest(50, "Days_Since_Last_Purchase")["Risk score"].mean()
    assert most_dormant > most_recent
