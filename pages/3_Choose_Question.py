"""Page 3: Ask 'What would you like to understand?' and collect the relevant columns."""

import streamlit as st

from utils.session_state import init_session_state
from utils.sidebar import render_global_sidebar
from utils.constants import (
    BUSINESS_QUESTIONS, QUESTION_DESCRIPTIONS,
    QUESTION_SALES_DRIVERS, QUESTION_PURCHASE_LIKELIHOOD, QUESTION_SEGMENTATION,
    QUESTION_CAMPAIGN, QUESTION_CHURN, QUESTION_CUSTOMER_VALUE, QUESTION_CHANNEL,
    QUESTION_TIME_SERIES, QUESTION_EXPLORE, QUESTION_CUSTOM,
    ROLE_SALES, ROLE_REVENUE, ROLE_YES_NO, ROLE_CAMPAIGN, ROLE_COST, ROLE_PROFIT,
    ROLE_DATE, ROLE_CUSTOMER_ID, ROLE_CATEGORY, ROLE_BEHAVIOUR, ROLE_NUMBER,
    TIME_PERIODS,
)

st.set_page_config(page_title="Choose a Question - MarketLens", page_icon="❓", layout="wide")
init_session_state()
render_global_sidebar()

st.title("❓ What would you like to understand?")

if st.session_state.get("cleaned_df") is None and st.session_state.get("raw_df") is None:
    st.warning("Please upload a file first.")
    st.page_link("pages/1_Upload_and_Review.py", label="Go to Upload and Review", icon="⬅️")
    st.stop()

from utils.session_state import active_df
df = active_df()
roles = st.session_state.get("roles", {})
columns = list(df.columns)


def cols_with_role(*wanted_roles):
    return [c for c in columns if roles.get(c) in wanted_roles]


def all_columns_except(exclude):
    return [c for c in columns if c not in exclude]


question = st.radio(
    "Select one:",
    BUSINESS_QUESTIONS,
    format_func=lambda q: f"{q} — {QUESTION_DESCRIPTIONS[q]}",
    index=BUSINESS_QUESTIONS.index(st.session_state["business_question"]) if st.session_state.get("business_question") in BUSINESS_QUESTIONS else 0,
)
st.session_state["business_question"] = question

st.markdown("---")
st.markdown(f"### Set up: {question}")

config = {}

if question == QUESTION_SALES_DRIVERS:
    sales_default = cols_with_role(ROLE_SALES, ROLE_REVENUE)
    config["sales_col"] = st.selectbox("Which column represents sales?", columns,
                                        index=columns.index(sales_default[0]) if sales_default else 0)
    factor_candidates = all_columns_except([config["sales_col"]])
    config["factor_cols"] = st.multiselect("Which factors should be tested?", factor_candidates,
                                            default=[c for c in factor_candidates if roles.get(c) not in (ROLE_CUSTOMER_ID,)][:6])
    config["excluded_cols"] = st.multiselect("Which columns should be excluded?", factor_candidates)
    config["factor_cols"] = [c for c in config["factor_cols"] if c not in config["excluded_cols"]]

elif question == QUESTION_PURCHASE_LIKELIHOOD:
    outcome_default = cols_with_role(ROLE_YES_NO)
    config["outcome_col"] = st.selectbox("Which column shows whether the customer purchased?", columns,
                                          index=columns.index(outcome_default[0]) if outcome_default else 0)
    factor_candidates = all_columns_except([config["outcome_col"]])
    config["factor_cols"] = st.multiselect("Which factors should be used?", factor_candidates,
                                            default=[c for c in factor_candidates if roles.get(c) not in (ROLE_CUSTOMER_ID,)][:6])
    outcome_values = df[config["outcome_col"]].dropna().unique().tolist()
    if outcome_values:
        config["positive_value"] = st.selectbox("Which result represents a successful purchase?", outcome_values)

elif question == QUESTION_SEGMENTATION:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    behaviour_default = cols_with_role(ROLE_BEHAVIOUR, ROLE_NUMBER, ROLE_SALES, ROLE_REVENUE)
    config["cluster_cols"] = st.multiselect("Which columns describe customer behaviour?", numeric_cols,
                                             default=[c for c in behaviour_default if c in numeric_cols][:5] or numeric_cols[:5])
    config["n_clusters"] = st.slider("Number of customer groups (leave MarketLens to suggest, or override)",
                                      2, 8, 4)
    config["auto_suggest"] = st.checkbox("Let MarketLens suggest the best number of groups", value=True)

elif question == QUESTION_CAMPAIGN:
    campaign_default = cols_with_role(ROLE_CAMPAIGN)
    config["campaign_col"] = st.selectbox("Campaign column", columns,
                                           index=columns.index(campaign_default[0]) if campaign_default else 0)
    outcome_default = cols_with_role(ROLE_YES_NO)
    config["outcome_col"] = st.selectbox("Outcome column (e.g. purchased)", columns,
                                          index=columns.index(outcome_default[0]) if outcome_default else 0)
    revenue_default = cols_with_role(ROLE_REVENUE, ROLE_SALES)
    config["revenue_col"] = st.selectbox("Revenue column (optional)", ["(none)"] + columns,
                                          index=(columns.index(revenue_default[0]) + 1) if revenue_default else 0)
    cost_default = cols_with_role(ROLE_COST)
    config["cost_col"] = st.selectbox("Cost column (optional)", ["(none)"] + columns,
                                       index=(columns.index(cost_default[0]) + 1) if cost_default else 0)
    campaign_values = df[config["campaign_col"]].dropna().unique().tolist()
    if len(campaign_values) >= 2:
        config["control_group"] = st.selectbox("Control / baseline group", campaign_values, index=0)
        config["test_group"] = st.selectbox("Test campaign", campaign_values,
                                             index=1 if len(campaign_values) > 1 else 0)

elif question == QUESTION_CHURN:
    behaviour_cols = all_columns_except([])
    outcome_default = cols_with_role(ROLE_YES_NO)
    config["risk_signal_cols"] = st.multiselect(
        "Which columns might signal a customer stopping purchases? (e.g. recency, complaints, delays)",
        behaviour_cols, default=[c for c in behaviour_cols if roles.get(c) in (ROLE_BEHAVIOUR, ROLE_NUMBER)][:6])
    if outcome_default:
        config["known_churn_col"] = st.selectbox(
            "If you already know which customers churned, select that column (optional)",
            ["(none)"] + outcome_default)
    else:
        config["known_churn_col"] = "(none)"

elif question == QUESTION_CUSTOMER_VALUE:
    id_default = cols_with_role(ROLE_CUSTOMER_ID)
    config["customer_id_col"] = st.selectbox("Customer ID column", columns,
                                              index=columns.index(id_default[0]) if id_default else 0)
    revenue_default = cols_with_role(ROLE_REVENUE, ROLE_SALES)
    config["revenue_col"] = st.selectbox("Revenue column", columns,
                                          index=columns.index(revenue_default[0]) if revenue_default else 0)
    date_default = cols_with_role(ROLE_DATE)
    config["date_col"] = st.selectbox("Purchase date column (optional)", ["(none)"] + columns,
                                       index=(columns.index(date_default[0]) + 1) if date_default else 0)
    cost_default = cols_with_role(ROLE_COST)
    config["acquisition_cost_col"] = st.selectbox("Acquisition cost column (optional)", ["(none)"] + columns,
                                                    index=(columns.index(cost_default[0]) + 1) if cost_default else 0)

elif question == QUESTION_CHANNEL:
    channel_default = cols_with_role(ROLE_CAMPAIGN)
    config["channel_col"] = st.selectbox("Marketing channel column", columns,
                                          index=columns.index(channel_default[0]) if channel_default else 0)
    revenue_default = cols_with_role(ROLE_REVENUE, ROLE_SALES)
    config["revenue_col"] = st.selectbox("Revenue column", ["(none)"] + columns,
                                          index=(columns.index(revenue_default[0]) + 1) if revenue_default else 0)
    cost_default = cols_with_role(ROLE_COST)
    config["cost_col"] = st.selectbox("Cost column", ["(none)"] + columns,
                                       index=(columns.index(cost_default[0]) + 1) if cost_default else 0)
    outcome_default = cols_with_role(ROLE_YES_NO)
    config["outcome_col"] = st.selectbox("Conversion / outcome column (optional)", ["(none)"] + columns,
                                          index=(columns.index(outcome_default[0]) + 1) if outcome_default else 0)

elif question == QUESTION_TIME_SERIES:
    date_default = cols_with_role(ROLE_DATE)
    config["date_col"] = st.selectbox("Date column", columns,
                                       index=columns.index(date_default[0]) if date_default else 0)
    sales_default = cols_with_role(ROLE_SALES, ROLE_REVENUE)
    config["value_col"] = st.selectbox("Value to track over time (e.g. sales)", columns,
                                        index=columns.index(sales_default[0]) if sales_default else 0)
    config["period"] = st.selectbox("Time period", TIME_PERIODS, index=2)

elif question == QUESTION_EXPLORE:
    st.info("Head to the Results page to build charts freely from any columns.")

elif question == QUESTION_CUSTOM:
    st.info("Head to the Results page to build a custom comparison or chart.")

st.session_state["question_config"] = config

st.markdown("---")
st.page_link("pages/4_Results.py", label="Continue to Results →", icon="➡️")
