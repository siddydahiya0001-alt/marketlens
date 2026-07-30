"""Routes a chosen business question + column configuration to the right analysis module.

Keeps pages/4_Results.py thin: it just calls `run(question, df, roles, config, settings)`
and renders whatever comes back with the shared result-panel components.
"""

from __future__ import annotations

from utils.constants import (
    QUESTION_SALES_DRIVERS, QUESTION_PURCHASE_LIKELIHOOD, QUESTION_SEGMENTATION,
    QUESTION_CAMPAIGN, QUESTION_CHURN, QUESTION_CUSTOMER_VALUE, QUESTION_CHANNEL,
    QUESTION_TIME_SERIES, QUESTION_EXPLORE, QUESTION_CUSTOM,
)


def run(question: str, df, roles: dict, config: dict, settings: dict) -> dict:
    threshold = settings.get("confidence_threshold", 0.05)

    if question == QUESTION_SALES_DRIVERS:
        from analyses import sales_driver_analysis
        return sales_driver_analysis.run_analysis(
            df, config.get("sales_col"), config.get("factor_cols", []), roles, threshold,
        )

    if question == QUESTION_PURCHASE_LIKELIHOOD:
        from analyses import purchase_prediction
        return purchase_prediction.run_analysis(
            df, config.get("outcome_col"), config.get("factor_cols", []), roles,
            config.get("positive_value"),
        )

    if question == QUESTION_SEGMENTATION:
        from analyses import customer_segmentation
        return customer_segmentation.run_analysis(
            df, config.get("cluster_cols", []), roles,
            n_clusters=config.get("n_clusters", 4), auto_suggest=config.get("auto_suggest", True),
        )

    if question == QUESTION_CAMPAIGN:
        from analyses import campaign_comparison
        return campaign_comparison.run_analysis(
            df, config.get("campaign_col"), config.get("outcome_col"),
            _none_if_placeholder(config.get("revenue_col")), _none_if_placeholder(config.get("cost_col")),
            config.get("control_group"), config.get("test_group"),
        )

    if question == QUESTION_CHURN:
        from analyses import churn_analysis
        return churn_analysis.run_analysis(
            df, config.get("risk_signal_cols", []), roles,
            _none_if_placeholder(config.get("known_churn_col")),
        )

    if question == QUESTION_CUSTOMER_VALUE:
        from analyses import customer_value
        return customer_value.run_analysis(
            df, config.get("customer_id_col"), config.get("revenue_col"),
            _none_if_placeholder(config.get("date_col")), _none_if_placeholder(config.get("acquisition_cost_col")),
        )

    if question == QUESTION_CHANNEL:
        from analyses import channel_performance
        return channel_performance.run_analysis(
            df, config.get("channel_col"), _none_if_placeholder(config.get("revenue_col")),
            _none_if_placeholder(config.get("cost_col")), _none_if_placeholder(config.get("outcome_col")),
        )

    if question == QUESTION_TIME_SERIES:
        from analyses import time_series_analysis
        return time_series_analysis.run_analysis(
            df, config.get("date_col"), config.get("value_col"), config.get("period", "Monthly"),
        )

    if question in (QUESTION_EXPLORE, QUESTION_CUSTOM):
        return {"ok": True, "explore_mode": True}

    return {"ok": False, "reason": "Unknown business question."}


def _none_if_placeholder(value):
    if value in (None, "(none)"):
        return None
    return value
