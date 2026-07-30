"""Module H: Which marketing channel performs best?"""

from __future__ import annotations

import pandas as pd

from services import validation_service as vs
from utils.helpers import safe_divide, infer_positive_value, positive_mask


def run_analysis(df: pd.DataFrame, channel_col: str, revenue_col: str | None, cost_col: str | None,
                  outcome_col: str | None) -> dict:
    warnings = []
    warnings += vs.check_dataset_size(df)

    if not channel_col or channel_col not in df.columns:
        return {"ok": False, "reason": "Please select a marketing channel column.", "limitations": warnings}

    agg_dict = {"Records": (channel_col, "count")}
    if revenue_col and revenue_col in df.columns:
        agg_dict["Revenue"] = (revenue_col, "sum")
    if cost_col and cost_col in df.columns:
        agg_dict["Cost"] = (cost_col, "sum")

    channel_summary = df.groupby(channel_col).agg(**agg_dict).reset_index()

    if outcome_col and outcome_col in df.columns:
        positive_value = infer_positive_value(df[outcome_col]) if df[outcome_col].notna().any() else None
        converted = positive_mask(df[outcome_col], positive_value)
        conv = converted.groupby(df[channel_col]).apply(
            lambda s: safe_divide(s.sum(), len(s))
        ).reset_index()
        conv.columns = [channel_col, "Conversion rate"]
        channel_summary = channel_summary.merge(conv, on=channel_col, how="left")

    if "Revenue" in channel_summary.columns and "Cost" in channel_summary.columns:
        channel_summary["Profit"] = channel_summary["Revenue"] - channel_summary["Cost"]
        channel_summary["ROAS"] = channel_summary.apply(lambda r: safe_divide(r["Revenue"], r["Cost"]), axis=1)
        channel_summary["Cost per acquisition"] = channel_summary.apply(
            lambda r: safe_divide(r["Cost"], r["Records"]), axis=1,
        )

    # Simple composite score for ranking: prioritise profit/ROAS if available, else revenue, else conversion
    if "ROAS" in channel_summary.columns:
        channel_summary["_rank_score"] = channel_summary["ROAS"].rank(ascending=False)
    elif "Revenue" in channel_summary.columns:
        channel_summary["_rank_score"] = channel_summary["Revenue"].rank(ascending=False)
    elif "Conversion rate" in channel_summary.columns:
        channel_summary["_rank_score"] = channel_summary["Conversion rate"].rank(ascending=False)
    else:
        channel_summary["_rank_score"] = channel_summary["Records"].rank(ascending=False)

    channel_summary = channel_summary.sort_values("_rank_score").reset_index(drop=True)
    best_channel = channel_summary.iloc[0][channel_col]

    # Sample size disparity check
    record_counts = channel_summary["Records"]
    if record_counts.max() > 0 and safe_divide(record_counts.min(), record_counts.max()) < 0.1:
        warnings.append("Channel sample sizes are very different, so comparisons between the smallest and largest channels are less reliable.")

    if not cost_col:
        warnings.append("No cost column was provided, so return on ad spend and profit cannot be calculated.")
    if not outcome_col:
        warnings.append("No conversion/outcome column was provided, so conversion rate cannot be compared.")
    warnings.append(
        "This comparison does not account for customers who were influenced by more than one channel "
        "(channel overlap), so attribution may be incomplete."
    )

    channel_summary = channel_summary.drop(columns=["_rank_score"])

    return {
        "ok": True,
        "channel_summary": channel_summary,
        "channel_col": channel_col,
        "best_channel": best_channel,
        "limitations": warnings,
    }


def render(result: dict, df: pd.DataFrame, roles: dict, config: dict, settings: dict):
    import streamlit as st
    from services.recommendation_engine import recommend_from_channel_performance
    from visualisations import result_panels, charts
    from utils.helpers import format_currency

    if not result.get("ok"):
        st.error(result.get("reason", "This analysis could not be run with the selected columns."))
        return

    summary = result["channel_summary"]
    result_panels.render_main_answer(
        f"'{result['best_channel']}' currently shows the strongest overall performance among the compared channels.",
        "Channels are ranked using return on spend where available, or revenue and conversion otherwise.",
    )
    result_panels.render_confidence("Moderate confidence", "Based on observed data, not a controlled experiment across channels.")

    ranking_metric = "return on ad spend (revenue earned per unit spent)" if "ROAS" in summary.columns else (
        "total revenue" if "Revenue" in summary.columns else (
            "conversion rate" if "Conversion rate" in summary.columns else "number of records"
        )
    )
    n_channels = len(summary)
    result_panels.render_reasoning_trail([
        f"We grouped the data by {result['channel_col']} into {n_channels} channels and totalled revenue, "
        "cost and conversions within each group.",
        f"Because {ranking_metric} was available, we ranked channels by that measure - it is the closest "
        "single number to 'value for money' that the data supports.",
        f"'{result['best_channel']}' ranked highest on {ranking_metric}, which is why it is named as the "
        "strongest performer.",
    ])

    st.markdown("### Channel ranking")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    value_col = "ROAS" if "ROAS" in summary.columns else ("Revenue" if "Revenue" in summary.columns else "Records")
    fig, expl = charts.bar_chart(summary, result["channel_col"], value_col, f"{value_col} by channel")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(expl)

    recommendations = recommend_from_channel_performance(result["best_channel"], settings.get("business_priority"))
    result_panels.render_recommendations(recommendations)
    result_panels.render_limitations(
        result.get("limitations", []),
        do_not_conclude=["This does not prove one channel directly caused better results - customer mix may differ by channel."],
    )

    st.markdown("### Download")
    from services.export_service import dataframe_to_excel_bytes
    from utils.download_helpers import lazy_download_button
    lazy_download_button(
        "Download channel comparison",
        lambda: dataframe_to_excel_bytes(summary, "Channel performance"),
        file_name="channel_performance.xlsx",
        key="channel_performance",
    )
