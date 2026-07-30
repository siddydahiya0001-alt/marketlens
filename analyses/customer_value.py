"""Module G: What is the value of each customer? (customer lifetime value estimation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from services import validation_service as vs
from utils.helpers import safe_divide


def run_analysis(df: pd.DataFrame, customer_id_col: str, revenue_col: str, date_col: str | None = None,
                  acquisition_cost_col: str | None = None) -> dict:
    warnings = []
    warnings += vs.check_dataset_size(df)

    if not customer_id_col or not revenue_col or customer_id_col not in df.columns or revenue_col not in df.columns:
        return {"ok": False, "reason": "Please select a customer ID column and a revenue column.", "limitations": warnings}

    working = df.copy()
    agg = {revenue_col: "sum"}
    agg_customer = working.groupby(customer_id_col).agg(**{
        "Historical value": (revenue_col, "sum"),
        "Number of purchases": (revenue_col, "count"),
    }).reset_index()

    can_project = False
    if date_col and date_col in working.columns:
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
        dates_valid = working[date_col].notna().any()
        if dates_valid:
            date_summary = working.groupby(customer_id_col)[date_col].agg(["min", "max"]).reset_index()
            date_summary.columns = [customer_id_col, "First purchase", "Last purchase"]
            agg_customer = agg_customer.merge(date_summary, on=customer_id_col, how="left")
            reference_date = working[date_col].max()
            agg_customer["Recency (days)"] = (reference_date - agg_customer["Last purchase"]).dt.days
            agg_customer["Tenure (days)"] = (agg_customer["Last purchase"] - agg_customer["First purchase"]).dt.days
            overall_span_days = (working[date_col].max() - working[date_col].min()).days

            if overall_span_days >= 60:
                can_project = True
                agg_customer["Projected annual value"] = agg_customer.apply(
                    lambda r: safe_divide(r["Historical value"], max(r["Tenure (days)"], 1)) * 365
                    if r["Tenure (days)"] and r["Tenure (days)"] > 0 else r["Historical value"],
                    axis=1,
                )
            else:
                warnings.append(
                    "The date range in this data is short, so MarketLens cannot reliably project long-term "
                    "customer value. Only historical value is shown."
                )
    else:
        warnings.append(
            "No purchase-date column was provided, so only historical customer value can be calculated. "
            "Add a date column to enable projected future value."
        )

    if acquisition_cost_col and acquisition_cost_col in working.columns:
        cost_summary = working.groupby(customer_id_col)[acquisition_cost_col].sum().reset_index()
        cost_summary.columns = [customer_id_col, "Acquisition cost"]
        agg_customer = agg_customer.merge(cost_summary, on=customer_id_col, how="left")
        agg_customer["Net value"] = agg_customer["Historical value"] - agg_customer["Acquisition cost"]

    agg_customer = agg_customer.sort_values("Historical value", ascending=False).reset_index(drop=True)

    total_value = agg_customer["Historical value"].sum()
    n_customers = len(agg_customer)
    top_20_pct_n = max(1, int(np.ceil(n_customers * 0.2)))
    top_20_value = agg_customer.head(top_20_pct_n)["Historical value"].sum()
    revenue_concentration_pct = safe_divide(top_20_value, total_value) * 100

    warnings += vs.check_dominant_category(
        pd.Series(["top20"] * top_20_pct_n + ["rest"] * (n_customers - top_20_pct_n))
    ) if revenue_concentration_pct > 90 else []

    return {
        "ok": True,
        "customer_value_df": agg_customer,
        "can_project": can_project,
        "average_value": float(agg_customer["Historical value"].mean()),
        "median_value": float(agg_customer["Historical value"].median()),
        "total_value": float(total_value),
        "revenue_concentration_pct": float(revenue_concentration_pct),
        "top_20_pct_n": top_20_pct_n,
        "n_customers": n_customers,
        "limitations": warnings,
    }


def render(result: dict, df: pd.DataFrame, roles: dict, config: dict, settings: dict):
    import streamlit as st
    from services.recommendation_engine import recommend_from_customer_value
    from visualisations import result_panels, charts
    from utils.helpers import format_currency

    if not result.get("ok"):
        st.error(result.get("reason", "This analysis could not be run with the selected columns."))
        return

    currency = settings.get("currency", "$")
    number_format = settings.get("number_format", "1,234.56")
    cv_df = result["customer_value_df"]

    main_answer = (
        f"The top 20% of customers ({result['top_20_pct_n']} of {result['n_customers']}) generate "
        f"{result['revenue_concentration_pct']:.0f}% of total value."
    )
    result_panels.render_main_answer(
        main_answer,
        "Prioritising retention for your highest-value customers protects the largest share of revenue.",
    )
    confidence = "High confidence" if result["can_project"] else "Moderate confidence"
    result_panels.render_confidence(
        confidence,
        "Based on enough purchase history to be reliable." if result["can_project"] else
        "Based on historical value only - not enough date history for a long-term projection.",
    )

    reasoning_steps = [
        f"We added up all revenue per customer across {result['n_customers']} customers to get each "
        "customer's historical value.",
    ]
    if result["can_project"]:
        reasoning_steps.append(
            "Because a purchase-date column was provided with enough history, we also estimated each "
            "customer's likely annual value by scaling their historical value to a 365-day rate based on "
            "how long they have been buying."
        )
    else:
        reasoning_steps.append(
            "No purchase-date column (or not enough date range) was available, so we could not responsibly "
            "project future value - only historical value is shown."
        )
    reasoning_steps.append(
        f"We sorted all customers by value and found that the top {result['top_20_pct_n']} customers "
        f"(20%) account for {result['revenue_concentration_pct']:.0f}% of total value."
    )
    result_panels.render_reasoning_trail(reasoning_steps)

    from visualisations.metric_cards import render_metric_row
    render_metric_row([
        {"label": "Average customer value", "value": format_currency(result["average_value"], currency, number_format)},
        {"label": "Median customer value", "value": format_currency(result["median_value"], currency, number_format)},
        {"label": "Total value", "value": format_currency(result["total_value"], currency, number_format)},
    ], columns_per_row=3)

    st.markdown("### Highest-value customers")
    st.dataframe(cv_df.head(15), use_container_width=True, hide_index=True)

    st.markdown("### Lowest-value customers")
    st.dataframe(cv_df.tail(15), use_container_width=True, hide_index=True)

    value_col_for_chart = "Projected annual value" if result["can_project"] else "Historical value"
    top_15 = cv_df.head(15).copy()
    top_15[top_15.columns[0]] = top_15[top_15.columns[0]].astype(str)
    fig, expl = charts.bar_chart(top_15, top_15.columns[0], value_col_for_chart, "Top customers by value")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(expl)

    recommendations = recommend_from_customer_value(result["revenue_concentration_pct"], settings.get("business_priority"))
    result_panels.render_recommendations(recommendations)
    result_panels.render_limitations(
        result.get("limitations", []),
        do_not_conclude=["Projected value assumes future behaviour resembles the past - unexpected changes can shift results."],
    )

    st.markdown("### Download")
    from services.export_service import dataframe_to_excel_bytes
    from utils.download_helpers import lazy_download_button
    lazy_download_button(
        "Download customer value table",
        lambda: dataframe_to_excel_bytes(cv_df, "Customer value"),
        file_name="customer_value.xlsx",
        key="customer_value",
    )
