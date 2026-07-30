"""Module E: Did a campaign perform better? (two-group comparison / hypothesis test under the hood)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

from services import validation_service as vs
from utils.helpers import safe_divide, infer_positive_value, positive_mask


def run_analysis(df: pd.DataFrame, campaign_col: str, outcome_col: str, revenue_col: str | None,
                  cost_col: str | None, control_group, test_group) -> dict:
    warnings = []
    warnings += vs.check_dataset_size(df)

    if not campaign_col or not outcome_col or control_group is None or test_group is None:
        return {"ok": False, "reason": "Please select a campaign column, outcome column, and two groups to compare.",
                "limitations": warnings}
    if control_group == test_group:
        return {"ok": False, "reason": "Please select two different groups to compare.", "limitations": warnings}

    control_df = df[df[campaign_col] == control_group]
    test_df = df[df[campaign_col] == test_group]

    if len(control_df) == 0 or len(test_df) == 0:
        return {"ok": False, "reason": "One of the selected groups has no rows.", "limitations": warnings}

    warnings += vs.check_dataset_size(control_df, min_rows=20)
    warnings += vs.check_dataset_size(test_df, min_rows=20)

    # Decide what counts as a conversion across the whole column, then slice - so both
    # groups are scored against the same rule even if one group happens to use only a
    # subset of the spellings present in the data.
    positive_value = infer_positive_value(df[outcome_col])
    converted = positive_mask(df[outcome_col], positive_value)
    control_conversions = int(converted.loc[control_df.index].sum())
    test_conversions = int(converted.loc[test_df.index].sum())
    n_control, n_test = len(control_df), len(test_df)

    control_rate = safe_divide(control_conversions, n_control)
    test_rate = safe_divide(test_conversions, n_test)
    diff = test_rate - control_rate

    count = np.array([test_conversions, control_conversions])
    nobs = np.array([n_test, n_control])
    try:
        z_stat, p_value = proportions_ztest(count, nobs)
        ci_low, ci_high = proportion_confint(test_conversions, n_test, method="wilson")
    except Exception:
        p_value, ci_low, ci_high = None, None, None

    result = {
        "ok": True,
        "campaign_col": campaign_col,
        "outcome_col": outcome_col,
        "control_group": control_group,
        "test_group": test_group,
        "n_control": n_control,
        "n_test": n_test,
        "control_conversion_rate": control_rate,
        "test_conversion_rate": test_rate,
        "conversion_diff": diff,
        "conversion_diff_pct": diff * 100,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "better_campaign": test_group if diff >= 0 else control_group,
        "limitations": warnings,
    }

    if revenue_col:
        result["control_revenue"] = float(control_df[revenue_col].sum())
        result["test_revenue"] = float(test_df[revenue_col].sum())
        result["incremental_revenue"] = result["test_revenue"] - result["control_revenue"] * safe_divide(n_test, n_control, 1)
    if cost_col:
        result["control_cost"] = float(control_df[cost_col].sum())
        result["test_cost"] = float(test_df[cost_col].sum())
    if revenue_col and cost_col:
        result["control_profit"] = result["control_revenue"] - result["control_cost"]
        result["test_profit"] = result["test_revenue"] - result["test_cost"]
        result["control_roas"] = safe_divide(result["control_revenue"], result["control_cost"])
        result["test_roas"] = safe_divide(result["test_revenue"], result["test_cost"])
        result["roas_diff"] = result["test_roas"] - result["control_roas"]

    warnings.append(vs.observational_data_warning())
    result["limitations"] = warnings
    return result


def render(result: dict, df: pd.DataFrame, roles: dict, config: dict, settings: dict):
    import streamlit as st
    from services.interpretation_service import get_interpreter
    from services.recommendation_engine import recommend_from_campaign_comparison
    from visualisations import result_panels, charts
    from utils.helpers import format_currency, format_percent

    if not result.get("ok"):
        st.error(result.get("reason", "This comparison could not be run with the selected columns."))
        return

    currency = settings.get("currency", "$")
    number_format = settings.get("number_format", "1,234.56")

    interpreter = get_interpreter()
    interpretation = interpreter.interpret_campaign_comparison(result)
    recommendations = recommend_from_campaign_comparison(
        result["better_campaign"], result.get("roas_diff"), settings.get("business_priority"),
        result.get("p_value"),
    )

    result_panels.render_main_answer(interpretation["main_answer"], interpretation["explanation"])
    result_panels.render_confidence(interpretation["confidence"])
    result_panels.render_reasoning_trail(interpretation.get("reasoning_steps", []))

    st.markdown("### Group comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{result['control_group']}** (n={result['n_control']})")
        st.metric("Conversion rate", format_percent(result["control_conversion_rate"]))
        if "control_revenue" in result:
            st.metric("Revenue", format_currency(result["control_revenue"], currency, number_format))
        if "control_cost" in result:
            st.metric("Cost", format_currency(result["control_cost"], currency, number_format))
        if "control_profit" in result:
            st.metric("Profit", format_currency(result["control_profit"], currency, number_format))
    with c2:
        st.markdown(f"**{result['test_group']}** (n={result['n_test']})")
        st.metric("Conversion rate", format_percent(result["test_conversion_rate"]))
        if "test_revenue" in result:
            st.metric("Revenue", format_currency(result["test_revenue"], currency, number_format))
        if "test_cost" in result:
            st.metric("Cost", format_currency(result["test_cost"], currency, number_format))
        if "test_profit" in result:
            st.metric("Profit", format_currency(result["test_profit"], currency, number_format))

    compare_df = pd.DataFrame({
        "Group": [result["control_group"], result["test_group"]],
        "Conversion rate": [result["control_conversion_rate"] * 100, result["test_conversion_rate"] * 100],
    })
    fig, expl = charts.bar_chart(compare_df, "Group", "Conversion rate", "Conversion rate by group")
    st.plotly_chart(fig, use_container_width=True)

    if result.get("test_revenue") is not None:
        st.caption(
            f"{result['test_group']} generated {'higher' if result['test_revenue'] >= result.get('control_revenue', 0) else 'lower'} "
            "revenue" + (f" but also had a {'higher' if result.get('test_cost', 0) >= result.get('control_cost', 0) else 'lower'} cost." if "test_cost" in result else ".")
        )

    result_panels.render_recommendations(recommendations)
    result_panels.render_limitations(
        result.get("limitations", []),
        do_not_conclude=["A higher conversion rate alone does not guarantee better profitability - check cost and revenue together."],
    )

    with st.expander("Show technical details", expanded=settings.get("show_technical", False)):
        if settings.get("show_technical"):
            st.write(f"Difference in conversion: {result['conversion_diff_pct']:.2f} percentage points")
            if result.get("p_value") is not None:
                st.write(f"P-value: {result['p_value']:.4f}")
            if result.get("ci_low") is not None:
                st.write(f"95% confidence interval for test group conversion rate: [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
            if "test_roas" in result:
                st.write(f"Control ROAS: {result['control_roas']:.2f} | Test ROAS: {result['test_roas']:.2f}")
        else:
            st.caption("Turn on 'Show technical details' in the sidebar to see statistical output.")
