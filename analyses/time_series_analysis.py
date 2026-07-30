"""Module I: How are sales changing over time?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from services import validation_service as vs
from utils.constants import MIN_ROWS_FOR_FORECAST

PERIOD_FREQ = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}


def run_analysis(df: pd.DataFrame, date_col: str, value_col: str, period: str = "Monthly") -> dict:
    warnings = []

    if not date_col or not value_col or date_col not in df.columns or value_col not in df.columns:
        return {"ok": False, "reason": "Please select a date column and a value column.", "limitations": warnings}

    working = df[[date_col, value_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.dropna(subset=[date_col, value_col])

    if working.empty:
        return {"ok": False, "reason": "No valid dates were found in the selected date column.", "limitations": warnings}

    freq = PERIOD_FREQ.get(period, "M")
    series = working.set_index(date_col)[value_col].resample(freq).sum().reset_index()
    series.columns = [date_col, value_col]

    n_periods = len(series)
    warnings += vs.check_forecast_feasibility(n_periods)

    # Trend line via simple linear regression on period index
    x = np.arange(n_periods)
    if n_periods >= 2:
        slope, intercept = np.polyfit(x, series[value_col].values, 1)
    else:
        slope, intercept = 0.0, series[value_col].values[0] if n_periods else 0.0
    trend_values = slope * x + intercept
    trend_df = pd.DataFrame({date_col: series[date_col], value_col: trend_values})

    # Growth is measured from the fitted trend line's endpoints, not the raw first/last
    # period values - a single unusually high or low period at either edge would otherwise
    # distort the headline growth figure.
    trend_start, trend_end = trend_values[0], trend_values[-1]
    growth_rate_pct = ((trend_end - trend_start) / trend_start * 100) if trend_start else None

    mean_val, std_val = series[value_col].mean(), series[value_col].std()
    peaks = series[series[value_col] > mean_val + std_val] if std_val and not np.isnan(std_val) else pd.DataFrame()
    declines = series[series[value_col] < mean_val - std_val] if std_val and not np.isnan(std_val) else pd.DataFrame()

    seasonality_note = None
    if period in ("Monthly", "Weekly") and n_periods >= 24:
        working["period_of_cycle"] = working[date_col].dt.month if period == "Monthly" else working[date_col].dt.isocalendar().week
        cycle_avg = working.groupby("period_of_cycle")[value_col].mean()
        if cycle_avg.std() / (cycle_avg.mean() + 1e-9) > 0.15:
            seasonality_note = "Sales appear to follow a repeating seasonal pattern across the year."

    ma_window = min(3, n_periods) if n_periods >= 3 else None
    moving_avg = series[value_col].rolling(window=ma_window).mean() if ma_window else None

    forecast_df = None
    if n_periods >= MIN_ROWS_FOR_FORECAST:
        try:
            from statsmodels.tsa.holtwinters import Holt
            model = Holt(series[value_col].values, initialization_method="estimated").fit()
            n_forecast = max(3, n_periods // 6)
            forecast_values = model.forecast(n_forecast)
            last_date = series[date_col].iloc[-1]
            forecast_dates = pd.date_range(start=last_date, periods=n_forecast + 1, freq=freq)[1:]
            forecast_df = pd.DataFrame({date_col: forecast_dates, value_col: forecast_values})
        except Exception:
            forecast_df = None

    return {
        "ok": True,
        "series": series,
        "trend_df": trend_df,
        "forecast_df": forecast_df,
        "date_col": date_col,
        "value_col": value_col,
        "period": period,
        "n_periods": n_periods,
        "growth_rate_pct": growth_rate_pct,
        "peaks": peaks,
        "declines": declines,
        "seasonality_note": seasonality_note,
        "moving_avg": moving_avg,
        "limitations": warnings,
    }


def render(result: dict, df: pd.DataFrame, roles: dict, config: dict, settings: dict):
    import streamlit as st
    from visualisations import result_panels, charts

    if not result.get("ok"):
        st.error(result.get("reason", "This analysis could not be run with the selected columns."))
        return

    growth = result["growth_rate_pct"]
    direction = "grown" if (growth or 0) >= 0 else "declined"
    main_answer = (
        f"The underlying trend in {result['value_col']} has {direction} by about {abs(growth):.1f}% "
        f"across the {result['n_periods']} periods analysed."
        if growth is not None else f"{result['value_col']} trend has been calculated over {result['n_periods']} periods."
    )
    result_panels.render_main_answer(
        main_answer,
        result["seasonality_note"] or "No strong repeating seasonal pattern was detected in this data.",
    )
    confidence = "High confidence" if result["n_periods"] >= 12 else "Moderate confidence"
    result_panels.render_confidence(confidence)

    result_panels.render_reasoning_trail([
        f"We grouped the data into {result['n_periods']} {result['period'].lower()} periods and summed "
        f"{result['value_col']} within each one.",
        "We fit a straight line through those periods (technical name: linear trend) to separate the "
        "overall direction from period-to-period noise, rather than just comparing the first and last "
        "periods directly.",
        f"The growth figure above compares that fitted line's start and end, which is why it can differ "
        "from a simple 'first period vs last period' comparison.",
    ] + ([
        f"Because at least {MIN_ROWS_FOR_FORECAST} periods were available, we also projected "
        f"{len(result['forecast_df'])} periods forward using a trend-and-level forecasting method "
        "(technical name: Holt's exponential smoothing)."
    ] if result.get("forecast_df") is not None else []))

    fig, expl = charts.trend_with_forecast_chart(
        result["series"], result["date_col"], result["value_col"],
        result["trend_df"], result["forecast_df"],
        f"{result['value_col']} over time ({result['period']})",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(expl)

    if result["forecast_df"] is None:
        st.info("Not enough historical periods are available yet to responsibly forecast future values.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Peaks (unusually high periods)**")
        if not result["peaks"].empty:
            st.dataframe(result["peaks"], use_container_width=True, hide_index=True)
        else:
            st.caption("No standout peaks detected.")
    with c2:
        st.markdown("**Declines (unusually low periods)**")
        if not result["declines"].empty:
            st.dataframe(result["declines"], use_container_width=True, hide_index=True)
        else:
            st.caption("No standout declines detected.")

    result_panels.render_recommendations([
        "Continue monitoring the trend regularly rather than reacting to any single period."
        + (" Investigate the cause of the identified peaks and declines." if not result["peaks"].empty or not result["declines"].empty else ""),
    ])
    result_panels.render_limitations(
        result.get("limitations", []),
        do_not_conclude=["A forecast is a projection based on past patterns, not a guarantee of future results."],
    )

    st.markdown("### Download")
    from services.export_service import dataframe_to_excel_bytes
    from utils.download_helpers import lazy_download_button
    lazy_download_button(
        "Download time series data",
        lambda: dataframe_to_excel_bytes(result["series"], "Time series"),
        file_name="time_series.xlsx",
        key="time_series",
    )
