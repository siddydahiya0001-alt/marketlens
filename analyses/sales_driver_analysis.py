"""Module B: What is influencing sales? (linear regression under the hood)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from services import validation_service as vs
from utils.helpers import count_outliers
from utils.terminology import strength_from_effect_size


def _encode_features(df: pd.DataFrame, factor_cols: list) -> pd.DataFrame:
    """One-hot encode categorical factors, pass numeric factors through."""
    frame = df[factor_cols].copy()
    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in frame.columns if c not in numeric_cols]

    for col in numeric_cols:
        frame[col] = frame[col].fillna(frame[col].median())
    for col in categorical_cols:
        frame[col] = frame[col].fillna(frame[col].mode().iloc[0] if not frame[col].mode().empty else "Unknown")

    encoded = pd.get_dummies(frame, columns=categorical_cols, drop_first=True)
    # Guard against columns with a single unique value after encoding (no variance)
    encoded = encoded.loc[:, encoded.nunique(dropna=False) > 1]
    return encoded


def run_analysis(df: pd.DataFrame, sales_col: str, factor_cols: list, roles: dict,
                  confidence_threshold: float = 0.05) -> dict:
    warnings = []
    warnings += vs.check_dataset_size(df)

    clean_factors, id_warnings = vs.clean_predictor_list(factor_cols, roles)
    warnings += id_warnings
    warnings += vs.check_date_used_as_number(factor_cols, roles)

    if sales_col not in df.columns or not clean_factors:
        return {
            "ok": False,
            "reason": "Please select a sales column and at least one factor to test.",
            "limitations": warnings,
        }

    working = df[[sales_col] + clean_factors].dropna(subset=[sales_col]).copy()
    warnings += vs.check_missing_data(working, clean_factors)

    X = _encode_features(working, clean_factors)
    y = working[sales_col].astype(float)

    # align rows (X may have fewer if some factor rows were all-NaN)
    valid_idx = X.index.intersection(y.index)
    X, y = X.loc[valid_idx], y.loc[valid_idx]

    if X.shape[1] == 0 or len(X) < 5:
        return {"ok": False, "reason": "Not enough usable data to run this analysis.", "limitations": warnings}

    n_outliers = count_outliers(y)
    if n_outliers > 0:
        warnings.append(f"The '{sales_col}' column contains {n_outliers} unusually high values, which can pull the results in their direction.")

    X_std = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    X_with_const = sm.add_constant(X_std, has_constant="add")

    model = sm.OLS(y, X_with_const).fit()

    r_squared = float(model.rsquared)
    adj_r_squared = float(model.rsquared_adj)
    model_p_value = float(model.f_pvalue) if not np.isnan(model.f_pvalue) else None

    coef_table = pd.DataFrame({
        "Factor": model.params.index,
        "Coefficient": model.params.values,
        "P-value": model.pvalues.values,
        "CI lower": model.conf_int()[0].values,
        "CI upper": model.conf_int()[1].values,
    })
    coef_table = coef_table[coef_table["Factor"] != "const"].reset_index(drop=True)

    vif_table = _compute_vif(X_with_const)
    warnings += vs.check_multicollinearity(vif_table)
    warnings += vs.check_outlier_influence(working, X.select_dtypes(include="number").columns.tolist())

    ranked_factors = []
    for _, row in coef_table.iterrows():
        strength = strength_from_effect_size(row["Coefficient"])
        direction = "positive" if row["Coefficient"] > 0 else "negative"
        ranked_factors.append({
            "name": row["Factor"],
            "direction": direction,
            "strength": strength,
            "score": abs(row["Coefficient"]),
            "p_value": row["P-value"],
            "significant": row["P-value"] < confidence_threshold,
        })
    ranked_factors.sort(key=lambda f: f["score"], reverse=True)

    residuals = model.resid.values
    fitted = model.fittedvalues.values

    # Unstandardized (raw-unit) coefficients and per-factor stats, for the what-if simulator.
    # The model was fit on standardized X, so a raw-unit effect is standardized_coef / std(X).
    # Only genuinely numeric input columns get a raw slope - one-hot encoded categories don't
    # have a meaningful "increase by one unit" interpretation.
    numeric_input_factors = [c for c in clean_factors if c in working.columns and pd.api.types.is_numeric_dtype(working[c])]
    factor_stats = {}
    raw_coefficients = {}
    for col in numeric_input_factors:
        if col not in X.columns:
            continue
        std = float(X[col].std(ddof=0))
        if std <= 0:
            continue
        standardized_coef = model.params.get(col)
        if standardized_coef is None:
            continue
        factor_stats[col] = {"mean": float(X[col].mean()), "std": std}
        raw_coefficients[col] = float(standardized_coef) / std

    return {
        "ok": True,
        "target_label": sales_col,
        "sales_col": sales_col,
        "factor_cols": clean_factors,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "model_p_value": model_p_value,
        "ranked_factors": ranked_factors,
        "coef_table": coef_table,
        "vif_table": vif_table,
        "residuals": residuals,
        "fitted": fitted,
        "n_obs": int(model.nobs),
        "limitations": warnings,
        "statsmodels_summary": model.summary().as_text(),
        "outcome_mean": float(y.mean()),
        "factor_stats": factor_stats,
        "raw_coefficients": raw_coefficients,
        "whatif_factors": list(factor_stats.keys()),
    }


def simulate_change(result: dict, factor: str, direction: str, unit: str, amount: float) -> dict:
    """Project the effect of increasing/decreasing a numeric factor, holding other factors constant.

    unit: "absolute" (raw units, e.g. currency or count), "percentage" (% of the factor's
    current average), or "ratio" (multiplier, e.g. 1.2 to scale the average by 1.2x).
    direction: "increase" or "decrease".
    """
    stats = result.get("factor_stats", {}).get(factor)
    raw_coef = result.get("raw_coefficients", {}).get(factor)
    if stats is None or raw_coef is None:
        return {"ok": False, "reason": f"'{factor}' cannot be used in a what-if scenario."}
    if amount is None or amount < 0:
        return {"ok": False, "reason": "Please enter a positive amount."}

    current_mean = stats["mean"]
    sign = 1 if direction == "increase" else -1

    if unit == "absolute":
        delta_x = sign * amount
    elif unit == "percentage":
        delta_x = sign * current_mean * (amount / 100.0)
    elif unit == "ratio":
        if amount <= 0:
            return {"ok": False, "reason": "Ratio must be a positive number."}
        delta_x = current_mean * (amount - 1) if direction == "increase" else current_mean * (1 / amount - 1)
    else:
        return {"ok": False, "reason": f"Unknown unit '{unit}'."}

    new_factor_value = current_mean + delta_x
    predicted_outcome_change = raw_coef * delta_x
    current_outcome_mean = result["outcome_mean"]
    new_outcome_mean = current_outcome_mean + predicted_outcome_change

    return {
        "ok": True,
        "factor": factor,
        "direction": direction,
        "unit": unit,
        "amount": amount,
        "current_factor_value": current_mean,
        "new_factor_value": new_factor_value,
        "delta_x": delta_x,
        "current_outcome_mean": current_outcome_mean,
        "new_outcome_mean": new_outcome_mean,
        "predicted_outcome_change": predicted_outcome_change,
    }


def render(result: dict, df: pd.DataFrame, roles: dict, config: dict, settings: dict):
    import streamlit as st
    from services.interpretation_service import get_interpreter
    from services.recommendation_engine import recommend_from_sales_drivers
    from services import validation_service as vs
    from visualisations import charts, result_panels
    from utils.terminology import label as term_label

    if not result.get("ok"):
        st.error(result.get("reason", "This analysis could not be run with the selected columns."))
        return

    interpreter = get_interpreter()
    interpretation = interpreter.interpret_regression(result)
    recommendations = recommend_from_sales_drivers(result["ranked_factors"], settings.get("business_priority"))

    ranking_fig, ranking_explanation = charts.horizontal_ranking_chart(
        result["ranked_factors"][:10], title="What is influencing sales?",
    )

    do_not_conclude = [
        "This does not prove that any single factor directly causes higher or lower sales.",
        vs.correlation_causation_warning(),
    ]
    additional_data = [
        "More historical rows would make the ranking of factors more stable.",
        "Additional context such as seasonality or competitor activity could improve the explanation.",
    ]

    result_panels.render_result_structure({
        "main_answer": interpretation["main_answer"],
        "explanation": interpretation["explanation"],
        "confidence": interpretation["confidence"],
        "confidence_sentence": interpretation["confidence_sentence"],
        "reasoning_steps": interpretation.get("reasoning_steps", []),
        "ranking_fig": ranking_fig,
        "ranking_explanation": ranking_explanation,
        "recommendations": recommendations,
        "limitations": result.get("limitations", []),
        "do_not_conclude": do_not_conclude,
        "additional_data_suggestions": additional_data,
        "technical_render_fn": lambda: _render_technical(result, df, settings),
    }, settings.get("show_technical", False))

    _render_whatif(result, roles, settings)

    st.markdown("### Download")
    from services.export_service import dataframe_to_excel_bytes
    from utils.download_helpers import lazy_download_button
    lazy_download_button(
        "Download factor ranking as Excel",
        lambda: dataframe_to_excel_bytes(result["coef_table"], "Sales drivers"),
        file_name="sales_driver_results.xlsx",
        key="sales_driver_results",
    )


UNIT_LABELS = {
    "absolute": "Absolute amount",
    "percentage": "Percentage change",
    "ratio": "Ratio (multiplier)",
}
CURRENCY_ROLES = {"Revenue", "Sales", "Cost", "Profit"}


def _render_whatif(result: dict, roles: dict, settings: dict):
    import streamlit as st
    from utils.helpers import format_currency, format_number

    whatif_factors = result.get("whatif_factors", [])
    if not whatif_factors:
        return

    currency = settings.get("currency", "$")
    number_format = settings.get("number_format", "1,234.56")
    sales_col = result["sales_col"]
    outcome_is_currency = roles.get(sales_col) in CURRENCY_ROLES

    def _fmt_outcome(value):
        return format_currency(value, currency, number_format) if outcome_is_currency else format_number(value, number_format, decimals=1)

    def _fmt_factor(name, value):
        if roles.get(name) in CURRENCY_ROLES:
            return format_currency(value, currency, number_format)
        return format_number(value, number_format, decimals=1)

    st.markdown("### Try a what-if scenario")
    st.caption(
        f"See the projected effect on {sales_col} if you increase or decrease a factor - based on a "
        "simple straight-line projection that holds every other factor at its current average."
    )

    c1, c2, c3, c4 = st.columns([1.3, 1, 1.1, 1])
    with c1:
        factor = st.selectbox("Factor", whatif_factors, key="whatif_factor")
    with c2:
        direction = st.selectbox("Direction", ["Increase", "Decrease"], key="whatif_direction")
    with c3:
        unit_label = st.selectbox("Specify change as", list(UNIT_LABELS.values()), key="whatif_unit")
        unit = next(key for key, label in UNIT_LABELS.items() if label == unit_label)
    with c4:
        default_amount = {"absolute": 1.0, "percentage": 10.0, "ratio": 1.1}[unit]
        amount_label = {
            "absolute": f"By how much ({currency if roles.get(factor) in CURRENCY_ROLES else 'units'})",
            "percentage": "By what % of its average",
            "ratio": "Multiplier (e.g. 1.2 = 1.2x)",
        }[unit]
        amount = st.number_input(amount_label, min_value=0.0, value=default_amount, key=f"whatif_amount_{unit}")

    scenario = simulate_change(result, factor, direction.lower(), unit, amount)

    if not scenario.get("ok"):
        st.warning(scenario.get("reason", "This scenario could not be calculated."))
        return

    change_text = "increases" if direction == "Increase" else "decreases"
    direction_word = "up" if scenario["predicted_outcome_change"] >= 0 else "down"

    st.markdown(
        f'<div class="ml-card ml-recommendation">'
        f'<div class="ml-card-body">📊 If <b>{factor}</b> {change_text} from '
        f'{_fmt_factor(factor, scenario["current_factor_value"])} to '
        f'{_fmt_factor(factor, scenario["new_factor_value"])}, <b>{sales_col}</b> is projected to move '
        f'<b>{direction_word}</b> by about {_fmt_outcome(abs(scenario["predicted_outcome_change"]))} - '
        f'from {_fmt_outcome(scenario["current_outcome_mean"])} to '
        f'{_fmt_outcome(scenario["new_outcome_mean"])} on average.</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This is a simple linear projection based on association, not a guarantee - it assumes the "
        "relationship stays the same at this new level and that every other factor stays unchanged. "
        "Treat it as a rough guide, especially for large changes far from the data you have."
    )


def _render_technical(result: dict, df: pd.DataFrame, settings: dict):
    import streamlit as st
    from visualisations import charts

    st.write(f"**Model:** Linear regression (Ordinary Least Squares)")
    c1, c2, c3 = st.columns(3)
    c1.metric("R-squared", f"{result['r_squared']:.3f}")
    c2.metric("Adjusted R-squared", f"{result['adj_r_squared']:.3f}")
    c3.metric("Observations", result["n_obs"])

    st.markdown("**Coefficients**")
    st.dataframe(result["coef_table"].style.format({
        "Coefficient": "{:.3f}", "P-value": "{:.4f}", "CI lower": "{:.3f}", "CI upper": "{:.3f}",
    }), use_container_width=True)

    if not result["vif_table"].empty:
        st.markdown("**Multicollinearity check (VIF)**")
        st.dataframe(result["vif_table"].style.format({"VIF": "{:.2f}"}), use_container_width=True)

    st.markdown("**Residual chart**")
    resid_fig, resid_explanation = charts.residual_chart(result["fitted"], result["residuals"])
    st.plotly_chart(resid_fig, use_container_width=True)
    st.caption(resid_explanation)

    with st.expander("Full statsmodels output"):
        st.text(result["statsmodels_summary"])


def _compute_vif(X_with_const: pd.DataFrame) -> pd.DataFrame:
    try:
        cols = [c for c in X_with_const.columns if c != "const"]
        if len(cols) < 2:
            return pd.DataFrame(columns=["Factor", "VIF"])
        vif_values = []
        for i, col in enumerate(X_with_const.columns):
            if col == "const":
                continue
            vif_values.append({
                "Factor": col,
                "VIF": float(variance_inflation_factor(X_with_const.values, X_with_const.columns.get_loc(col))),
            })
        return pd.DataFrame(vif_values)
    except Exception:
        return pd.DataFrame(columns=["Factor", "VIF"])
