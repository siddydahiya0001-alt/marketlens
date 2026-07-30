import numpy as np
import pandas as pd
import pytest

from analyses import sales_driver_analysis, purchase_prediction, customer_segmentation, campaign_comparison
from analyses.sales_driver_analysis import simulate_change
from services.interpretation_service import get_interpreter
from utils.terminology import confidence_from_p_value, r_squared_sentence, plain


def _regression_df(n=200, seed=1):
    rng = np.random.default_rng(seed)
    repeat_purchases = rng.normal(5, 2, n)
    discount = rng.normal(10, 5, n)
    noise = rng.normal(0, 1, n)
    sales = 50 + 8 * repeat_purchases + 2 * discount + noise * 5
    return pd.DataFrame({
        "sales": sales,
        "repeat_purchases": repeat_purchases,
        "discount": discount,
        "customer_id": [f"C{i}" for i in range(n)],
    })


def test_sales_driver_analysis_finds_strong_positive_factor():
    df = _regression_df()
    result = sales_driver_analysis.run_analysis(df, "sales", ["repeat_purchases", "discount"], {}, 0.05)
    assert result["ok"]
    assert result["r_squared"] > 0.5
    top_factor_names = [f["name"] for f in result["ranked_factors"][:2]]
    assert "repeat_purchases" in top_factor_names


def test_sales_driver_analysis_excludes_id_columns():
    df = _regression_df()
    result = sales_driver_analysis.run_analysis(
        df, "sales", ["repeat_purchases", "discount", "customer_id"], {"customer_id": "Customer ID"}, 0.05,
    )
    assert result["ok"]
    assert "customer_id" not in [f["name"] for f in result["ranked_factors"]]
    assert any("customer id" in w.lower() for w in result["limitations"])


def test_sales_driver_analysis_requires_factors():
    df = _regression_df()
    result = sales_driver_analysis.run_analysis(df, "sales", [], {}, 0.05)
    assert not result["ok"]


def _classification_df(n=300, seed=2):
    rng = np.random.default_rng(seed)
    visits = rng.poisson(4, n)
    recency = rng.exponential(30, n)
    score = 0.5 * visits - 0.05 * recency + rng.normal(0, 1, n)
    purchased = np.where(score > np.median(score), "Yes", "No")
    return pd.DataFrame({"visits": visits, "recency": recency, "purchased": purchased})


def test_purchase_prediction_runs_and_scores_reasonably():
    df = _classification_df()
    result = purchase_prediction.run_analysis(df, "purchased", ["visits", "recency"], {}, positive_value="Yes")
    assert result["ok"]
    assert 0 <= result["accuracy"] <= 1
    assert result["predictions_df"]["Purchase probability"].between(0, 1).all()
    assert set(result["predictions_df"]["Group"].dropna().unique()).issubset({"Low", "Medium", "High"})


def _segmentation_df(n=150, seed=3):
    rng = np.random.default_rng(seed)
    group_a = rng.normal([10, 10], 1, (n // 2, 2))
    group_b = rng.normal([50, 50], 1, (n // 2, 2))
    data = np.vstack([group_a, group_b])
    return pd.DataFrame({"spend": data[:, 0], "frequency": data[:, 1]})


def test_customer_segmentation_finds_clusters():
    df = _segmentation_df()
    result = customer_segmentation.run_analysis(df, ["spend", "frequency"], {}, n_clusters=2, auto_suggest=False)
    assert result["ok"]
    assert result["n_groups"] == 2
    assert result["silhouette_score"] > 0.5
    assert len(result["group_summaries"]) == 2
    assert "assigned_df" in result and "Cluster" in result["assigned_df"].columns


def _campaign_df(n=400, seed=4):
    rng = np.random.default_rng(seed)
    campaign = rng.choice(["A", "B"], n)
    base_prob = np.where(campaign == "B", 0.5, 0.3)
    purchased = np.where(rng.random(n) < base_prob, "Yes", "No")
    revenue = np.where(purchased == "Yes", rng.normal(500, 50, n), 0)
    cost = rng.normal(50, 5, n)
    return pd.DataFrame({"campaign": campaign, "purchased": purchased, "revenue": revenue, "cost": cost})


def test_campaign_comparison_detects_higher_conversion():
    df = _campaign_df()
    result = campaign_comparison.run_analysis(df, "campaign", "purchased", "revenue", "cost", "A", "B")
    assert result["ok"]
    assert result["test_conversion_rate"] > result["control_conversion_rate"]
    assert result["better_campaign"] == "B"
    assert result["p_value"] is not None


def test_campaign_comparison_requires_two_distinct_groups():
    df = _campaign_df()
    result = campaign_comparison.run_analysis(df, "campaign", "purchased", "revenue", "cost", "A", "A")
    assert not result["ok"]


def test_plain_language_interpretation_regression():
    interpreter = get_interpreter()
    result = interpreter.interpret_regression({
        "r_squared": 0.68, "target_label": "sales", "n_obs": 200,
        "ranked_factors": [{"name": "repeat_purchases", "direction": "positive",
                             "strength": "Strong", "score": 0.6, "significant": True}],
        "model_p_value": 0.0001,
    })
    assert "68%" in result["explanation"]
    assert result["confidence"] == "Very high confidence"
    assert result["reasoning_steps"]
    assert "repeat_purchases" in result["main_answer"]


def test_plain_language_interpretation_regression_hedges_when_not_significant():
    interpreter = get_interpreter()
    result = interpreter.interpret_regression({
        "r_squared": 0.02, "target_label": "sales", "n_obs": 40,
        "ranked_factors": [{"name": "weak_factor", "direction": "positive",
                             "strength": "Very weak", "score": 0.05, "significant": False}],
        "model_p_value": 0.8,
    })
    assert "none of the tested factors" in result["main_answer"].lower()
    assert result["confidence"] == "Not enough evidence"


def test_confidence_from_p_value_thresholds():
    assert confidence_from_p_value(0.0001) == "Very high confidence"
    assert confidence_from_p_value(0.005) == "High confidence"
    assert confidence_from_p_value(0.03) == "Moderate confidence"
    assert confidence_from_p_value(0.08) == "Low confidence"
    assert confidence_from_p_value(0.5) == "Not enough evidence"


def test_r_squared_sentence_uses_plain_language():
    sentence = r_squared_sentence(0.68, "sales")
    assert "68%" in sentence
    assert "R-squared" not in sentence


def test_plain_dictionary_has_no_technical_leakage_by_default():
    assert plain("Linear regression") == "Find what influences a number"
    assert plain("P-value") == "Evidence that the pattern is not random"


def _known_slope_df(n=300, seed=7, slope=10.0, noise=1.0):
    rng = np.random.default_rng(seed)
    visits = rng.normal(5, 2, n)
    sales = 100 + slope * visits + rng.normal(0, noise, n)
    return pd.DataFrame({"sales": sales, "visits": visits})


def test_whatif_factors_exposed_for_numeric_predictors():
    df = _known_slope_df()
    result = sales_driver_analysis.run_analysis(df, "sales", ["visits"], {}, 0.05)
    assert result["ok"]
    assert "visits" in result["whatif_factors"]
    assert result["raw_coefficients"]["visits"] == pytest.approx(10.0, abs=1.0)


def test_simulate_change_absolute_matches_raw_coefficient():
    df = _known_slope_df()
    result = sales_driver_analysis.run_analysis(df, "sales", ["visits"], {}, 0.05)
    raw_coef = result["raw_coefficients"]["visits"]

    scenario = simulate_change(result, "visits", "increase", "absolute", 2.0)
    assert scenario["ok"]
    assert scenario["delta_x"] == pytest.approx(2.0)
    assert scenario["predicted_outcome_change"] == pytest.approx(raw_coef * 2.0)
    assert scenario["new_outcome_mean"] == pytest.approx(result["outcome_mean"] + raw_coef * 2.0)


def test_simulate_change_percentage_scales_from_current_average():
    df = _known_slope_df()
    result = sales_driver_analysis.run_analysis(df, "sales", ["visits"], {}, 0.05)
    current_mean = result["factor_stats"]["visits"]["mean"]

    scenario = simulate_change(result, "visits", "increase", "percentage", 10.0)
    assert scenario["ok"]
    assert scenario["delta_x"] == pytest.approx(current_mean * 0.10)


def test_simulate_change_ratio_increase_vs_decrease():
    df = _known_slope_df()
    result = sales_driver_analysis.run_analysis(df, "sales", ["visits"], {}, 0.05)
    current_mean = result["factor_stats"]["visits"]["mean"]

    up = simulate_change(result, "visits", "increase", "ratio", 1.2)
    down = simulate_change(result, "visits", "decrease", "ratio", 1.2)
    assert up["new_factor_value"] == pytest.approx(current_mean * 1.2)
    assert down["new_factor_value"] == pytest.approx(current_mean / 1.2)
    assert up["predicted_outcome_change"] > 0
    assert down["predicted_outcome_change"] < 0


def test_simulate_change_rejects_unknown_factor():
    df = _known_slope_df()
    result = sales_driver_analysis.run_analysis(df, "sales", ["visits"], {}, 0.05)
    scenario = simulate_change(result, "not_a_real_factor", "increase", "absolute", 1.0)
    assert not scenario["ok"]


def test_simulate_change_rejects_negative_amount():
    df = _known_slope_df()
    result = sales_driver_analysis.run_analysis(df, "sales", ["visits"], {}, 0.05)
    scenario = simulate_change(result, "visits", "increase", "absolute", -5.0)
    assert not scenario["ok"]
