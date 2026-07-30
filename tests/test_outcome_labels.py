"""Regression tests for messy Yes/No outcome labels.

Real spreadsheets spell the same outcome several ways in one column ("Yes",
"yes", "Y", "Purchased"). Matching a single exact string counted only one
spelling as a conversion and silently treated the rest as non-conversions,
which corrupted conversion rates and classification targets. These tests pin
the equivalence behaviour, including against the shipped sample workbook in
its raw, uncleaned state.
"""

import numpy as np
import pandas as pd
import pytest

from analyses import campaign_comparison, channel_performance, purchase_prediction
from services.data_cleaner import apply_cleaning_actions
from utils.helpers import infer_positive_value, normalise_label, positive_mask

from tests.sample_workbook import load_sample_customer_data

MESSY_AFFIRMATIVES = ["Yes", "yes", "Y", "Purchased"]
MESSY_NEGATIVES = ["No", "no", "N"]


def _messy_outcome_series():
    return pd.Series(MESSY_AFFIRMATIVES + MESSY_NEGATIVES + [None])


# ---------------------------------------------------------------------------
# positive_mask
# ---------------------------------------------------------------------------

def test_positive_mask_matches_every_affirmative_spelling():
    series = _messy_outcome_series()
    mask = positive_mask(series, "Yes")
    assert mask.sum() == len(MESSY_AFFIRMATIVES)
    assert list(series[mask]) == MESSY_AFFIRMATIVES


@pytest.mark.parametrize("chosen", MESSY_AFFIRMATIVES)
def test_positive_mask_is_independent_of_which_affirmative_is_chosen(chosen):
    """'Purchased' and 'Y' must behave identically to 'Yes' - the user's pick
    of positive value should not change how many conversions are counted."""
    series = _messy_outcome_series()
    assert positive_mask(series, chosen).sum() == len(MESSY_AFFIRMATIVES)


def test_positive_mask_matches_every_negative_spelling():
    series = _messy_outcome_series()
    mask = positive_mask(series, "No")
    assert mask.sum() == len(MESSY_NEGATIVES)


def test_positive_mask_never_matches_missing_values():
    series = _messy_outcome_series()
    assert not positive_mask(series, "Yes")[series.isna()].any()


def test_positive_mask_does_not_merge_unrelated_categories():
    """Equivalence only applies to recognisable yes/no columns - a campaign
    column must still match one campaign, not every campaign."""
    series = pd.Series(["Campaign A", "Campaign B", "Campaign C"])
    assert positive_mask(series, "Campaign A").sum() == 1


def test_positive_mask_still_normalises_case_for_non_yes_no_columns():
    series = pd.Series(["Campaign A", "campaign a", " CAMPAIGN A ", "Campaign B"])
    assert positive_mask(series, "Campaign A").sum() == 3


def test_positive_mask_keeps_three_valued_columns_exact():
    """'Maybe' makes this not a yes/no column, so no spellings are merged."""
    series = pd.Series(["Yes", "Y", "No", "Maybe"])
    assert positive_mask(series, "Yes").sum() == 1


def test_positive_mask_handles_numeric_outcomes_directly():
    series = pd.Series([1, 0, 1, 1, 0])
    assert positive_mask(series, 1).sum() == 3


def test_positive_mask_handles_float_outcomes_without_text_mangling():
    """Normalising 1.0 as text would strip the dot and produce '10'."""
    series = pd.Series([1.0, 0.0, 1.0])
    assert positive_mask(series, 1.0).sum() == 2


def test_positive_mask_handles_boolean_outcomes():
    series = pd.Series([True, False, True])
    assert positive_mask(series, True).sum() == 2


def test_positive_mask_with_no_positive_value_matches_nothing():
    assert positive_mask(_messy_outcome_series(), None).sum() == 0


# ---------------------------------------------------------------------------
# infer_positive_value
# ---------------------------------------------------------------------------

def test_infer_positive_value_prefers_the_most_common_affirmative():
    """Row order used to decide this, so a 14-row 'Purchased' could beat a
    265-row 'Yes'. The most recognisable spelling should win."""
    series = pd.Series(["no"] + ["Purchased"] * 3 + ["Yes"] * 20 + ["No"] * 10)
    assert infer_positive_value(series) == "Yes"


def test_infer_positive_value_falls_back_when_nothing_looks_affirmative():
    series = pd.Series(["Group A", "Group B"])
    assert infer_positive_value(series) == "Group A"


def test_infer_positive_value_returns_none_for_empty_column():
    assert infer_positive_value(pd.Series([None, np.nan], dtype=object)) is None


# ---------------------------------------------------------------------------
# End-to-end: messy labels must not change the analysis conclusion
# ---------------------------------------------------------------------------

def _campaign_df(n=400, seed=4, messy=False):
    rng = np.random.default_rng(seed)
    campaign = rng.choice(["A", "B"], n)
    base_prob = np.where(campaign == "B", 0.5, 0.3)
    purchased = np.where(rng.random(n) < base_prob, "Yes", "No")
    if messy:
        # Spell the same two outcomes several ways, as a real export would.
        yes_variants = rng.choice(MESSY_AFFIRMATIVES, n)
        no_variants = rng.choice(MESSY_NEGATIVES, n)
        purchased = np.where(purchased == "Yes", yes_variants, no_variants)
    return pd.DataFrame({"campaign": campaign, "purchased": purchased})


def test_campaign_comparison_conversion_rates_survive_messy_labels():
    clean = campaign_comparison.run_analysis(
        _campaign_df(), "campaign", "purchased", None, None, "A", "B")
    messy = campaign_comparison.run_analysis(
        _campaign_df(messy=True), "campaign", "purchased", None, None, "A", "B")

    assert clean["ok"] and messy["ok"]
    assert messy["control_conversion_rate"] == pytest.approx(clean["control_conversion_rate"])
    assert messy["test_conversion_rate"] == pytest.approx(clean["test_conversion_rate"])
    assert messy["p_value"] == pytest.approx(clean["p_value"])
    assert messy["better_campaign"] == clean["better_campaign"] == "B"


def test_channel_performance_conversion_rates_survive_messy_labels():
    df = _campaign_df(messy=True)
    result = channel_performance.run_analysis(df, "campaign", None, None, "purchased")
    assert result["ok"]
    rates = result["channel_summary"]["Conversion rate"]
    # Roughly 30% and 50% by construction - not the ~12% a single spelling would give.
    assert rates.min() > 0.2
    assert rates.max() > 0.4


def _classification_df(n=300, seed=2, messy=False):
    rng = np.random.default_rng(seed)
    visits = rng.poisson(4, n)
    recency = rng.exponential(30, n)
    score = 0.5 * visits - 0.05 * recency + rng.normal(0, 1, n)
    purchased = np.where(score > np.median(score), "Yes", "No")
    if messy:
        purchased = np.where(purchased == "Yes",
                              rng.choice(MESSY_AFFIRMATIVES, n),
                              rng.choice(MESSY_NEGATIVES, n))
    return pd.DataFrame({"visits": visits, "recency": recency, "purchased": purchased})


def test_purchase_prediction_target_survives_messy_labels():
    """Half the rows are buyers by construction; a single-spelling match used to
    collapse that to a quarter of them and inflate accuracy on the imbalance."""
    result = purchase_prediction.run_analysis(
        _classification_df(messy=True), "purchased", ["visits", "recency"], {})
    assert result["ok"]
    assert result["auc"] is not None and result["auc"] > 0.7
    assert result["n_high"] > 0


# ---------------------------------------------------------------------------
# The shipped sample workbook, exactly as a user would first upload it
# ---------------------------------------------------------------------------

def test_sample_data_still_ships_mixed_outcome_spellings():
    """Guards the premise of the tests below - if the generator is ever changed
    to emit clean labels, these regressions would silently stop testing anything."""
    df = load_sample_customer_data()
    spellings = {normalise_label(v) for v in df["Purchased"].dropna().unique()}
    assert {"yes", "y", "purchased"}.issubset(spellings)
    assert {"no", "n"}.issubset(spellings)


@pytest.mark.parametrize("chosen", MESSY_AFFIRMATIVES)
def test_sample_data_purchase_prediction_target_is_spelling_independent(chosen):
    """Whichever affirmative spelling reaches run_analysis - inferred, or picked
    by the user on the Choose a Question page - the model must train on the same
    ~47% of buyers, not on the 2% that spell it one particular way."""
    df = load_sample_customer_data()
    factors = ["Website_Visits", "Purchase_Frequency", "Income"]

    result = purchase_prediction.run_analysis(df, "Purchased", factors, {}, positive_value=chosen)
    assert result["ok"]

    # Buyers are roughly half the file, so the degenerate all-negative model that
    # used to score 97% accuracy cannot occur, and the imbalance warning must not fire.
    assert result["accuracy"] < 0.9
    assert not any("far more common" in w for w in result["limitations"])
    assert result["n_high"] > 0


def test_sample_data_purchase_prediction_matches_cleaned_result():
    raw = load_sample_customer_data()
    cleaned, _ = apply_cleaning_actions(raw, {"standardise_labels": True}, {})
    factors = ["Website_Visits", "Purchase_Frequency", "Income"]

    raw_result = purchase_prediction.run_analysis(raw, "Purchased", factors, {})
    cleaned_result = purchase_prediction.run_analysis(cleaned, "Purchased", factors, {})

    assert raw_result["accuracy"] == pytest.approx(cleaned_result["accuracy"])
    assert raw_result["auc"] == pytest.approx(cleaned_result["auc"])
    assert raw_result["n_high"] == cleaned_result["n_high"]


def test_sample_data_campaign_comparison_matches_cleaned_result():
    """Standardising labels on the Data Quality page is an optional step. The
    conclusion must not depend on whether the user happened to take it."""
    raw = load_sample_customer_data()
    cleaned, _ = apply_cleaning_actions(raw, {"standardise_labels": True}, {})

    raw_result = campaign_comparison.run_analysis(
        raw, "Campaign", "Purchased", None, None, "Campaign A", "Campaign B")
    cleaned_result = campaign_comparison.run_analysis(
        cleaned, "Campaign", "Purchased", None, None, "Campaign A", "Campaign B")

    assert raw_result["control_conversion_rate"] == pytest.approx(
        cleaned_result["control_conversion_rate"])
    assert raw_result["test_conversion_rate"] == pytest.approx(
        cleaned_result["test_conversion_rate"])
    assert raw_result["p_value"] == pytest.approx(cleaned_result["p_value"])
    assert raw_result["better_campaign"] == cleaned_result["better_campaign"]
    # Conversion sits near 50%, not the ~2% a single spelling produced.
    assert raw_result["control_conversion_rate"] > 0.3
