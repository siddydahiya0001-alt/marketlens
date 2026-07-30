"""Regression tests for the campaign-comparison headline sentence.

`better_campaign` is always the higher-converting group, but the wording used
to be derived from the sign of the difference - which is measured from the test
group. When the control group won, the sentence named the winner and then said
it performed *lower*, reversing the finding. The gap was also reported as "%"
when it is a difference in percentage points.
"""

import pandas as pd
import pytest

from analyses import campaign_comparison
from services.interpretation_service import get_interpreter


def _result(control_rate, test_rate, p_value=0.01,
            control_group="Campaign A", test_group="Campaign B"):
    diff_pct = (test_rate - control_rate) * 100
    return {
        "control_group": control_group,
        "test_group": test_group,
        "n_control": 100,
        "n_test": 100,
        "control_conversion_rate": control_rate,
        "test_conversion_rate": test_rate,
        "conversion_diff_pct": diff_pct,
        "better_campaign": test_group if diff_pct >= 0 else control_group,
        "p_value": p_value,
    }


def _headline(control_rate, test_rate, **kwargs):
    return get_interpreter().interpret_campaign_comparison(
        _result(control_rate, test_rate, **kwargs))["main_answer"]


def test_headline_names_the_control_group_as_higher_when_it_wins():
    """The bug: 'Campaign A' won on 20% vs 15% but was reported as 'lower'."""
    headline = _headline(control_rate=0.20, test_rate=0.15)
    assert "'Campaign A' converted" in headline
    assert "higher" in headline
    assert "lower" not in headline


def test_headline_names_the_test_group_as_higher_when_it_wins():
    headline = _headline(control_rate=0.15, test_rate=0.20)
    assert "'Campaign B' converted" in headline
    assert "higher" in headline
    assert "lower" not in headline


@pytest.mark.parametrize("control_rate,test_rate,winner,loser", [
    (0.20, 0.15, "Campaign A", "Campaign B"),
    (0.15, 0.20, "Campaign B", "Campaign A"),
])
def test_headline_always_puts_the_winner_first(control_rate, test_rate, winner, loser):
    headline = _headline(control_rate, test_rate)
    assert headline.index(winner) < headline.index(loser)


def test_headline_reports_the_gap_in_percentage_points_not_percent():
    """40.6% -> 51.8% is 11.2 percentage points, not an 11.2% increase."""
    headline = _headline(control_rate=0.518, test_rate=0.406)
    assert "11.2 percentage points" in headline
    assert "11.2%" not in headline
    assert "11%" not in headline


def test_headline_quotes_both_underlying_rates():
    headline = _headline(control_rate=0.518, test_rate=0.406)
    assert "51.8%" in headline
    assert "40.6%" in headline


def test_headline_handles_an_exact_tie_without_claiming_a_winner():
    headline = _headline(control_rate=0.20, test_rate=0.20)
    assert "same rate" in headline
    assert "higher" not in headline
    assert "lower" not in headline


def test_headline_is_consistent_with_better_campaign_end_to_end():
    """Run the real analysis with a control group that outperforms the test
    group, and check the sentence agrees with the computed winner."""
    df = pd.DataFrame({
        "campaign": ["A"] * 100 + ["B"] * 100,
        "purchased": (["Yes"] * 60 + ["No"] * 40) + (["Yes"] * 25 + ["No"] * 75),
    })
    result = campaign_comparison.run_analysis(df, "campaign", "purchased", None, None, "A", "B")
    assert result["ok"]
    assert result["better_campaign"] == "A"
    assert result["control_conversion_rate"] > result["test_conversion_rate"]

    headline = get_interpreter().interpret_campaign_comparison(result)["main_answer"]
    assert headline.startswith("'A' converted")
    assert "higher" in headline
    assert "35.0 percentage points" in headline
