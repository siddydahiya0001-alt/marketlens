"""Turns raw analysis output into plain-language interpretation.

This module is intentionally the single seam where an LLM-backed
interpreter could be swapped in later. `InterpretationService` is an
abstract base; `RuleBasedInterpreter` is the deterministic v1 implementation
built entirely from templates and thresholds (no external API, no cost).

Beyond the headline answer, each `interpret_*` method also returns
`reasoning_steps`: a short, plain-language trail of *how* MarketLens arrived
at that answer (what data was used, what technique was applied, what the
key numbers were). This is distinct from the "Show technical details"
toggle - it explains the chain of reasoning in business language, not the
raw statistics.

To add an LLM-backed interpreter later:
    1. Create `LLMInterpreter(InterpretationService)` in a new module.
    2. Implement the same method signatures.
    3. Swap `get_interpreter()` to return the LLM version (e.g. via a
       feature flag or settings toggle).
Nothing in the `analyses/` or `pages/` modules should need to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from utils.terminology import confidence_from_p_value, confidence_sentence, r_squared_sentence


class InterpretationService(ABC):
    @abstractmethod
    def interpret_regression(self, result: dict) -> dict:
        ...

    @abstractmethod
    def interpret_classification(self, result: dict) -> dict:
        ...

    @abstractmethod
    def interpret_segmentation(self, result: dict) -> dict:
        ...

    @abstractmethod
    def interpret_campaign_comparison(self, result: dict) -> dict:
        ...


class RuleBasedInterpreter(InterpretationService):
    """Deterministic, template-driven interpretation. No external API calls."""

    def interpret_regression(self, result: dict) -> dict:
        r2 = result.get("r_squared", 0)
        target_label = result.get("target_label", "the result")
        ranked = result.get("ranked_factors", [])
        p_value = result.get("model_p_value")
        n_obs = result.get("n_obs", 0)

        significant_factors = [f for f in ranked if f.get("significant")]
        headline_pool = significant_factors if significant_factors else ranked
        top_factors = [f["name"] for f in headline_pool[:2]]

        if not ranked:
            main_answer = f"No strong factors were found that clearly explain {target_label}."
        elif significant_factors:
            main_answer = (
                f"{' and '.join(top_factors)} "
                f"{'are' if len(top_factors) > 1 else 'is'} the strongest factor"
                f"{'s' if len(top_factors) > 1 else ''} connected with {target_label}."
            )
        else:
            # None of the tested factors cleared the significance bar - the claim is hedged
            # rather than stated as fact, so the headline never overstates the evidence.
            main_answer = (
                f"{' and '.join(top_factors)} {'show' if len(top_factors) > 1 else 'shows'} the largest "
                f"measured connection with {target_label}, though none of the tested factors reached "
                "a level of evidence MarketLens considers reliable."
            )

        reasoning_steps = []
        reasoning_steps.append(
            f"We used {n_obs} customer records after removing rows with missing values in the tested columns."
        )
        reasoning_steps.append(
            f"We tested {len(ranked)} factor{'s' if len(ranked) != 1 else ''} against {target_label} using a "
            "statistical model that finds relationships between numbers."
        )
        if ranked:
            reasoning_steps.append(
                f"{len(significant_factors)} of {len(ranked)} factor{'s' if len(ranked) != 1 else ''} showed a "
                "pattern strong enough to trust, based on how consistently it appeared across the data."
            )
            top = ranked[0]
            reasoning_steps.append(
                f"'{top['name']}' had the largest measured effect, which we describe as a "
                f"'{top['strength'].lower()} {top['direction']} influence'."
            )
        reasoning_steps.append(
            f"The model explains about {round(r2 * 100)}% of the change in {target_label} - the rest comes "
            "from factors not included in this analysis, or from randomness."
        )

        confidence_label = confidence_from_p_value(p_value)
        if confidence_label in ("Very high confidence", "High confidence") and r2 < 0.3:
            # Confidence (is the pattern real?) and R-squared (how much does it explain?) answer
            # different questions - flag this explicitly so a high-confidence, low-R² result never
            # reads as contradictory.
            reasoning_steps.append(
                "Note: 'confidence' and 'how much this explains' are different questions. The evidence "
                "that this pattern is real (not random chance) is strong, even though the pattern only "
                "accounts for a small share of the overall change - other, unmeasured factors likely "
                "matter more."
            )

        return {
            "main_answer": main_answer,
            "explanation": r_squared_sentence(r2, target_label),
            "confidence": confidence_from_p_value(p_value),
            "confidence_sentence": confidence_sentence(p_value) if p_value is not None else
                "Confidence could not be calculated for the overall model.",
            "reasoning_steps": reasoning_steps,
        }


    def interpret_classification(self, result: dict) -> dict:
        accuracy = result.get("accuracy", 0)
        top_signals = result.get("top_positive_signals", [])
        signal_text = ", ".join(top_signals[:2]) if top_signals else "the selected factors"
        main_answer = f"Customers with stronger {signal_text} are most likely to buy."
        correct_per_100 = round(accuracy * 100)

        n_high = result.get("n_high", 0)
        n_medium = result.get("n_medium", 0)
        n_low = result.get("n_low", 0)
        n_total = n_high + n_medium + n_low

        reasoning_steps = [
            f"We split the {n_total} customer records into a training group (75%) and a held-out test group "
            "(25%) that the model never saw while learning.",
            "We used a statistical model that estimates the likelihood of an outcome (technical name: "
            "logistic regression), trained on the training group.",
            f"We checked its predictions against the real outcomes in the held-out test group: it correctly "
            f"identified {correct_per_100} out of every 100 cases.",
        ]
        if top_signals:
            reasoning_steps.append(
                f"'{top_signals[0]}' had the strongest positive connection with purchasing, so it carries "
                "the most weight in each customer's probability score."
            )
        reasoning_steps.append(
            f"We then scored every one of the {n_total} customers and grouped them into Low, Medium and "
            "High probability bands based on that score."
        )

        return {
            "main_answer": main_answer,
            "explanation": f"The model correctly identifies approximately {correct_per_100} out of every 100 cases.",
            "confidence": _confidence_from_accuracy(accuracy),
            "reasoning_steps": reasoning_steps,
        }


    def interpret_segmentation(self, result: dict) -> dict:
        n_groups = result.get("n_groups", 0)
        silhouette = result.get("silhouette_score", 0)
        quality = "clearly separated" if silhouette > 0.5 else (
            "reasonably separated" if silhouette > 0.25 else "somewhat overlapping"
        )
        n_cols = len(result.get("cluster_cols", []))

        reasoning_steps = [
            f"We put each customer's {n_cols} selected behaviours on a common scale, so no single large-number "
            "column (like income) could dominate over smaller-number ones (like visit count).",
            "We grouped customers by similarity (technical name: K-means clustering), trying group counts from "
            "2 to 8.",
            f"We picked {n_groups} groups because that count gave the clearest separation between groups, "
            f"measured by how close each customer is to its own group versus other groups.",
            "Each group's description and suggested name come from comparing its average behaviour to the "
            "overall average across all customers.",
        ]

        return {
            "main_answer": f"The customers naturally split into {n_groups} groups with {quality} behaviour.",
            "explanation": "Each group is described by its typical spending, frequency and recency.",
            "confidence": "High confidence" if silhouette > 0.5 else (
                "Moderate confidence" if silhouette > 0.25 else "Low confidence"
            ),
            "reasoning_steps": reasoning_steps,
        }


    def interpret_campaign_comparison(self, result: dict) -> dict:
        diff_pct = result.get("conversion_diff_pct", 0)
        better = result.get("better_campaign", "The tested campaign")
        p_value = result.get("p_value")

        n_control = result.get("n_control", 0)
        n_test = result.get("n_test", 0)
        control_group = result.get("control_group", "the control group")
        test_group = result.get("test_group", "the test group")
        control_rate = result.get("control_conversion_rate")
        test_rate = result.get("test_conversion_rate")
        comparison_group = control_group if better == test_group else test_group

        reasoning_steps = [
            f"We compared {n_control} customers in '{control_group}' against {n_test} customers in "
            f"'{test_group}'.",
        ]
        if control_rate is not None and test_rate is not None:
            reasoning_steps.append(
                f"Conversion rate was {control_rate * 100:.1f}% for '{control_group}' versus "
                f"{test_rate * 100:.1f}% for '{test_group}' - a difference of {abs(diff_pct):.1f} "
                "percentage points."
            )
        reasoning_steps.append(
            "We ran a statistical test that checks whether a difference this size is likely to happen just "
            "by chance if there were really no difference between the groups."
        )
        if p_value is not None:
            reasoning_steps.append(
                f"That test came back with {confidence_from_p_value(p_value).lower()}, which is why the "
                "result is labelled that way above."
            )

        return {
            "main_answer": _campaign_headline(
                better, comparison_group, diff_pct, control_group, control_rate, test_rate,
            ),
            "explanation": confidence_sentence(p_value) if p_value is not None else
                "Not enough information to judge statistical confidence.",
            "confidence": confidence_from_p_value(p_value) if p_value is not None else "Not enough evidence",
            "reasoning_steps": reasoning_steps,
        }


def _campaign_headline(better, comparison_group, diff_pct, control_group, control_rate, test_rate) -> str:
    """State which group converted better, by how much, and on what base.

    `better_campaign` is always the higher-converting group, so the comparison
    reads "higher" from its point of view regardless of which side happened to
    be the test group. Deriving the wording from the sign of the difference
    instead - which is measured from the *test* group - flipped the sentence
    whenever the control group won, reporting the winner as the loser.

    The gap is stated in percentage points, not "%": a move from 40.6% to 51.8%
    is 11.2 percentage points, not an 11.2% increase.
    """
    if diff_pct == 0:
        return f"'{better}' and '{comparison_group}' converted at the same rate."

    better_rate = control_rate if better == control_group else test_rate
    worse_rate = test_rate if better == control_group else control_rate

    sentence = (
        f"'{better}' converted {abs(diff_pct):.1f} percentage points higher than '{comparison_group}'"
    )
    if better_rate is not None and worse_rate is not None:
        sentence += f" ({better_rate * 100:.1f}% versus {worse_rate * 100:.1f}%)"
    return sentence + "."


def _confidence_from_accuracy(accuracy: float) -> str:
    if accuracy >= 0.85:
        return "Very high confidence"
    if accuracy >= 0.75:
        return "High confidence"
    if accuracy >= 0.65:
        return "Moderate confidence"
    if accuracy >= 0.55:
        return "Low confidence"
    return "Not enough evidence"


_DEFAULT_INTERPRETER = RuleBasedInterpreter()


def get_interpreter() -> InterpretationService:
    """Single access point used by the rest of the app. Swap here to change engines."""
    return _DEFAULT_INTERPRETER
