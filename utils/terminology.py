"""Plain-language dictionary and helpers that turn statistics into business language.

Every place in the app that would otherwise show a statistical term should
route through this module first. Technical terms are only shown when the
user turns on "Show technical details".
"""

# ---------------------------------------------------------------------------
# Technical term -> plain language definition
# ---------------------------------------------------------------------------
PLAIN_LANGUAGE_DICTIONARY = {
    "Linear regression": "Find what influences a number",
    "Logistic regression": "Find what is likely to happen",
    "R-squared": "How much of the result can be explained",
    "Adjusted R-squared": "Explanation strength after considering the number of factors",
    "P-value": "Evidence that the pattern is not random",
    "Alpha": "Minimum evidence required before trusting a result",
    "Coefficient": "Strength and direction of influence",
    "Correlation": "How closely two things move together",
    "Statistical significance": "Evidence strong enough to take seriously",
    "Confidence interval": "Likely range of the true result",
    "Cluster analysis": "Find groups with similar behaviour",
    "Multicollinearity": "Two factors are giving almost the same information",
    "Outlier": "An unusually high or low value",
    "Residual": "Difference between the predicted and actual result",
    "Accuracy": "Percentage of total predictions that were correct",
    "Precision": "How often a predicted buyer was actually a buyer",
    "Recall": "How many actual buyers were successfully identified",
    "Standard deviation": "How widely the values are spread",
    "Mean": "Average",
    "Median": "Middle value",
    "Null hypothesis": "Starting assumption that there is no meaningful difference",
    "ROC-AUC": "How well the model separates buyers from non-buyers",
    "VIF": "How much a factor overlaps with other factors",
    "F1 score": "Balance between precision and recall",
    "Silhouette score": "How clearly separated the customer groups are",
    "Standard error": "Typical size of estimation error",
    "Degrees of freedom": "Amount of independent information used in the calculation",
    "Homoscedasticity": "Whether errors are similarly sized across the data",
    "Normality": "Whether values are shaped like a bell curve",
    "T-test": "A check comparing the averages of two groups",
    "Chi-square test": "A check comparing category proportions between groups",
    "ROAS": "Revenue earned per currency unit spent on marketing",
    "CAC": "Average cost to acquire one paying customer",
    "CLV": "Total value a customer is expected to bring over time",
    "RFM": "Recency, frequency and monetary value used to describe a customer",
}


def plain(term: str) -> str:
    """Return the plain-language definition for a technical term, or the term itself."""
    return PLAIN_LANGUAGE_DICTIONARY.get(term, term)


def label(term: str, show_technical: bool) -> str:
    """Return a display label for `term`.

    When technical details are off, only the plain phrase is shown.
    When on, both the plain phrase and the technical term are shown.
    """
    plain_text = plain(term)
    if show_technical and plain_text != term:
        return f"{plain_text} ({term})"
    return plain_text


def confidence_from_p_value(p_value: float) -> str:
    """Translate a p-value into a plain confidence label."""
    if p_value is None or (isinstance(p_value, float) and (p_value != p_value)):
        return "Not enough evidence"
    if p_value < 0.001:
        return "Very high confidence"
    if p_value < 0.01:
        return "High confidence"
    if p_value < 0.05:
        return "Moderate confidence"
    if p_value < 0.10:
        return "Low confidence"
    return "Not enough evidence"


def confidence_sentence(p_value: float) -> str:
    """A short plain-language sentence describing statistical confidence."""
    level = confidence_from_p_value(p_value)
    sentences = {
        "Very high confidence": "There is very strong evidence that this pattern is not random.",
        "High confidence": "There is strong evidence that this pattern is not random.",
        "Moderate confidence": "There is some evidence that this pattern is not random.",
        "Low confidence": "The evidence is weak and this pattern could be due to chance.",
        "Not enough evidence": "There is not enough evidence to trust this pattern yet.",
    }
    return sentences[level]


def r_squared_sentence(r_squared: float, target_label: str = "the result") -> str:
    """Plain-language explanation of R-squared."""
    pct = round(r_squared * 100)
    return f"We can explain about {pct}% of the changes in {target_label} using the selected factors."


def effect_direction_label(coefficient_sign: int, strength: str) -> str:
    """Combine direction and strength into a plain phrase, e.g. 'Strong positive influence'."""
    direction = "positive" if coefficient_sign > 0 else "negative" if coefficient_sign < 0 else "no"
    return f"{strength} {direction} influence".strip().capitalize()


def strength_from_effect_size(standardised_value: float) -> str:
    """Map a standardised effect size (e.g. |standardised coefficient|) to a strength word."""
    value = abs(standardised_value)
    if value >= 0.5:
        return "Strong"
    if value >= 0.25:
        return "Moderate"
    if value >= 0.1:
        return "Weak"
    return "Very weak"
