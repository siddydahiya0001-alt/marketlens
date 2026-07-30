"""Shared constants used across the MarketLens application."""

APP_NAME = "MarketLens"
APP_TAGLINE = "Marketing analytics without the statistics jargon."

# ---------------------------------------------------------------------------
# Business roles a column can be assigned to.
# ---------------------------------------------------------------------------
ROLE_CUSTOMER_ID = "Customer ID"
ROLE_NUMBER = "Number"
ROLE_CATEGORY = "Category"
ROLE_DATE = "Date"
ROLE_YES_NO = "Yes or No result"
ROLE_REVENUE = "Revenue"
ROLE_SALES = "Sales"
ROLE_COST = "Cost"
ROLE_PROFIT = "Profit"
ROLE_CAMPAIGN = "Campaign"
ROLE_BEHAVIOUR = "Customer behaviour"
ROLE_TEXT = "Text"
ROLE_UNKNOWN = "Unknown"

ALL_ROLES = [
    ROLE_CUSTOMER_ID,
    ROLE_NUMBER,
    ROLE_CATEGORY,
    ROLE_DATE,
    ROLE_YES_NO,
    ROLE_REVENUE,
    ROLE_SALES,
    ROLE_COST,
    ROLE_PROFIT,
    ROLE_CAMPAIGN,
    ROLE_BEHAVIOUR,
    ROLE_TEXT,
    ROLE_UNKNOWN,
]

NUMERIC_ROLES = {ROLE_NUMBER, ROLE_REVENUE, ROLE_SALES, ROLE_COST, ROLE_PROFIT, ROLE_BEHAVIOUR}
ID_LIKE_ROLES = {ROLE_CUSTOMER_ID}

# ---------------------------------------------------------------------------
# Business questions offered on the "Choose a question" page.
# ---------------------------------------------------------------------------
QUESTION_SALES_DRIVERS = "What is influencing sales?"
QUESTION_PURCHASE_LIKELIHOOD = "Who is most likely to purchase?"
QUESTION_SEGMENTATION = "Which customers behave similarly?"
QUESTION_CAMPAIGN = "Did a campaign perform better?"
QUESTION_CHURN = "Which customers may stop buying?"
QUESTION_CUSTOMER_VALUE = "What is the value of each customer?"
QUESTION_CHANNEL = "Which marketing channel performs best?"
QUESTION_TIME_SERIES = "How are sales changing over time?"
QUESTION_EXPLORE = "Explore the data visually"
QUESTION_CUSTOM = "Create a custom analysis"

BUSINESS_QUESTIONS = [
    QUESTION_SALES_DRIVERS,
    QUESTION_PURCHASE_LIKELIHOOD,
    QUESTION_SEGMENTATION,
    QUESTION_CAMPAIGN,
    QUESTION_CHURN,
    QUESTION_CUSTOMER_VALUE,
    QUESTION_CHANNEL,
    QUESTION_TIME_SERIES,
    QUESTION_EXPLORE,
    QUESTION_CUSTOM,
]

QUESTION_DESCRIPTIONS = {
    QUESTION_SALES_DRIVERS: "Find what factors influence sales.",
    QUESTION_PURCHASE_LIKELIHOOD: "Find which customers are likely to buy next.",
    QUESTION_SEGMENTATION: "Find customer groups with similar behaviour.",
    QUESTION_CAMPAIGN: "Compare two campaigns or offers.",
    QUESTION_CHURN: "Find customers who may stop buying.",
    QUESTION_CUSTOMER_VALUE: "Estimate how valuable each customer is.",
    QUESTION_CHANNEL: "Compare marketing channels or sources.",
    QUESTION_TIME_SERIES: "See how sales are trending over time.",
    QUESTION_EXPLORE: "Look freely at charts and summaries.",
    QUESTION_CUSTOM: "Build your own chart or comparison.",
}

# ---------------------------------------------------------------------------
# Flexible interface options
# ---------------------------------------------------------------------------
AUDIENCE_OPTIONS = ["Student", "Marketing manager", "Senior management", "Analyst", "Client"]
EXPLANATION_LEVELS = ["Very simple", "Business level", "Detailed", "Technical"]
BUSINESS_PRIORITIES = [
    "Increase sales",
    "Increase profit",
    "Retain customers",
    "Reduce marketing cost",
    "Reduce risk",
    "Improve campaign performance",
]
CURRENCY_OPTIONS = ["$", "€", "£", "₹", "¥"]
NUMBER_FORMATS = ["1,234.56", "1.234,56", "1 234.56"]

# Broader currency list for user-added columns in the data editor (code, symbol, display label).
CURRENCY_CHOICES = [
    ("USD", "$", "USD - US Dollar ($)"),
    ("EUR", "€", "EUR - Euro (€)"),
    ("GBP", "£", "GBP - British Pound (£)"),
    ("INR", "₹", "INR - Indian Rupee (₹)"),
    ("JPY", "¥", "JPY - Japanese Yen (¥)"),
    ("CNY", "¥", "CNY - Chinese Yuan (¥)"),
    ("AUD", "A$", "AUD - Australian Dollar (A$)"),
    ("CAD", "C$", "CAD - Canadian Dollar (C$)"),
    ("CHF", "Fr", "CHF - Swiss Franc (Fr)"),
    ("SGD", "S$", "SGD - Singapore Dollar (S$)"),
    ("HKD", "HK$", "HKD - Hong Kong Dollar (HK$)"),
    ("NZD", "NZ$", "NZD - New Zealand Dollar (NZ$)"),
    ("SEK", "kr", "SEK - Swedish Krona (kr)"),
    ("NOK", "kr", "NOK - Norwegian Krone (kr)"),
    ("DKK", "kr", "DKK - Danish Krone (kr)"),
    ("ZAR", "R", "ZAR - South African Rand (R)"),
    ("AED", "د.إ", "AED - UAE Dirham (د.إ)"),
    ("SAR", "﷼", "SAR - Saudi Riyal (﷼)"),
    ("KRW", "₩", "KRW - South Korean Won (₩)"),
    ("BRL", "R$", "BRL - Brazilian Real (R$)"),
    ("MXN", "$", "MXN - Mexican Peso ($)"),
    ("RUB", "₽", "RUB - Russian Ruble (₽)"),
    ("TRY", "₺", "TRY - Turkish Lira (₺)"),
    ("IDR", "Rp", "IDR - Indonesian Rupiah (Rp)"),
    ("MYR", "RM", "MYR - Malaysian Ringgit (RM)"),
    ("THB", "฿", "THB - Thai Baht (฿)"),
    ("PHP", "₱", "PHP - Philippine Peso (₱)"),
    ("VND", "₫", "VND - Vietnamese Dong (₫)"),
    ("PLN", "zł", "PLN - Polish Zloty (zł)"),
    ("EGP", "E£", "EGP - Egyptian Pound (E£)"),
    ("NGN", "₦", "NGN - Nigerian Naira (₦)"),
]

CHART_TYPES = [
    "Bar chart",
    "Line chart",
    "Histogram",
    "Box plot",
    "Scatter plot",
    "Pie chart",
    "Funnel chart",
    "Correlation heatmap",
]

TIME_PERIODS = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]

CONFIDENCE_LEVELS = [
    "Very high confidence",
    "High confidence",
    "Moderate confidence",
    "Low confidence",
    "Not enough evidence",
]

MIN_ROWS_FOR_ANALYSIS = 30
MIN_ROWS_FOR_FORECAST = 12
