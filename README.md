# MarketLens

A simple, local alternative to SPSS built specifically for marketing analytics. Upload a
spreadsheet, tell MarketLens what you want to understand in plain business language, and get
visual, easy-to-understand answers - no statistics background required.

## Problem being solved

Marketing teams routinely have the data needed to answer questions like "what's driving sales?"
or "which customers are about to churn?" but lack access to (or comfort with) statistical software
like SPSS, R, or Python notebooks. Existing tools assume the user already knows terms like
*p-value*, *R-squared*, or *logistic regression*. MarketLens runs the same real statistical
techniques (linear/logistic regression, K-means clustering, hypothesis testing, time-series
decomposition) but translates every result into plain business language by default, with
technical detail available on demand for analysts who want it.

## Features

- **Upload & review** - CSV, XLSX, XLS with multi-worksheet support, data preview, dtypes,
  missing-value and duplicate-row summaries, and friendly error messages for bad files.
- **Automatic column understanding** - Every column is auto-classified into a business role
  (Customer ID, Revenue, Sales, Cost, Profit, Campaign, Customer behaviour, Date, Yes/No result,
  etc.) using name + data-shape heuristics. Every suggestion is user-editable and the original
  file is never modified.
- **Data-quality dashboard** - Plain-language findings ("12% of values are missing in the Income
  column") with user-chosen cleaning actions (drop, impute, deduplicate, standardise labels,
  exclude columns) applied to a working copy, with before/after row counts.
- **Ten business questions**, each mapped to a real statistical technique under the hood but
  explained in plain language: sales drivers (linear regression), purchase likelihood (logistic
  regression), customer segments (K-means), campaign comparison (two-proportion hypothesis test),
  churn risk (interpretable risk scoring / logistic regression), customer value (RFM-style CLV),
  channel performance, sales-over-time (trend + Holt forecasting), free-form exploration, and a
  custom chart builder.
- **Consistent result structure** for every analysis: what we found, what it means, how
  confident we are, which factors matter most, recommended action, what should *not* be
  concluded, limitations, and what additional data would help.
- **Plain-language dictionary** - a single source of truth mapping every statistical term to a
  business-friendly phrase (`utils/terminology.py`), used everywhere in the UI.
- **"Show technical details" toggle** - reveals R-squared, coefficients, p-values, confidence
  intervals, VIF, confusion matrices, ROC-AUC, residual plots, and full model summaries, on demand.
- **Analytical guardrails** - warnings for small samples, imbalanced outcomes, missing data,
  dominant categories, multicollinearity, outlier influence, ID/date/name columns used as
  predictors, observational (non-experimental) campaign data, correlation-vs-causation, and
  insufficient history for forecasting.
- **Downloads** - cleaned dataset, analysis results, customer-level predictions/segments (Excel),
  charts (PNG), and a summary report (HTML, with a best-effort PDF export).

## Installation

```bash
python -m venv venv
```

Mac or Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Supported file formats

`.csv`, `.xlsx`, `.xls`. For Excel files with multiple worksheets, MarketLens lists every sheet
name and loads only the one you select.

## Project structure

```text
marketlens/
├── app.py                     # Entry point / landing page
├── requirements.txt
├── README.md
├── pytest.ini
│
├── pages/
│   ├── 1_Upload_and_Review.py
│   ├── 2_Data_Quality.py
│   ├── 3_Choose_Question.py
│   └── 4_Results.py
│
├── services/
│   ├── file_loader.py         # CSV/Excel loading + validation
│   ├── column_detector.py     # Business-role auto-detection
│   ├── data_cleaner.py        # Quality findings + cleaning actions
│   ├── analysis_router.py     # Routes a question to its analysis module
│   ├── interpretation_service.py  # Plain-language interpretation (LLM-ready seam)
│   ├── recommendation_engine.py   # Priority-aware recommendations
│   ├── validation_service.py  # Analytical guardrails / warnings
│   └── export_service.py      # Excel / HTML / PDF / PNG export
│
├── analyses/                  # One module per business question (pure `run_analysis`
│                               # functions plus a Streamlit `render` function each)
│   ├── descriptive_analysis.py
│   ├── sales_driver_analysis.py
│   ├── purchase_prediction.py
│   ├── customer_segmentation.py
│   ├── campaign_comparison.py
│   ├── churn_analysis.py
│   ├── customer_value.py
│   ├── channel_performance.py
│   └── time_series_analysis.py
│
├── visualisations/
│   ├── charts.py               # Plotly chart builders with plain-language captions
│   ├── metric_cards.py
│   └── result_panels.py        # Shared "what we found / confidence / action" layout
│
├── utils/
│   ├── terminology.py          # The plain-language dictionary
│   ├── constants.py
│   ├── helpers.py
│   ├── session_state.py
│   └── sidebar.py
│
├── sample_data/
│   ├── generate_sample_data.py
│   └── marketing_sample.xlsx   # 600+ fictional customers, intentionally messy
│
└── tests/
    ├── test_file_loader.py
    ├── test_column_detector.py
    ├── test_data_cleaner.py
    └── test_analyses.py
```

## Sample workflow

1. Run `streamlit run app.py` and open **1 Upload and Review** in the sidebar.
2. Upload `sample_data/marketing_sample.xlsx` and select the **Customer_Data** worksheet.
3. Review the auto-detected column roles (e.g. `Revenue` → Revenue, `Purchased` → Yes or No
   result, `Customer_ID` → Customer ID) and correct any if needed.
4. Go to **2 Data Quality**, review the plain-language findings, choose cleaning actions
   (e.g. standardise labels, remove duplicates), and apply them.
5. Go to **3 Choose a Question**, pick e.g. *"What is influencing sales?"*, and map the Revenue
   column plus a few factors (Website_Visits, Purchase_Frequency, Discount_Percentage, ...).
6. Go to **4 Results** to see the plain-language answer, confidence level, factor ranking chart,
   recommended action, and limitations. Toggle **Show technical details** in the sidebar for the
   full regression output.
7. Download the results, predictions, or a summary report from the buttons provided.

## Regenerating the sample dataset

```bash
python sample_data/generate_sample_data.py
```

This regenerates `sample_data/marketing_sample.xlsx` with a fresh random seed-free dataset of
600+ fictional customers, including intentional missing values, duplicate rows, outliers, and
inconsistent category labels (e.g. "Yes"/"Y"/"yes"/"Purchased") so the Data Quality page has
something real to demonstrate.

## Running tests

```bash
pytest
```

Tests cover file loading (CSV/Excel/worksheet selection/empty & corrupted files), column-role
detection, data-quality findings and cleaning actions, and the statistical output of the
regression, classification, clustering, and campaign-comparison modules, plus the plain-language
interpretation layer.

## Adding an LLM-backed interpreter later

All plain-language generation goes through `services/interpretation_service.py`. The default
`RuleBasedInterpreter` is fully deterministic (templates + thresholds, no external calls, no
cost). To add an LLM-backed version later:

1. Create a new class implementing the same `InterpretationService` interface (e.g.
   `LLMInterpreter` in a new module).
2. Update `get_interpreter()` to return it (behind a settings flag, if you want both available).

Nothing in `analyses/` or `pages/` needs to change, since they only ever call `get_interpreter()`.

## Limitations

- The rule-based interpreter uses templates and thresholds, not natural-language generation - it
  is deterministic and free, but phrasing is less flexible than an LLM-backed version would be.
- Churn analysis without a known outcome column uses a transparent heuristic risk score (weighted,
  direction-aware z-scores), not a trained classifier - this is disclosed in the UI.
- Campaign and channel comparisons are based on observational data; MarketLens will not (and
  cannot) prove causation, and says so explicitly in every relevant result.
- Forecasting (Module I) only appears when enough historical periods exist, and is a projection
  based on past patterns, not a guarantee.
- PDF export is a best-effort, text-based fallback (via `reportlab`) rather than a full visual
  reproduction of the HTML report; the HTML report is the more complete export format.
- Designed and tested for datasets that fit comfortably in memory (tens of thousands of rows);
  it is not built for big-data-scale files.

## Future improvements

- Optional LLM-backed interpretation service for richer, more contextual explanations.
- Richer attribution modelling for channel performance (multi-touch, not last-touch/independent).
- A/B test design assistance (sample-size calculators) ahead of running a campaign comparison.
- Saved projects / session persistence across browser restarts.
- More chart types (cohort/retention heatmaps, funnel-over-time).
