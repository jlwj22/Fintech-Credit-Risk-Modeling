# Credit Risk Scoring

A full-pipeline credit risk model: probability-of-default prediction, a
300–850 scaled risk score, SHAP-based explainability, and an interactive
Streamlit demo — built on a real, public consumer credit dataset. This is the
exact underwriting problem card issuers (Capital One, Amex, Chase, Discover,
and every consumer lender) solve continuously: given an applicant's credit
profile, decide who gets approved, at what limit, and at what risk.

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-portfolio%20project-lightgrey)

**[Live demo →](#running-the-demo)** &nbsp;|&nbsp; **[Notebook →](notebooks/01_eda_and_modeling.ipynb)**

![App demo](reports/screenshots/app_demo.png)

![Score distribution](reports/figures/score_distribution.png)

## The business problem

Every consumer lender — a credit card issuer, a personal-loan platform, a
buy-now-pay-later shop — needs to answer one question fast, cheaply, and
consistently for every applicant: *how likely is this person to default, and
what should that mean for the decision?* Underwriting, credit-line
management, and collections prioritization all come down to the same
probability. This project builds that pipeline end to end — from raw, messy
credit-bureau-style data to a probability, a scaled score a non-technical
underwriter can read, and a plain-language explanation of *why*.

## Dataset

[**Give Me Some Credit**](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle),
~150,000 anonymized U.S. consumer credit records, 2011. Target: whether the
borrower experienced serious delinquency (90+ days past due) within two years.
Base rate: **6.7%** default — a realistically imbalanced problem, not a toy
50/50 dataset.

The raw data has the data-quality issues real credit data actually has:
~20% missing income, a cluster of sentinel/error codes (96/98) in the
delinquency-count columns, and utilization/debt-ratio outliers in the
thousands where they should be roughly 0–2. All of this is handled explicitly
in [`src/data_prep.py`](src/data_prep.py) rather than silently dropped.

## Approach

1. **Clean & engineer** (`src/data_prep.py`) — sentinel-code capping, missing-value
   imputation with explicit "was-missing" flags, log-transforms for skewed
   features, and a handful of engineered ratios (total delinquency count,
   income per dependent, credit lines per year of age).
2. **Model, champion vs. challenger** (`src/train.py`) — a class-weighted
   **logistic regression** (the auditable, regulator-friendly industry
   baseline) against an **XGBoost** classifier (the accuracy upgrade most
   shops layer on top), both evaluated the way credit risk teams actually
   evaluate: ROC-AUC, PR-AUC, **KS-statistic**, and **Gini**, plus top-decile
   capture rate for a targeting use case.
3. **Score it like an issuer would** (`src/scorecard.py`) — probabilities are
   converted to a 300–850 scaled score using the points-to-double-the-odds
   log-odds transform that underlies commercial credit scorecards industry-wide
   (the same math bank-built and bureau scorecards use), so a risk score
   translates directly into an approve/review/decline cutoff a lender can act on.
4. **Explain every decision** — SHAP values, both globally (which features
   matter overall) and per-applicant (why *this* applicant got *this* score),
   surfaced live in the demo app.

## Results

Evaluated on a held-out 25% test split (37,511 records):

| Model | ROC-AUC | PR-AUC | KS | Gini |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.863 | 0.389 | 0.571 | 0.726 |
| **XGBoost** | **0.868** | **0.403** | **0.576** | **0.737** |

A KS above 0.4 is considered good separation in credit risk; above 0.5 is very
good. **Targeting the riskiest 10% of applicants by predicted score catches 55%
of all actual defaults** in the test set — the number that matters for a
"review the top N% manually" workflow.

| ROC / PR curves | KS curve | Calibration |
|---|---|---|
| ![ROC/PR](reports/figures/roc_pr_curves.png) | ![KS](reports/figures/ks_curve.png) | ![Calibration](reports/figures/calibration.png) |

SHAP confirms the model leans on the same signals a credit analyst would:
prior delinquency count and revolving utilization dominate, followed by age
and debt ratio.

![SHAP summary](reports/figures/shap_summary.png)

Full narrative, additional EDA, and every chart above (regenerated from
scratch, not screenshots) live in
[`notebooks/01_eda_and_modeling.ipynb`](notebooks/01_eda_and_modeling.ipynb).

## Running the demo

```bash
git clone <this-repo>
cd fintech-credit-risk
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce data cleaning, training, evaluation, and all figures:
python -m src.train

# Launch the interactive scoring demo:
streamlit run app.py
```

The app takes an applicant profile (income, utilization, delinquency history,
etc.), returns a probability of default, scaled score, risk tier, and the
top SHAP factors behind that specific prediction.

## Project structure

```
├── app.py                     # Streamlit demo
├── src/
│   ├── data_prep.py           # cleaning + feature engineering (shared by training & app)
│   ├── train.py                # trains both models, evaluates, saves figures
│   └── scorecard.py           # PD -> scaled score -> risk tier
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── models/                    # trained model artifacts + metrics.json (generated)
├── reports/figures/           # evaluation charts (generated)
├── tests/                     # unit tests for the data pipeline
└── requirements.txt
```

## Notes & limitations

This is a portfolio project trained on public benchmark data, not a
production underwriting system — a real deployment (at a card issuer, a
personal-loan lender, or a bank credit-risk team) would need fairness/bias
auditing across protected classes, monitoring for population drift, a
reject-inference strategy (this dataset only contains applicants who were
actually extended credit), and sign-off from model risk management before
touching a live decision.

## License

MIT — see [LICENSE](LICENSE). Dataset used under its original Kaggle
competition terms for non-commercial/educational use.
