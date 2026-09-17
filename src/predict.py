"""
Score a single applicant with the trained XGBoost model.

Shared by the Streamlit app and the FastAPI service so both return the same
PD, score, tier and reason codes for the same input.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.data_prep import RAW_FEATURES, clean_and_engineer
from src.scorecard import pd_to_score, score_to_tier

MODEL_PATH = "models/xgb.joblib"

# Plain-language reasons for the features that can push a score down. When a
# lender declines an application, ECOA / Regulation B requires telling the
# applicant the principal reasons, so each model feature maps to a reason a
# person could act on. Several engineered features share a reason.
REASON_TEXT = {
    "RevolvingUtilizationOfUnsecuredLines": "High balances relative to credit limits",
    "RevolvingUtilization_capped": "High balances relative to credit limits",
    "NumberOfTime30-59DaysPastDueNotWorse": "Recent 30-59 day late payments",
    "NumberOfTime60-89DaysPastDueNotWorse": "Recent 60-89 day late payments",
    "NumberOfTimes90DaysLate": "Serious delinquency (90+ days late)",
    "TotalTimesPastDue": "Number of late payments",
    "HasPastDueHistory": "History of late payments",
    "DebtRatio": "Debt payments high relative to income",
    "DebtRatio_log": "Debt payments high relative to income",
    "MonthlyIncome": "Income level",
    "MonthlyIncome_log": "Income level",
    "MonthlyIncome_missing": "Income not reported",
    "IncomePerDependent": "Income relative to number of dependents",
    "NumberOfDependents": "Number of dependents",
    "NumberOfDependents_missing": "Dependents not reported",
    "age": "Length of credit profile (age)",
    "NumberOfOpenCreditLinesAndLoans": "Number of open accounts",
    "CreditLinesPerYearAge": "Number of accounts relative to age",
    "NumberRealEstateLoansOrLines": "Number of real estate loans",
    "HasRealEstate": "No real estate loans on file",
}


# These reasons only make sense when the applicant actually has the adverse
# value (a late payment, a missing field). A tree model can give a small
# positive contribution to "late payments" for someone with none, and that
# must never show up on an adverse action notice.
REQUIRES_POSITIVE_VALUE = {
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "TotalTimesPastDue",
    "HasPastDueHistory",
    "MonthlyIncome_missing",
    "NumberOfDependents_missing",
}

# Contributions smaller than this (in log-odds) are noise, not reasons.
MIN_CONTRIBUTION = 0.05


@lru_cache(maxsize=1)
def load_bundle(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)


@lru_cache(maxsize=1)
def _explainer():
    import shap

    return shap.TreeExplainer(load_bundle()["model"])


def reason_codes(contributions: pd.Series, values: pd.Series, n: int = 4) -> list[str]:
    """Top distinct reasons that increased risk, largest first."""
    reasons: list[str] = []
    for feature, contribution in contributions.sort_values(ascending=False).items():
        if contribution < MIN_CONTRIBUTION:
            break
        if feature in REQUIRES_POSITIVE_VALUE and not values[feature] > 0:
            continue
        text = REASON_TEXT.get(feature, feature)
        if text not in reasons:
            reasons.append(text)
        if len(reasons) == n:
            break
    return reasons


def score_applicant(applicant: dict, explain: bool = True) -> dict:
    """Score one applicant given the ten raw bureau-style fields.

    Missing income or dependents can be passed as None; they are imputed with
    the training-set values the same way they were during training.
    """
    bundle = load_bundle()
    raw = pd.DataFrame([{col: applicant.get(col) for col in RAW_FEATURES}])
    raw = raw.astype(float)
    X = clean_and_engineer(raw, bundle["cleaning_params"])[bundle["features"]]

    pd_default = float(bundle["model"].predict_proba(X)[0, 1])
    score = float(pd_to_score(pd_default))
    result = {
        "probability_of_default": pd_default,
        "score": round(score),
        "tier": score_to_tier(score),
    }
    if explain:
        shap_values = _explainer()(X).values[0]
        contributions = pd.Series(shap_values, index=bundle["features"])
        result["contributions"] = contributions
        result["reasons"] = reason_codes(contributions, X.iloc[0])
    return result


def score_batch(df: pd.DataFrame) -> np.ndarray:
    """PD for a frame of raw records (no explanations)."""
    bundle = load_bundle()
    X = clean_and_engineer(df, bundle["cleaning_params"])[bundle["features"]]
    return bundle["model"].predict_proba(X)[:, 1]
