"""
Data cleaning and feature engineering for the credit risk dataset.

Source: "Give Me Some Credit" (Kaggle), ~150k consumer credit records.
Target: SeriousDlqin2yrs -- borrower experienced 90+ days past due
delinquency (or worse) within two years of the credit report date.

This module is intentionally dependency-light (pandas/numpy only) so it can
be imported by both the training pipeline and the Streamlit app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/cs-training.csv"
PROCESSED_PATH = "data/processed/credit_risk_clean.parquet"

TARGET = "SeriousDlqin2yrs"

# Known sentinel/error codes in this dataset: a small number of records carry
# 96 or 98 in the "times past due" columns, which do not represent real
# counts (max legitimate value elsewhere in the data is ~17-24). We treat
# them as data-entry errors rather than dropping the rows outright.
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

ENGINEERED_FEATURES = [
    "MonthlyIncome_missing",
    "NumberOfDependents_missing",
    "TotalTimesPastDue",
    "HasPastDueHistory",
    "HasRealEstate",
    "IncomePerDependent",
    "DebtRatio_log",
    "MonthlyIncome_log",
    "RevolvingUtilization_capped",
    "CreditLinesPerYearAge",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Several numeric columns receive float values (quantile caps, medians)
    # during cleaning below. Cast up front so a cap/impute that happens to
    # land on a fractional value doesn't hit pandas' strict int64 setitem
    # dtype check -- on some samples a quantile lands on a whole number
    # (e.g. 6.0) and silently works, but that's not guaranteed in general.
    float_cols = PAST_DUE_COLS + ["age", "RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]
    df[float_cols] = df[float_cols].astype(float)

    # --- Sentinel/error-code handling -------------------------------------
    # 96/98 in the past-due columns are data-entry error codes, not counts.
    # Cap them at the highest plausible observed value instead of dropping
    # ~3k rows, and keep the raw column so the cap is auditable.
    for col in PAST_DUE_COLS:
        sentinel_mask = df[col].isin([96, 98])
        cap_value = df.loc[~sentinel_mask, col].quantile(0.999)
        df.loc[sentinel_mask, col] = cap_value

    # A single record has age == 0, which is impossible for a credit
    # applicant; impute with the median age.
    df.loc[df["age"] < 18, "age"] = df["age"].median()

    # RevolvingUtilizationOfUnsecuredLines should theoretically sit in
    # [0, ~1.5]; a handful of records report values in the thousands. Cap at
    # the 99.5th percentile of "plausible" values to avoid one outlier
    # dominating the model, and keep the capped version as its own feature.
    cap_util = df["RevolvingUtilizationOfUnsecuredLines"].quantile(0.995)
    df["RevolvingUtilization_capped"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=cap_util)

    # DebtRatio has the same issue (division blow-ups when MonthlyIncome is
    # ~0); cap before log-transforming.
    cap_debt = df["DebtRatio"].quantile(0.995)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=cap_debt)

    # --- Missing values ------------------------------------------------
    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    df["NumberOfDependents_missing"] = df["NumberOfDependents"].isna().astype(int)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # --- Feature engineering -----------------------------------------
    df["TotalTimesPastDue"] = df[PAST_DUE_COLS].sum(axis=1)
    df["HasPastDueHistory"] = (df["TotalTimesPastDue"] > 0).astype(int)
    df["HasRealEstate"] = (df["NumberRealEstateLoansOrLines"] > 0).astype(int)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["DebtRatio_log"] = np.log1p(df["DebtRatio"])
    df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"])
    df["CreditLinesPerYearAge"] = df["NumberOfOpenCreditLinesAndLoans"] / df["age"]

    return df


def prepare(raw_path: str = RAW_PATH, save_path: str | None = PROCESSED_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clean_and_engineer(df)
    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path)
    return df


if __name__ == "__main__":
    out = prepare()
    print(f"Prepared {len(out):,} rows, {out.shape[1]} columns -> {PROCESSED_PATH}")
    print(f"Default rate: {out[TARGET].mean():.3%}")
