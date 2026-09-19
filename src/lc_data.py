"""
Lending Club loan data: load, label, and join state unemployment from FRED.

Scope: 36-month loans. Loans issued 2007-2015 have (almost) all reached a
final outcome by the end of the file in 2018Q4, so they get a default label.
Loans issued 2016-2018 are mostly still open; they are kept unlabeled and
used as the "production" population for drift monitoring.

Only fields known when the application comes in are used as model features.
State unemployment is joined from FRED for analysis and stress testing.
Lending Club's own grade and interest rate are kept for benchmarking and
pricing, but the model never sees them.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.download_data import FRED_DIR, LC_PATH

PROCESSED_PATH = "data/processed/lending_club.parquet"
TARGET = "default"

LOAD_COLS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade",
    "sub_grade", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "loan_status", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "application_type", "total_pymnt", "total_rec_prncp", "total_rec_int",
    "recoveries", "collection_recovery_fee",
    # Trended bureau fields Lending Club started reporting in 2012.
    "acc_open_past_24mths", "bc_util", "percent_bc_gt_75", "num_tl_op_past_12m",
    "mths_since_recent_inq", "mo_sin_rcnt_tl", "avg_cur_bal", "total_rev_hi_lim",
    "num_actv_rev_tl", "pct_tl_nvr_dlq",
]

DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
PAID_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc_log",
    "loan_to_income",
    "dti",
    "fico",
    "emp_years",
    "credit_history_years",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "total_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "revol_bal_log",
    "revol_util",
    "mort_acc",
    "acc_open_past_24mths",
    "bc_util",
    "percent_bc_gt_75",
    "num_tl_op_past_12m",
    "mths_since_recent_inq",
    "mo_sin_rcnt_tl",
    "avg_cur_bal_log",
    "total_rev_hi_lim_log",
    "num_actv_rev_tl",
    "pct_tl_nvr_dlq",
]
# Joined from FRED. Tested as model features (src/lc_train.py runs the
# ablation) but they add no lift and drift heavily as the economy moves, so
# the PD model leaves them out and macro risk is handled by the stress
# overlay in src/macro.py instead.
MACRO_FEATURES = ["state_unemployment", "state_unemployment_change_12m"]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fixed category lists so train, test and scoring data share one encoding.
CATEGORIES = {
    "home_ownership": ["MORTGAGE", "RENT", "OWN", "OTHER"],
    "verification_status": ["Not Verified", "Source Verified", "Verified"],
    "purpose": [
        "debt_consolidation", "credit_card", "home_improvement", "other",
        "major_purchase", "small_business", "car", "medical", "moving",
        "vacation", "house", "wedding", "renewable_energy", "educational",
    ],
}


def load_state_unemployment(fred_dir: str = FRED_DIR) -> pd.DataFrame:
    """Monthly unemployment rate per state, long format."""
    frames = []
    for fname in sorted(os.listdir(fred_dir)):
        # State series are two letters + "UR"; skip national series.
        if len(fname) != 8 or not fname.endswith("UR.csv"):
            continue
        s = pd.read_csv(os.path.join(fred_dir, fname), parse_dates=["observation_date"])
        s.columns = ["month", "state_unemployment"]
        s["addr_state"] = fname[:2]
        s = s.sort_values("month")
        s["state_unemployment_change_12m"] = s["state_unemployment"].diff(12)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def load_raw(path: str = LC_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=LOAD_COLS, low_memory=False)
    df = df[df["term"].str.strip() == "36 months"]
    df = df[df["application_type"] == "Individual"]
    return df.reset_index(drop=True)


def engineer(df: pd.DataFrame, state_ur: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["issue_year"] = df["issue_date"].dt.year

    status = df["loan_status"]
    df[TARGET] = np.where(
        status.isin(DEFAULT_STATUSES), 1, np.where(status.isin(PAID_STATUSES), 0, np.nan)
    )

    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_years"] = (df["issue_date"] - earliest).dt.days / 365.25
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["emp_years"] = (
        df["emp_length"]
        .str.replace("< 1 year", "0", regex=False)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )
    df["annual_inc_log"] = np.log1p(df["annual_inc"])
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    df["revol_bal_log"] = np.log1p(df["revol_bal"])
    df["avg_cur_bal_log"] = np.log1p(df["avg_cur_bal"])
    df["total_rev_hi_lim_log"] = np.log1p(df["total_rev_hi_lim"])
    df["home_ownership"] = df["home_ownership"].replace({"NONE": "OTHER", "ANY": "OTHER"})

    # Unemployment in the borrower's state, using the prior month's print since
    # that is what would have been published when the loan was underwritten.
    if state_ur is not None:
        ur = state_ur.copy()
        ur["issue_date"] = ur["month"] + pd.DateOffset(months=1)
        df = df.merge(
            ur[["addr_state", "issue_date", "state_unemployment",
                "state_unemployment_change_12m"]],
            on=["addr_state", "issue_date"],
            how="left",
        )

    # Realized economics, used by the business layer (never as features).
    df["net_recovery"] = df["recoveries"] - df["collection_recovery_fee"]
    df["realized_profit"] = df["total_pymnt"] - df["collection_recovery_fee"] - df["funded_amnt"]
    df["exposure_at_default"] = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0)

    return df


def to_model_frame(df: pd.DataFrame, features: list[str] = MODEL_FEATURES) -> pd.DataFrame:
    """Feature matrix with fixed categorical encodings (XGBoost handles NaN)."""
    X = df[features].copy()
    for col, cats in CATEGORIES.items():
        if col in X:
            X[col] = pd.Categorical(X[col], categories=cats)
    return X


def prepare(save_path: str | None = PROCESSED_PATH) -> pd.DataFrame:
    df = engineer(load_raw(), load_state_unemployment())
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path)
    return df


if __name__ == "__main__":
    out = prepare()
    labeled = out[out[TARGET].notna()]
    print(f"{len(out):,} 36-month loans, {len(labeled):,} with a final outcome")
    print(labeled.groupby("issue_year")[TARGET].agg(["size", "mean"]).round(3))
