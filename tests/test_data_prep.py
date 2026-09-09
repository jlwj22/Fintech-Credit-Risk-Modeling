import numpy as np
import pandas as pd
import pytest

from src.data_prep import MODEL_FEATURES, PAST_DUE_COLS, TARGET, clean_and_engineer


@pytest.fixture
def raw_sample():
    return pd.DataFrame(
        {
            "SeriousDlqin2yrs": [0, 1, 0],
            "RevolvingUtilizationOfUnsecuredLines": [0.2, 5000.0, 0.5],
            "age": [0, 45, 60],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 98, 1],
            "DebtRatio": [0.3, 500000.0, 0.6],
            "MonthlyIncome": [5000.0, np.nan, 3000.0],
            "NumberOfOpenCreditLinesAndLoans": [5, 10, 3],
            "NumberOfTimes90DaysLate": [0, 96, 0],
            "NumberRealEstateLoansOrLines": [1, 0, 2],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 98, 0],
            "NumberOfDependents": [0, np.nan, 2],
        }
    )


def test_clean_and_engineer_adds_expected_columns(raw_sample):
    out = clean_and_engineer(raw_sample)
    for col in MODEL_FEATURES:
        assert col in out.columns, f"missing engineered column: {col}"


def test_no_missing_values_after_cleaning(raw_sample):
    out = clean_and_engineer(raw_sample)
    assert out[MODEL_FEATURES].isna().sum().sum() == 0


def test_sentinel_codes_are_capped(raw_sample):
    out = clean_and_engineer(raw_sample)
    for col in PAST_DUE_COLS:
        assert out[col].max() < 96


def test_zero_age_is_imputed(raw_sample):
    out = clean_and_engineer(raw_sample)
    assert (out["age"] >= 18).all()


def test_missing_flags_are_binary(raw_sample):
    out = clean_and_engineer(raw_sample)
    assert set(out["MonthlyIncome_missing"].unique()) <= {0, 1}
    assert set(out["NumberOfDependents_missing"].unique()) <= {0, 1}


def test_target_untouched(raw_sample):
    out = clean_and_engineer(raw_sample)
    pd.testing.assert_series_equal(out[TARGET], raw_sample[TARGET])
