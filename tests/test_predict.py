import pandas as pd

from src.predict import reason_codes, score_applicant

CLEAN = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 38,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 4500,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
}


def test_no_delinquency_reasons_without_delinquency():
    reasons = score_applicant(CLEAN)["reasons"]
    assert not any("late" in r.lower() or "delinquency" in r.lower() for r in reasons)
    assert "Income not reported" not in reasons


def test_reason_requires_adverse_value():
    contributions = pd.Series({"NumberOfTimes90DaysLate": 0.4, "DebtRatio": 0.2})
    values = pd.Series({"NumberOfTimes90DaysLate": 0, "DebtRatio": 0.9})
    assert reason_codes(contributions, values) == ["Debt payments high relative to income"]


def test_small_contributions_are_ignored():
    contributions = pd.Series({"DebtRatio": 0.01})
    assert reason_codes(contributions, pd.Series({"DebtRatio": 0.5})) == []


def test_duplicate_reasons_collapse():
    contributions = pd.Series({"DebtRatio": 0.3, "DebtRatio_log": 0.2})
    values = pd.Series({"DebtRatio": 0.9, "DebtRatio_log": 0.6})
    assert reason_codes(contributions, values) == ["Debt payments high relative to income"]
