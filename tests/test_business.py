import numpy as np
import pandas as pd
import pytest

from src.business import (
    CostAssumptions,
    LossAssumptions,
    cutoff_table,
    estimate_loss_assumptions,
    expected_loss,
    expected_profit,
)

A = LossAssumptions(lgd=0.9, ead_ratio=0.6, interest_before_default=0.15,
                    paid_interest_share=0.8)
NO_COSTS = CostAssumptions(servicing_fee=0, funding_cost_annual=0)


def test_expected_loss_formula():
    assert expected_loss(0.1, 10_000, A) == pytest.approx(0.1 * 0.9 * 0.6 * 10_000)


def test_expected_profit_falls_as_pd_rises():
    profits = expected_profit(np.array([0.01, 0.1, 0.4]), 10_000, 330, A)
    assert profits[0] > profits[1] > profits[2]


def test_expected_profit_without_default_or_costs_is_prepay_adjusted_interest():
    # PD 0: collect 80% of scheduled interest (330 * 36 - 10,000 = 1,880).
    assert expected_profit(0.0, 10_000, 330, A, NO_COSTS) == pytest.approx(0.8 * 1_880)


def test_estimate_loss_assumptions():
    loans = pd.DataFrame({
        "default": [1, 1, 0],
        "funded_amnt": [1000.0, 1000.0, 1000.0],
        "exposure_at_default": [600.0, 400.0, 0.0],
        "net_recovery": [100.0, 0.0, 0.0],
        "total_rec_int": [150.0, 50.0, 100.0],
        "installment": [33.0, 33.0, 32.0],
    })
    a = estimate_loss_assumptions(loans)
    assert a.lgd == pytest.approx(1 - 100 / 1000)
    assert a.ead_ratio == pytest.approx(1000 / 2000)
    assert a.interest_before_default == pytest.approx(200 / 2000)
    assert a.paid_interest_share == pytest.approx(100 / (32 * 36 - 1000))


def test_cutoff_table_approves_lowest_pd_first():
    pd_default = np.array([0.4, 0.05, 0.2, 0.1])
    default = np.array([1, 0, 0, 1])
    profit = np.array([-500.0, 100.0, 80.0, -300.0])
    funded = np.full(4, 1000.0)
    table = cutoff_table(pd_default, default, profit, funded, approval_rates=[0.25, 0.5, 1.0])
    assert table["pd_cutoff"].tolist() == [0.05, 0.1, 0.4]
    assert table["profit"].tolist() == [100.0, -200.0, -620.0]
    assert table["bad_rate"].iloc[-1] == 0.5
