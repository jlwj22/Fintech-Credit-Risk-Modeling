"""
Turn PDs into lending decisions: expected loss, expected profit, and cutoffs.

    EL = PD x LGD x EAD

LGD, the EAD ratio, the interest a defaulter pays before charging off, and
the share of scheduled interest a paid-off loan actually delivers (borrowers
prepay) are all estimated from training-period loans, then applied to new
loans. Expected profit uses the loan's actual price (Lending Club's rate),
so the question being answered is the one an investor buying these loans
faces: at this price, is this loan worth funding?

Profit is economic profit: after a servicing fee on payments collected and a
charge for the money tied up in the loan. Without those costs almost every
Lending Club loan looks profitable and there is no real cutoff decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TERM_MONTHS = 36


@dataclass
class LossAssumptions:
    lgd: float  # share of exposure at default not recovered
    ead_ratio: float  # exposure at default / funded amount
    interest_before_default: float  # interest collected / funded, defaulters
    paid_interest_share: float  # interest collected / scheduled, paid-off loans

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass
class CostAssumptions:
    # Lending Club charged note investors 1% of each payment received.
    servicing_fee: float = 0.01
    # Funding plus operating cost, per year, on the outstanding balance.
    funding_cost_annual: float = 0.04
    # A 36-month amortizing loan carries roughly 1.5 years of its original
    # balance on average (less with prepayment, which is ignored here).
    avg_balance_years: float = 1.5

    @property
    def carry_cost(self) -> float:
        """Funding cost as a share of the funded amount over the loan's life."""
        return self.funding_cost_annual * self.avg_balance_years

    def to_dict(self) -> dict:
        return {**asdict(self), "carry_cost": round(self.carry_cost, 4)}


DEFAULT_COSTS = CostAssumptions()


def estimate_loss_assumptions(loans: pd.DataFrame) -> LossAssumptions:
    """Estimate loss and prepayment assumptions from resolved loans."""
    defaults = loans[loans["default"] == 1]
    paid = loans[loans["default"] == 0]
    ead = defaults["exposure_at_default"].sum()
    scheduled = (paid["installment"] * TERM_MONTHS - paid["funded_amnt"]).sum()
    return LossAssumptions(
        lgd=float(1 - defaults["net_recovery"].sum() / ead),
        ead_ratio=float(ead / defaults["funded_amnt"].sum()),
        interest_before_default=float(
            defaults["total_rec_int"].sum() / defaults["funded_amnt"].sum()
        ),
        paid_interest_share=float(paid["total_rec_int"].sum() / scheduled),
    )


def expected_loss(pd_default, funded, a: LossAssumptions) -> np.ndarray:
    return np.asarray(pd_default) * a.lgd * a.ead_ratio * np.asarray(funded)


def expected_profit(pd_default, funded, installment, a: LossAssumptions,
                    costs: CostAssumptions = DEFAULT_COSTS) -> np.ndarray:
    """Expected lifetime economic profit per loan.

    Paid in full: collect the scheduled interest, scaled down for prepayment.
    Default: collect some interest first, then lose LGD x EAD.
    Then subtract servicing on what is collected and the carry cost.
    """
    pd_default = np.asarray(pd_default)
    funded = np.asarray(funded)
    paid_interest = a.paid_interest_share * (np.asarray(installment) * TERM_MONTHS - funded)
    default_outcome = a.interest_before_default * funded - a.lgd * a.ead_ratio * funded
    gross = (1 - pd_default) * paid_interest + pd_default * default_outcome
    collected = funded + gross
    return gross - costs.servicing_fee * collected - costs.carry_cost * funded


def economic_profit(df: pd.DataFrame, costs: CostAssumptions = DEFAULT_COSTS) -> pd.Series:
    """Realized profit after the same servicing and carry costs."""
    return (
        df["realized_profit"]
        - costs.servicing_fee * df["total_pymnt"]
        - costs.carry_cost * df["funded_amnt"]
    )


def cutoff_table(
    pd_default, default, profit, funded, approval_rates=None
) -> pd.DataFrame:
    """Approve the lowest-PD share of applicants and report what happens.

    One row per approval rate: the PD cutoff, the realized bad rate among
    approved loans, and realized profit and return on funded dollars.
    """
    if approval_rates is None:
        approval_rates = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    order = np.argsort(pd_default)
    pd_sorted = np.asarray(pd_default)[order]
    default_sorted = np.asarray(default)[order]
    profit_sorted = np.asarray(profit)[order]
    funded_sorted = np.asarray(funded)[order]

    rows = []
    n = len(order)
    for rate in approval_rates:
        k = max(1, int(round(rate * n)))
        rows.append({
            "approval_rate": rate,
            "pd_cutoff": pd_sorted[k - 1],
            "loans": k,
            "bad_rate": default_sorted[:k].mean(),
            "funded": funded_sorted[:k].sum(),
            "profit": profit_sorted[:k].sum(),
            "return_on_funded": profit_sorted[:k].sum() / funded_sorted[:k].sum(),
        })
    return pd.DataFrame(rows)


def summarize_strategy(name: str, approved: np.ndarray, df: pd.DataFrame,
                       profit_col: str = "economic_profit") -> dict:
    """Realized results of an approve/decline rule on a labeled portfolio."""
    a = df[approved]
    return {
        "strategy": name,
        "approval_rate": approved.mean(),
        "loans": int(approved.sum()),
        "bad_rate": a["default"].mean(),
        "funded": a["funded_amnt"].sum(),
        "profit": a[profit_col].sum(),
        "return_on_funded": a[profit_col].sum() / a["funded_amnt"].sum(),
    }
