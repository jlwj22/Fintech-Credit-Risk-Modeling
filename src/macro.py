"""
Macro stress testing with a simple satellite model.

Banks stress their loss forecasts by linking portfolio default rates to
macro variables, then pushing those variables through scenarios (the Fed's
CCAR / DFAST exercises work this way). This is a small version of that:

1. Fit, on quarterly FRED data from 1985-2019,

       logit(card charge-off rate) = a + b1 * UR + b2 * (UR - UR a year ago)

   Losses respond to unemployment *rising*, not just to its level: the
   change term takes R^2 from about 0.15 (level only) to about 0.5.
   2020-2021 is left out because stimulus and forbearance pushed charge-offs
   down while unemployment spiked, which would distort the relationship.
2. Define each scenario as a 12-quarter unemployment path over the life of a
   36-month loan (rise, peak, partial recovery), the way supervisory
   scenarios are written.
3. Run the path and a flat baseline through the satellite model, average the
   implied loss rate over the 12 quarters, and shift every loan's PD in
   log-odds space by the difference.

The satellite model is fit on bank-wide card charge-offs, not on this
portfolio, so step 3 assumes Lending Club borrowers respond to the cycle in
the same proportion as card borrowers. That is the main simplification and
it is stated wherever results are shown.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.download_data import FRED_DIR

FIT_START, FIT_END = "1985-01-01", "2019-12-31"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def load_quarterly(fred_dir: str = FRED_DIR) -> pd.DataFrame:
    ur = pd.read_csv(os.path.join(fred_dir, "UNRATE.csv"), parse_dates=["observation_date"])
    ur = ur.set_index("observation_date")["UNRATE"].resample("QS").mean()
    co = pd.read_csv(os.path.join(fred_dir, "CORCCACBS.csv"), parse_dates=["observation_date"])
    co = co.set_index("observation_date")["CORCCACBS"]
    return pd.concat({"unemployment": ur, "chargeoff_rate": co}, axis=1).dropna()


def _design(ur: pd.Series) -> np.ndarray:
    return np.column_stack([np.ones(len(ur)), ur, ur - ur.shift(4)])


def fit_satellite(q: pd.DataFrame) -> dict:
    fit = q.loc[FIT_START:FIT_END]
    X = _design(fit["unemployment"])
    y = _logit(fit["chargeoff_rate"] / 100)
    keep = ~np.isnan(X).any(axis=1)
    coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    resid = y[keep] - X[keep] @ coef
    return {
        "intercept": float(coef[0]),
        "beta_level": float(coef[1]),
        "beta_change": float(coef[2]),
        "r_squared": float(1 - resid.var() / y[keep].var()),
        "quarters": int(keep.sum()),
        "latest_unemployment": round(float(q["unemployment"].iloc[-1]), 1),
        "latest_quarter": str(q.index[-1].date()),
    }


def fitted_chargeoff(sat: dict, ur: pd.Series) -> np.ndarray:
    """Fitted charge-off rate (as a fraction) for a quarterly UR series."""
    b = np.array([sat["intercept"], sat["beta_level"], sat["beta_change"]])
    return _expit(_design(ur) @ b)


def unemployment_path(base: float, peak: float, quarters_to_peak: int,
                      quarters_at_peak: int, end: float, n: int = 12) -> np.ndarray:
    rise = np.linspace(base, peak, quarters_to_peak + 1)[1:]
    hold = np.full(quarters_at_peak, peak)
    fall = np.linspace(peak, end, n - quarters_to_peak - quarters_at_peak + 1)[1:]
    return np.concatenate([rise, hold, fall])[:n]


def scenarios(base: float) -> dict[str, np.ndarray]:
    """12-quarter unemployment paths, loosely shaped like the Fed's
    supervisory scenarios. The severe path peaks at 10%, as in 2009."""
    severe_peak = max(10.0, base + 4.0)
    return {
        "Baseline": np.full(12, base),
        "Mild recession": unemployment_path(base, base + 2.0, 4, 4, base + 1.0),
        "Severe recession": unemployment_path(base, severe_peak, 6, 2, severe_peak - 2.0),
    }


def lifetime_logit_shift(sat: dict, path: np.ndarray, base: float) -> float:
    """Log-odds shift to apply to lifetime PD under ``path`` vs. a flat base."""
    def avg_loss(p):
        ur = pd.Series(np.concatenate([np.full(4, base), p]))
        return fitted_chargeoff(sat, ur)[4:].mean()

    return float(_logit(avg_loss(path)) - _logit(avg_loss(np.full(len(path), base))))


def stress_pd(pd_default, logit_shift: float) -> np.ndarray:
    return _expit(_logit(pd_default) + logit_shift)
