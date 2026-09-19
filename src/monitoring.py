"""
Population Stability Index (PSI) for drift monitoring.

PSI compares the distribution of a score or feature in a new population
against the population the model was built on:

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Common rule of thumb in credit risk:
    < 0.10   stable
    0.10-0.25 moderate shift, investigate
    > 0.25   significant shift, the model may need recalibration or a rebuild
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def psi(expected, actual, bins: int = 10) -> float:
    """PSI of a numeric variable using quantile bins from ``expected``.

    Missing values get their own bin, since a change in missing rate is
    drift too.
    """
    expected = pd.Series(expected, dtype=float)
    actual = pd.Series(actual, dtype=float)

    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(s: pd.Series) -> np.ndarray:
        counts = pd.cut(s.dropna(), edges, include_lowest=True).value_counts(sort=False)
        counts = np.append(counts.to_numpy(), s.isna().sum())
        return np.clip(counts / len(s), EPS, None)

    e, a = shares(expected), shares(actual)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual) -> float:
    e = pd.Series(expected).astype(str).value_counts(normalize=True)
    a = pd.Series(actual).astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    e = e.reindex(idx, fill_value=0).clip(lower=EPS)
    a = a.reindex(idx, fill_value=0).clip(lower=EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "significant shift"


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        if isinstance(reference[col].dtype, pd.CategoricalDtype) or reference[col].dtype == object:
            value = psi_categorical(reference[col], current[col])
        else:
            value = psi(reference[col], current[col])
        rows.append({"variable": col, "psi": round(value, 4), "status": psi_status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
