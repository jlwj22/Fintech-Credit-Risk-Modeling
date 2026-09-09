"""
Probability-of-default -> credit score conversion.

Uses the industry-standard "points to double the odds" (PDO) log-odds
scaling that underpins commercial credit scorecards across the lending
industry -- card issuers, bureaus, and bank-built scorecards alike. This is
the same transform, not a claim to reproduce any specific proprietary model
-- the point is to translate a model's raw probability output into a
business-friendly score that risk teams and underwriters actually use.

    odds  = (1 - PD) / PD
    score = Offset + Factor * ln(odds)
    Factor = PDO / ln(2)
    Offset = BaseScore - Factor * ln(BaseOdds)

With BaseScore=600, BaseOdds=50 (50 good : 1 bad), PDO=20, the score range
for this dataset's PD distribution lands roughly in the familiar [300, 850]
consumer credit-score window.
"""
from __future__ import annotations

import numpy as np

BASE_SCORE = 600
BASE_ODDS = 50.0
PDO = 20.0

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

SCORE_MIN = 300
SCORE_MAX = 850

RISK_TIERS = [
    (300, 579, "Very Poor"),
    (580, 649, "Poor"),
    (650, 699, "Fair"),
    (700, 749, "Good"),
    (750, 850, "Excellent"),
]


def pd_to_score(pd_default: np.ndarray | float) -> np.ndarray:
    """Convert predicted probability of default (PD) to a scaled score."""
    pd_default = np.clip(np.asarray(pd_default, dtype=float), 1e-6, 1 - 1e-6)
    odds = (1 - pd_default) / pd_default
    score = OFFSET + FACTOR * np.log(odds)
    return np.clip(score, SCORE_MIN, SCORE_MAX)


def score_to_tier(score: float) -> str:
    for lo, hi, label in RISK_TIERS:
        if lo <= score <= hi:
            return label
    return "Unknown"


def pd_to_tier(pd_default: float) -> str:
    return score_to_tier(float(pd_to_score(pd_default)))
