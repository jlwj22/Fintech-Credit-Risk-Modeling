"""
Probability-of-default -> credit score conversion.

Uses the industry-standard "points to double the odds" (PDO) log-odds
scaling that underpins commercial credit scorecards across the lending
industry. It is the same transform, not an attempt to reproduce any specific
proprietary score. The point is to turn a model's probability into a number
underwriters can set cutoffs on.

    odds  = (1 - PD) / PD
    score = Offset + Factor * ln(odds)
    Factor = PDO / ln(2)
    Offset = BaseScore - Factor * ln(BaseOdds)

With BaseScore=720, BaseOdds=50 (50 good : 1 bad) and PDO=40, a 2% PD lands
near 720 and the model's calibrated PD range spreads across the familiar
300-850 window. The tier cutoffs then correspond roughly to:

    750+  PD below ~1.2%      Excellent
    700   PD ~2.8%            Good
    650   PD ~6.3%            Fair
    580   PD ~18%             Poor / Very Poor boundary
"""
from __future__ import annotations

import numpy as np

BASE_SCORE = 720
BASE_ODDS = 50.0
PDO = 40.0

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
    # Compare against lower bounds only so fractional scores such as 579.6
    # don't fall into the gap between two integer ranges.
    if score < SCORE_MIN or score > SCORE_MAX:
        return "Unknown"
    for lo, _hi, label in reversed(RISK_TIERS):
        if score >= lo:
            return label
    return "Unknown"


def pd_to_tier(pd_default: float) -> str:
    return score_to_tier(float(pd_to_score(pd_default)))
