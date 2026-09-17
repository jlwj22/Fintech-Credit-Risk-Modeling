import numpy as np

from src.scorecard import pd_to_score, pd_to_tier, score_to_tier


def test_lower_pd_gives_higher_score():
    low_risk = pd_to_score(0.01)
    high_risk = pd_to_score(0.5)
    assert low_risk > high_risk


def test_score_within_bounds():
    scores = pd_to_score(np.array([0.0001, 0.5, 0.9999]))
    assert (scores >= 300).all() and (scores <= 850).all()


def test_score_to_tier_boundaries():
    assert score_to_tier(300) == "Very Poor"
    assert score_to_tier(850) == "Excellent"
    assert score_to_tier(700) == "Good"


def test_pd_to_tier_consistency():
    tier = pd_to_tier(0.02)
    assert tier in {"Excellent", "Good", "Fair", "Poor", "Very Poor"}


def test_fractional_scores_between_tiers_are_assigned():
    assert score_to_tier(579.6) == "Very Poor"
    assert score_to_tier(649.5) == "Poor"
    assert score_to_tier(749.99) == "Good"


def test_two_percent_pd_near_base_score():
    assert abs(float(pd_to_score(0.02)) - 720) < 2
