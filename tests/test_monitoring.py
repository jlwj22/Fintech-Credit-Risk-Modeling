import numpy as np
import pandas as pd

from src.monitoring import psi, psi_categorical, psi_report, psi_status


def test_identical_distributions_have_near_zero_psi():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20_000)
    assert psi(x, rng.permutation(x)) < 0.001


def test_shifted_distribution_is_flagged():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=20_000), rng.normal(1.0, 1, size=20_000)) > 0.25


def test_change_in_missing_rate_counts_as_drift():
    ref = pd.Series(np.arange(1000, dtype=float))
    cur = ref.copy()
    cur[:300] = np.nan
    assert psi(ref, cur) > 0.1


def test_categorical_psi():
    assert psi_categorical(list("aabb"), list("aabb")) == 0
    assert psi_categorical(list("aaab"), list("abbb")) > 0.25


def test_status_thresholds():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.15) == "moderate shift"
    assert psi_status(0.4) == "significant shift"


def test_report_sorted_by_psi():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(size=5000)})
    cur = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(2, 1, size=5000)})
    report = psi_report(ref, cur, ["a", "b"])
    assert report["variable"].tolist() == ["b", "a"]
