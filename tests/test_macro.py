import numpy as np
import pandas as pd
import pytest

from src.macro import (
    fit_satellite,
    lifetime_logit_shift,
    scenarios,
    stress_pd,
    unemployment_path,
)


@pytest.fixture
def synthetic_quarterly():
    # Charge-offs generated from a known relationship, so the fit can be checked.
    idx = pd.date_range("1985-01-01", "2019-10-01", freq="QS")
    t = np.arange(len(idx))
    ur = 6 + 2 * np.sin(t / 8)
    change = ur - pd.Series(ur).shift(4).fillna(ur[0]).to_numpy()
    logit = -3.5 + 0.06 * ur + 0.2 * change
    co = 100 / (1 + np.exp(-logit))
    return pd.DataFrame({"unemployment": ur, "chargeoff_rate": co}, index=idx)


def test_satellite_recovers_known_coefficients(synthetic_quarterly):
    sat = fit_satellite(synthetic_quarterly)
    assert sat["beta_level"] == pytest.approx(0.06, abs=1e-6)
    assert sat["beta_change"] == pytest.approx(0.2, abs=1e-6)
    assert sat["r_squared"] > 0.99


def test_unemployment_path_shape():
    path = unemployment_path(4.0, 10.0, 6, 2, 8.0)
    assert len(path) == 12
    assert path.max() == 10.0
    assert path[-1] == pytest.approx(8.0)


def test_scenarios_are_ordered(synthetic_quarterly):
    sat = fit_satellite(synthetic_quarterly)
    shifts = [lifetime_logit_shift(sat, p, 4.0) for p in scenarios(4.0).values()]
    assert shifts[0] == pytest.approx(0.0)
    assert shifts[0] < shifts[1] < shifts[2]


def test_stress_pd_moves_in_the_right_direction():
    base = np.array([0.02, 0.1, 0.3])
    assert (stress_pd(base, 0.5) > base).all()
    np.testing.assert_allclose(stress_pd(base, 0.0), base)
