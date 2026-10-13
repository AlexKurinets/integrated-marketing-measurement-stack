from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from measurement.geo.matching import control_pool, market_distance, match_markets, select_treatment_markets


@pytest.fixture
def wide():
    rng = np.random.default_rng(0)
    weeks = [date(2026, 1, 5) + timedelta(weeks=i) for i in range(30)]
    base = 100 + 30 * np.sin(np.arange(30) / 2)
    data = {
        "a": base + rng.normal(0, 1, 30),
        "a_twin": 0.5 * base + rng.normal(0, 0.5, 30),  # same shape, half the size
        "b": 100 + rng.normal(0, 8, 30),  # noise
        "c": 300 - 15 * np.sin(np.arange(30) / 2) + rng.normal(0, 1, 30),  # anti-correlated and bigger
        "d": base * 1.1 + rng.normal(0, 1, 30),  # near copy
    }
    return pd.DataFrame(data, index=pd.Index(weeks, name="week_start"))


def test_best_match_is_the_near_copy(wide):
    m = match_markets(wide, ["a"], n_controls=2)
    assert m.iloc[0].control == "d"
    assert m.correlation.iloc[0] > 0.8
    assert len(m) == 2


def test_exclusions_and_treatment_never_appear_as_controls(wide):
    m = match_markets(wide, ["a"], n_controls=4, exclude=("d",))
    assert "a" not in set(m.control)
    assert "d" not in set(m.control)
    assert control_pool(m)[0] == m.sort_values("score").control.iloc[0]


def test_scale_penalty_is_applied(wide):
    d_twin = market_distance(wide["a"], wide["a_twin"])
    d_copy = market_distance(wide["a"], wide["d"])
    assert d_twin["correlation"] > 0.8
    assert d_twin["score"] > d_copy["score"]


def test_select_treatment_markets_is_reproducible(wide):
    pop = pd.Series({"a": 1.0, "a_twin": 0.5, "b": 1.0, "c": 3.0, "d": 1.1})
    t1 = select_treatment_markets(wide, pop, n_treat=2, seed=4)
    t2 = select_treatment_markets(wide, pop, n_treat=2, seed=4)
    assert t1 == t2 and len(set(t1)) == 2
