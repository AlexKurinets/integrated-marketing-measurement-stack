from datetime import date, timedelta

import numpy as np
import pandas as pd

from measurement.geo.power import minimum_detectable_effect, power_analysis


def _history(seed: int = 0, n_markets: int = 14, T: int = 36):
    rng = np.random.default_rng(seed)
    weeks = [date(2025, 6, 2) + timedelta(weeks=i) for i in range(T)]
    common = 1 + 0.08 * np.sin(np.arange(T) / 5)
    data = {f"m{i:02d}": rng.uniform(80, 300) * common * (1 + rng.normal(0, 0.03, T)) for i in range(n_markets)}
    return pd.DataFrame(data, index=pd.Index(weeks, name="week_start"))


def test_power_rises_with_lift():
    table = power_analysis(
        _history(),
        n_treat=2,
        pre_weeks=16,
        test_weeks=6,
        lifts=[0.0, 0.3],
        simulations=12,
        alpha=0.1,
        placebo_permutations=8,
        seed=3,
    )
    assert list(table.lift) == [0.0, 0.3]
    assert table.power.iloc[1] >= table.power.iloc[0]
    assert table.power.iloc[1] >= 0.75
    assert table.power.iloc[0] <= 0.4


def test_minimum_detectable_effect():
    table = pd.DataFrame({"lift": [0.05, 0.10, 0.20], "power": [0.3, 0.85, 0.99]})
    assert minimum_detectable_effect(table, 0.8) == 0.10
    assert minimum_detectable_effect(table, 0.999) is None
