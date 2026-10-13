from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from measurement.geo.synthetic_control import fit_weights, placebo_inference, synthetic_control


def _panel(lift: float, seed: int = 0, n_markets: int = 12, T: int = 40):
    rng = np.random.default_rng(seed)
    weeks = [date(2026, 1, 5) + timedelta(weeks=i) for i in range(T)]
    common = 1 + 0.1 * np.sin(np.arange(T) / 4)
    data = {}
    for i in range(n_markets):
        level = rng.uniform(80, 400)
        data[f"m{i:02d}"] = level * common * (1 + rng.normal(0, 0.03, T))
    df = pd.DataFrame(data, index=pd.Index(weeks, name="week_start"))
    treat = ["m00", "m01"]
    post = np.arange(T) >= T - 8
    df.loc[post, treat] = df.loc[post, treat] * (1 + lift)
    return df, treat, weeks[0], weeks[T - 8], weeks[-1]


def test_weights_are_a_convex_combination():
    rng = np.random.default_rng(2)
    x = rng.uniform(size=(20, 5))
    y = x @ np.array([0.5, 0.5, 0, 0, 0])
    w = fit_weights(y, x)
    assert np.all(w >= 0) and np.isclose(w.sum(), 1.0)
    assert np.isclose(w[0] + w[1], 1.0, atol=0.05)


def test_recovers_injected_lift():
    df, treat, pre_start, test_start, test_end = _panel(0.10)
    controls = [c for c in df.columns if c not in treat]
    res = synthetic_control(df, treat, controls, pre_start, test_start, test_end)
    assert abs(res.lift_pct - 0.10) < 0.03
    assert res.pre_rmspe < 0.05
    assert len(res.post_weeks) == 8


def test_placebo_p_value_separates_signal_from_noise():
    df, treat, pre_start, test_start, test_end = _panel(0.25)
    controls = [c for c in df.columns if c not in treat]
    res = synthetic_control(df, treat, controls, pre_start, test_start, test_end)
    inf = placebo_inference(df, treat, controls, pre_start, test_start, test_end, res, n_permutations=30, seed=1)
    assert inf["p_value"] < 0.1
    assert inf["ci_low"] > 0

    # Under the null the placebo p-value is roughly uniform, so check it across several draws.
    p_values, covered = [], 0
    for seed in range(5):
        df0, treat, pre_start, test_start, test_end = _panel(0.0, seed=10 + seed)
        res0 = synthetic_control(df0, treat, controls, pre_start, test_start, test_end)
        inf0 = placebo_inference(df0, treat, controls, pre_start, test_start, test_end, res0, n_permutations=30, seed=1)
        p_values.append(inf0["p_value"])
        covered += inf0["ci_low"] < 0 < inf0["ci_high"]
    assert sum(p_values) / len(p_values) > 0.25
    assert covered >= 4


def test_rejects_short_pre_period():
    df, treat, _, test_start, test_end = _panel(0.1)
    controls = [c for c in df.columns if c not in treat]
    with pytest.raises(ValueError):
        synthetic_control(df, treat, controls, test_start - timedelta(weeks=2), test_start, test_end)
