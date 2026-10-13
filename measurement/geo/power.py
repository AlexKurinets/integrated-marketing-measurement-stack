"""Simulation-based power analysis for geo tests.

Before a test spends anything, we want to know the smallest lift it can detect. The approach is to replay
history: pick random treatment markets and a random test window, inject a known lift into the treated
outcomes, run the same synthetic control readout we would run for real, and count how often the lift is
detected at the chosen alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from measurement.geo.matching import control_pool, match_markets
from measurement.geo.synthetic_control import placebo_inference, synthetic_control


def power_analysis(
    outcome_wide: pd.DataFrame,
    n_treat: int,
    pre_weeks: int,
    test_weeks: int,
    lifts: list[float],
    simulations: int = 100,
    alpha: float = 0.1,
    n_controls: int = 3,
    placebo_permutations: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = list(pd.to_datetime(outcome_wide.index).date)
    markets = list(outcome_wide.columns)
    T = len(weeks)
    if T < pre_weeks + test_weeks:
        raise ValueError("Not enough history for the requested pre-period and test length")

    rows = []
    for lift in lifts:
        detected = 0
        for _ in range(simulations):
            treat = rng.choice(markets, size=n_treat, replace=False).tolist()
            start = int(rng.integers(0, T - pre_weeks - test_weeks + 1))
            pre_start = weeks[start]
            test_start = weeks[start + pre_weeks]
            test_end = weeks[start + pre_weeks + test_weeks - 1]
            data = outcome_wide.copy()
            idx = pd.to_datetime(data.index).date
            post = (idx >= test_start) & (idx <= test_end)
            data.loc[post, treat] = data.loc[post, treat] * (1.0 + lift)
            matches = match_markets(data.loc[(idx >= pre_start) & (idx < test_start)], treat, n_controls=n_controls)
            controls = control_pool(matches)
            if len(controls) < n_treat + 3:
                controls = [m for m in markets if m not in treat]
            res = synthetic_control(data, treat, controls, pre_start, test_start, test_end)
            inf = placebo_inference(
                data,
                treat,
                controls,
                pre_start,
                test_start,
                test_end,
                res,
                n_permutations=placebo_permutations,
                seed=int(rng.integers(0, 1_000_000)),
            )
            if inf["p_value"] < alpha and res.lift_abs > 0:
                detected += 1
        rows.append({"lift": lift, "power": detected / simulations, "simulations": simulations})
    return pd.DataFrame(rows)


def minimum_detectable_effect(power_table: pd.DataFrame, target_power: float = 0.8) -> float | None:
    ok = power_table[power_table.power >= target_power].sort_values("lift")
    return float(ok.lift.iloc[0]) if len(ok) else None
