"""Geo test readout: incremental conversions, cost per incremental conversion and the MMM calibration target."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from measurement.config import geo_config
from measurement.geo.matching import control_pool, match_markets
from measurement.geo.synthetic_control import placebo_inference, synthetic_control
from measurement.mmm.data import Panel


def design_test(
    panel: Panel,
    test_id: str,
    name: str,
    channel: str,
    treatment: list[str],
    pre_start: date,
    test_start: date,
    test_end: date,
    n_controls: int | None = None,
) -> dict:
    """Pick matched controls for the chosen treatment markets and return the test record."""
    cfg = geo_config()["design"]
    wide = panel.outcome_wide()
    idx = pd.to_datetime(wide.index).date
    pre = wide[(idx >= pre_start) & (idx < test_start)]
    matches = match_markets(
        pre,
        treatment,
        n_controls=n_controls or cfg["controls_per_treatment"],
        min_correlation=cfg["min_pre_period_correlation"],
    )
    controls = control_pool(matches)
    if len(controls) < len(treatment) + 3:
        matches = match_markets(pre, treatment, n_controls=n_controls or cfg["controls_per_treatment"])
        controls = control_pool(matches)
    return {
        "test_id": test_id,
        "name": name,
        "channel": channel,
        "treatment_markets": list(treatment),
        "control_markets": controls,
        "pre_start": pre_start,
        "test_start": test_start,
        "test_end": test_end,
        "incremental_spend": 0.0,
        "status": "designed",
        "readout": None,
    }


def incremental_spend(panel: Panel, test: dict) -> float:
    """Actual minus counterfactual spend in the treatment markets over the test weeks (negative for a holdout)."""
    c = panel.channels.index(test["channel"])
    treat = np.array([m in test["treatment_markets"] for m in panel.markets])
    weeks = panel.week_mask(pd.to_datetime(test["test_start"]).date(), pd.to_datetime(test["test_end"]).date())
    others = ~treat
    per_capita = panel.spend[others][:, :, c].sum(axis=0) / max(panel.population[others].sum(), 1.0)
    actual = panel.spend[treat][:, weeks, c].sum()
    cf = (per_capita[weeks][None, :] * panel.population[treat][:, None]).sum()
    return float(actual - cf)


def read_geo_test(panel: Panel, test: dict, seed: int = 0) -> dict:
    cfg = geo_config()["readout"]
    wide = panel.outcome_wide()
    pre_start = pd.to_datetime(test["pre_start"]).date()
    test_start = pd.to_datetime(test["test_start"]).date()
    test_end = pd.to_datetime(test["test_end"]).date()
    treatment = list(test["treatment_markets"])
    controls = list(test["control_markets"])
    res = synthetic_control(wide, treatment, controls, pre_start, test_start, test_end)
    inf = placebo_inference(
        wide,
        treatment,
        controls,
        pre_start,
        test_start,
        test_end,
        res,
        n_permutations=int(cfg["placebo_permutations"]),
        confidence=float(cfg["confidence"]),
        seed=seed,
    )
    spend_delta = incremental_spend(panel, test)
    inc = res.lift_abs
    cost_per = abs(spend_delta) / abs(inc) if abs(inc) > 1e-9 else None
    return {
        "incremental_conversions": round(inc, 2),
        "incremental_se": round(float(inf["incremental_se"]), 2),
        "lift_pct": round(res.lift_pct, 4),
        "ci_low": round(float(inf["ci_low"]), 2),
        "ci_high": round(float(inf["ci_high"]), 2),
        "p_value": round(float(inf["p_value"]), 4),
        "placebo_count": int(inf["placebo_count"]),
        "incremental_spend": round(spend_delta, 2),
        "cost_per_incremental": round(cost_per, 2) if cost_per is not None else None,
        "conversions_per_1k_spend": round(abs(inc) / max(abs(spend_delta), 1e-9) * 1000.0, 4),
        "pre_rmspe": round(res.pre_rmspe, 4),
        "post_rmspe": round(res.post_rmspe, 4),
        "weights": {k: round(v, 4) for k, v in res.weights.items() if v > 1e-4},
        "actual_total": round(float(res.actual_post.sum()), 2),
        "counterfactual_total": round(float(res.counterfactual_post.sum()), 2),
        "test_weeks": len(res.post_weeks),
    }
