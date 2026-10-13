"""Model checks on the session-wide fit (test profile, 20 invented markets, 60 weeks)."""

import numpy as np
import pandas as pd

from measurement.mmm.data import CalibrationTarget
from measurement.mmm.fit import counterfactual_spend


def test_sampler_diagnostics_are_sane(fitted_repo, refresh_summary):
    run = fitted_repo.get_run(refresh_summary["run_id"])
    d = run["diagnostics"]
    assert d["r_hat_max"] < 1.25
    assert d["divergences"] <= 5
    assert d["holdout_mape"] < 0.15
    assert d["train_mape"] < 0.10
    assert d["n_markets"] == 20 and d["n_weeks"] == 60


def test_channel_contributions_recover_the_truth(fitted_repo, refresh_summary, small_seed):
    res = fitted_repo.get_channel_results(refresh_summary["run_id"]).set_index("channel")
    truth = small_seed.truth.groupby("channel").incremental.sum()
    # A single short chain on 16 markets gets the order of magnitude right; the full profile on the
    # production-sized panel is what the production figures came from.
    total_truth, total_fit = truth.sum(), res.incremental_mean.sum()
    assert abs(total_fit - total_truth) / total_truth < 0.4
    for ch in res.index:
        inside = res.loc[ch, "incremental_low"] <= truth[ch] <= res.loc[ch, "incremental_high"]
        close = abs(res.loc[ch, "incremental_mean"] - truth[ch]) / truth[ch] < 0.6
        assert inside or close, f"{ch}: truth {truth[ch]:.0f} vs fit {res.loc[ch].to_dict()}"
    assert (res.roi_mean > 0.3).all() and (res.roi_mean < 8).all()
    assert (res.decay.between(0.02, 0.95)).all()


def test_calibration_terms_track_the_geo_reads(fitted_repo, refresh_summary):
    run = fitted_repo.get_run(refresh_summary["run_id"])
    checks = run["diagnostics"]["calibration"]
    assert {c["test_id"] for c in checks} == {"gt-linear-tv-holdout-2026q1", "gt-ctv-spend-up-2026q2"}
    by_id = {c["test_id"]: c for c in checks}
    tv = by_id["gt-linear-tv-holdout-2026q1"]  # going dark is a large, unmistakable effect
    assert tv["implied_mean"] < 0 and tv["target"] < 0
    assert abs(tv["implied_mean"] - tv["target"]) < 2.5 * tv["target_se"] + 0.25 * abs(tv["target"])
    for c in checks:  # every implied lift is finite and carries an interval
        assert np.isfinite(c["implied_mean"]) and c["implied_low"] <= c["implied_mean"] <= c["implied_high"]


def test_market_effects_are_partially_pooled(fitted_repo, refresh_summary):
    me = fitted_repo.get_market_effects(refresh_summary["run_id"])
    assert len(me) == 20 * 5
    spread = me.groupby("channel").beta_market.agg(lambda s: s.std() / s.mean())
    assert (spread < 0.6).all()


def test_counterfactual_spend_fills_in_the_dark_weeks(panel, small_seed):
    tv = next(t for t in small_seed.geo_tests if t["channel"] == "linear_tv")
    target = CalibrationTarget(
        test_id=tv["test_id"],
        channel="linear_tv",
        markets=tv["treatment_markets"],
        week_start=pd.to_datetime(tv["test_start"]).date(),
        week_end=pd.to_datetime(tv["test_end"]).date(),
        incremental_mean=-1.0,
        incremental_se=1.0,
    )
    cf = counterfactual_spend(panel, target)
    c = panel.channels.index("linear_tv")
    weeks = panel.week_mask(target.week_start, target.week_end)
    for m in tv["treatment_markets"]:
        i = panel.market_index(m)
        assert panel.spend[i, weeks, c].sum() == 0
        assert (cf[i, weeks, c] > 0).all()
        assert np.allclose(cf[i, ~weeks, c], panel.spend[i, ~weeks, c])
    others = [i for i, m in enumerate(panel.markets) if m not in tv["treatment_markets"]]
    assert np.allclose(cf[others], panel.spend[others])
