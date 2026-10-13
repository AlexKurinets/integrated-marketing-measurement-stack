import pytest

from measurement.harmonize.allocate import expected_weekly_conversions, propose_reallocation, simulate_shift

PARAMS = {
    "linear_tv": {"beta": 40.0, "half_sat": 1.0, "slope": 1.3, "spend_scale": 1_000.0, "n_markets": 20},
    "ctv": {"beta": 30.0, "half_sat": 0.9, "slope": 1.2, "spend_scale": 600.0, "n_markets": 20},
    "paid_search": {"beta": 25.0, "half_sat": 0.7, "slope": 1.0, "spend_scale": 300.0, "n_markets": 20},
}
# TV is spent at three times its half-saturation point, CTV well below it.
SPEND = {"linear_tv": 60_000.0, "ctv": 6_000.0, "paid_search": 6_000.0}


def test_response_curve_has_diminishing_returns():
    p = PARAMS["linear_tv"]
    y1, y2, y3 = (expected_weekly_conversions(p, s) for s in (10_000.0, 20_000.0, 40_000.0))
    assert 0 < y1 < y2 < y3
    assert (y2 - y1) > (y3 - y2)


def test_shift_from_saturated_to_unsaturated_channel_helps():
    out = simulate_shift(PARAMS, SPEND, "linear_tv", "ctv", 10_000.0)
    assert out["delta_conversions"] > 0
    assert abs(sum(out["weekly_spend_after"].values()) - sum(SPEND.values())) < 1e-6
    assert out["weekly_spend_after"]["ctv"] == 16_000.0


def test_shift_validation():
    with pytest.raises(ValueError):
        simulate_shift(PARAMS, SPEND, "ctv", "linear_tv", 50_000.0)
    with pytest.raises(KeyError):
        simulate_shift(PARAMS, SPEND, "radio", "ctv", 100.0)


def test_reallocation_respects_bounds_and_improves_total():
    out = propose_reallocation(PARAMS, SPEND, max_shift_pct=0.2)
    assert out["total_after"] >= out["total_before"]
    assert out["moves"] > 0
    for row in out["plan"].values():
        assert -0.2 - 1e-9 <= row["change_pct"] <= 0.2 + 1e-9
    assert abs(sum(r["proposed"] for r in out["plan"].values()) - sum(SPEND.values())) < 1e-6
    assert out["plan"]["linear_tv"]["change_pct"] < 0 < out["plan"]["ctv"]["change_pct"]
