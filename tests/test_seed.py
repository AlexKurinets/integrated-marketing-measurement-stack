import numpy as np
import pandas as pd

from measurement.config import channel_names
from seed.generate import BUDGET_SHARE, generate, weekly_media_budget
from seed.markets import MARKETS

REAL_PLACES = {"New York", "Los Angeles", "Chicago", "Dallas", "Houston", "Atlanta", "Boston", "Seattle", "Denver"}


def test_markets_are_invented(small_seed):
    names = set(small_seed.markets.name)
    assert names.isdisjoint(REAL_PLACES)
    assert names <= {n for n, _ in MARKETS}
    assert small_seed.markets.market_id.is_unique


def test_panel_is_complete(small_seed):
    M, T, C = len(small_seed.markets), small_seed.outcomes.week_start.nunique(), len(channel_names())
    assert len(small_seed.media) == M * T * C
    assert len(small_seed.outcomes) == M * T
    assert (small_seed.outcomes.new_subscriptions >= 0).all()
    assert (small_seed.media.spend >= 0).all()


def test_geo_tests_are_embedded_in_spend(small_seed):
    media = small_seed.media
    tv = next(t for t in small_seed.geo_tests if t["channel"] == "linear_tv")
    weeks = (media.week_start >= tv["test_start"]) & (media.week_start <= tv["test_end"])
    dark = media[weeks & (media.channel == "linear_tv") & media.market_id.isin(tv["treatment_markets"])]
    assert len(dark) and (dark.spend == 0).all()

    ctv = next(t for t in small_seed.geo_tests if t["channel"] == "ctv")
    weeks = (media.week_start >= ctv["test_start"]) & (media.week_start <= ctv["test_end"])
    sub = media[weeks & (media.channel == "ctv")]
    pop = small_seed.markets.set_index("market_id").population
    per_cap = sub.assign(pc=sub.spend / sub.market_id.map(pop))
    treated = per_cap[per_cap.market_id.isin(ctv["treatment_markets"])].pc.mean()
    others = per_cap[~per_cap.market_id.isin(ctv["treatment_markets"])].pc.mean()
    assert treated > 1.5 * others

    designed = next(t for t in small_seed.geo_tests if t["status"] == "designed")
    assert designed["test_start"] > small_seed.outcomes.week_start.max()
    for t in small_seed.geo_tests:
        assert len(t["control_markets"]) >= 3
        assert set(t["control_markets"]).isdisjoint(t["treatment_markets"])


def test_airing_costs_match_weekly_tv_budget(small_seed):
    expected = weekly_media_budget(small_seed.markets) * BUDGET_SHARE["linear_tv"] * 3
    assert abs(small_seed.airings.cost.sum() - expected) / expected < 0.01
    assert small_seed.airings.airing_id.is_unique
    assert len(small_seed.sessions) == 3 * 7 * 24 * 60


def test_last_touch_attribution_has_the_usual_bias(small_seed):
    truth = small_seed.truth.groupby("channel").incremental.sum()
    attr = small_seed.attribution.groupby("channel").last_touch_conversions.sum()
    assert attr["paid_search"] > 1.5 * truth["paid_search"]
    assert attr["linear_tv"] < 0.5 * truth["linear_tv"]


def test_generator_is_deterministic():
    a = generate(n_markets=6, n_weeks=20, seed=3, tv_weeks=1)
    b = generate(n_markets=6, n_weeks=20, seed=3, tv_weeks=1)
    pd.testing.assert_frame_equal(a.outcomes, b.outcomes)
    pd.testing.assert_frame_equal(a.airings, b.airings)
    assert np.allclose(a.sessions.to_numpy(), b.sessions.to_numpy())
