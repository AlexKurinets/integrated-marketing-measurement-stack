from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from measurement.tv.spot_lift import attribute_spots, rollup, weekly_response

CFG = {"baseline_window_minutes": 10, "response_window_minutes": 5, "contamination_minutes": 8}
T0 = datetime(2026, 9, 14, 20, 0)


def _sessions(n: int = 120, level: float = 20.0) -> pd.Series:
    idx = pd.date_range(T0, periods=n, freq="min")
    return pd.Series(np.full(n, level), index=idx)


def _airing(i: int, minute: int, impressions: int, **kw) -> dict:
    return {
        "airing_id": f"a{i}",
        "aired_at": T0 + timedelta(minutes=minute),
        "network": kw.get("network", "Meridian Network"),
        "daypart": kw.get("daypart", "prime"),
        "creative": kw.get("creative", "Weeknight Dinner :30"),
        "impressions": impressions,
        "cost": kw.get("cost", 500.0),
    }


def test_single_airing_baseline_and_lift():
    s = _sessions()
    s.iloc[30:35] += [10, 8, 5, 2, 1]  # 26 extra sessions after an airing at minute 30
    out = attribute_spots(s, pd.DataFrame([_airing(1, 30, 100_000)]), CFG)
    assert out.baseline.iloc[0] == 20.0
    assert np.isclose(out.attributed_sessions.iloc[0], 26.0)
    assert np.isclose(out.cost_per_response.iloc[0], 500.0 / 26.0)


def test_overlapping_airings_split_lift_by_impressions():
    s = _sessions()
    s.iloc[40:45] += 30  # one flat burst covering both airings at minute 40
    air = pd.DataFrame([_airing(1, 40, 300_000), _airing(2, 40, 100_000)])
    out = attribute_spots(s, air, CFG).set_index("airing_id")
    assert np.isclose(out.loc["a1", "attributed_sessions"] + out.loc["a2", "attributed_sessions"], 150.0)
    assert np.isclose(out.loc["a1", "attributed_sessions"] / out.loc["a2", "attributed_sessions"], 3.0)


def test_baseline_ignores_contaminated_minutes():
    s = _sessions()
    s.iloc[50:58] += 40  # first airing at 50 lifts minutes 50..57
    air = pd.DataFrame([_airing(1, 50, 100_000), _airing(2, 58, 100_000)])
    out = attribute_spots(s, air, CFG).set_index("airing_id")
    # The second airing's 10-minute baseline window (48..57) overlaps the first airing's response; only
    # minutes 48 and 49 are clean, which is fewer than three, so the median of the whole window is used.
    assert out.loc["a2", "baseline"] == 60.0
    s2 = _sessions()
    s2.iloc[50:55] += 40
    air2 = pd.DataFrame([_airing(1, 50, 100_000), _airing(2, 70, 100_000)])
    out2 = attribute_spots(s2, air2, CFG).set_index("airing_id")
    assert out2.loc["a2", "baseline"] == 20.0


def test_negative_deviations_do_not_create_negative_lift():
    s = _sessions()
    s.iloc[20:25] -= 5
    out = attribute_spots(s, pd.DataFrame([_airing(1, 20, 50_000)]), CFG)
    assert out.attributed_sessions.iloc[0] == 0.0
    assert np.isnan(out.cost_per_response.iloc[0])


def test_rollup_and_weekly_response():
    s = _sessions(300)
    s.iloc[30:35] += 10
    s.iloc[100:105] += 20
    air = pd.DataFrame(
        [
            _airing(1, 30, 100_000, network="Harbor Sports", cost=400.0),
            _airing(2, 100, 100_000, network="Lantern TV", cost=600.0),
            _airing(3, 200, 100_000, network="Lantern TV", cost=300.0),
        ]
    )
    attributed = attribute_spots(s, air, CFG)
    r = rollup(attributed)
    nets = r[r.dimension == "network"].set_index("key")
    assert nets.loc["Lantern TV", "airings"] == 2
    assert np.isclose(nets.loc["Lantern TV", "attributed_sessions"], 100.0)
    assert np.isclose(nets.loc["Lantern TV", "cost_per_response"], 900.0 / 100.0)
    assert np.isclose(r[r.dimension == "daypart"].attributed_sessions.sum(), 150.0)
    w = weekly_response(attributed)
    assert len(w) == 1 and np.isclose(w.attributed_sessions.iloc[0], 150.0)


def test_seeded_airings_recover_injected_response_in_aggregate(small_seed):
    attributed = attribute_spots(small_seed.sessions, small_seed.airings)
    injected = (small_seed.airings.impressions / 1000.0).sum()  # average creative factor is about one
    ratio = attributed.attributed_sessions.sum() / injected
    assert 0.6 < ratio < 1.5
    # Per-airing reads are noisy; the rollups are what the readout uses and they should carry the signal.
    by_dp = attributed.groupby("daypart").agg(imps=("impressions", "sum"), lift=("attributed_sessions", "sum"))
    assert np.corrcoef(by_dp.imps, by_dp.lift)[0, 1] > 0.5
    by_creative = attributed.groupby("creative").agg(imps=("impressions", "sum"), lift=("attributed_sessions", "sum"))
    per_1k = (by_creative.lift / by_creative.imps * 1000).sort_values(ascending=False)
    assert per_1k.index[0] == "First Box Free :30"  # the creative seeded with the strongest response
