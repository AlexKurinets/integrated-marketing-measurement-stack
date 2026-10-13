from datetime import date

import pandas as pd

from measurement.harmonize.reconcile import explain_channel, reconcile

AS_OF = date(2026, 9, 28)


def _summary():
    return pd.DataFrame(
        [
            {
                "channel": "linear_tv",
                "spend": 1_000_000.0,
                "incremental_mean": 9_000.0,
                "incremental_low": 6_000.0,
                "incremental_high": 12_000.0,
            },
            {
                "channel": "ctv",
                "spend": 500_000.0,
                "incremental_mean": 6_000.0,
                "incremental_low": 4_000.0,
                "incremental_high": 8_000.0,
            },
            {
                "channel": "paid_search",
                "spend": 200_000.0,
                "incremental_mean": 4_000.0,
                "incremental_low": 3_000.0,
                "incremental_high": 5_000.0,
            },
        ]
    )


def _test(test_id, channel, end, inc, se, dspend):
    return {
        "test_id": test_id,
        "channel": channel,
        "test_end": end,
        "status": "read",
        "readout": {"incremental_conversions": inc, "incremental_se": se, "incremental_spend": dspend},
    }


def _attr():
    return pd.DataFrame(
        {"channel": ["linear_tv", "ctv", "paid_search"], "last_touch_conversions": [1_200, 4_000, 11_000]}
    )


def test_channel_without_test_uses_mmm():
    out = reconcile(_summary(), [], _attr(), 180.0, as_of=AS_OF).set_index("channel")
    assert out.loc["paid_search", "harmonized_source"] == "mmm"
    assert out.loc["paid_search", "harmonized_incremental"] == 4_000.0
    assert abs(out.loc["paid_search", "harmonized_roi"] - 4_000 * 180 / 200_000) < 1e-9
    assert pd.isna(out.loc["paid_search", "test_incremental"])


def test_calibrated_mmm_within_tolerance_keeps_mmm_figure():
    tests = [_test("gt-tv", "linear_tv", "2026-08-03", -300.0, 40.0, -50_000.0)]
    checks = [{"test_id": "gt-tv", "target": -300.0, "target_se": 50.0, "implied_mean": -330.0}]
    out = reconcile(_summary(), tests, _attr(), 180.0, calibration_checks=checks, as_of=AS_OF).set_index("channel")
    row = out.loc["linear_tv"]
    assert row.harmonized_source == "mmm_calibrated:gt-tv"
    assert row.harmonized_incremental == 9_000.0
    assert abs(row.mmm_vs_test_gap - 0.10) < 1e-9
    assert abs(row.test_incremental - 9_000.0 * 300 / 330) < 1e-6
    assert row.test_low < row.test_incremental < row.test_high


def test_outside_tolerance_scales_total_to_the_test():
    tests = [_test("gt-ctv", "ctv", "2026-07-06", 400.0, 60.0, 80_000.0)]
    checks = [{"test_id": "gt-ctv", "target": 400.0, "target_se": 75.0, "implied_mean": 800.0}]
    out = reconcile(_summary(), tests, _attr(), 180.0, calibration_checks=checks, as_of=AS_OF).set_index("channel")
    row = out.loc["ctv"]
    assert row.harmonized_source == "geo_test:gt-ctv"
    assert abs(row.harmonized_incremental - 3_000.0) < 1e-6
    assert abs(row.mmm_vs_test_gap - 1.0) < 1e-9


def test_stale_and_future_tests_are_ignored():
    tests = [
        _test("gt-old", "ctv", "2025-01-06", 400.0, 60.0, 80_000.0),
        _test("gt-future", "ctv", "2026-12-07", 400.0, 60.0, 80_000.0),
    ]
    out = reconcile(_summary(), tests, _attr(), 180.0, as_of=AS_OF).set_index("channel")
    assert out.loc["ctv", "harmonized_source"] == "mmm"


def test_rate_fallback_when_no_calibration_record():
    # MMM rate 6000/500k = 12 per 1k; test rate 400/80k = 5 per 1k -> gap +140%, test wins.
    tests = [_test("gt-ctv", "ctv", "2026-07-06", 400.0, 60.0, 80_000.0)]
    out = reconcile(_summary(), tests, _attr(), 180.0, as_of=AS_OF).set_index("channel")
    row = out.loc["ctv"]
    assert row.harmonized_source == "geo_test:gt-ctv"
    assert abs(row.test_incremental - 2_500.0) < 1e-6
    assert row.mmm_vs_test_gap > 1.0


def test_explanations_read_like_a_readout():
    tests = [_test("gt-tv", "linear_tv", "2026-08-03", -300.0, 40.0, -50_000.0)]
    checks = [{"test_id": "gt-tv", "target": -300.0, "target_se": 50.0, "implied_mean": -330.0}]
    out = reconcile(_summary(), tests, _attr(), 180.0, calibration_checks=checks, as_of=AS_OF)
    labels = {"linear_tv": "Linear TV", "paid_search": "Paid search"}
    tv = explain_channel(out.iloc[0], labels)
    assert tv.startswith("Linear TV:") and "gt-tv" in tv and "within tolerance" in tv and "under-count" in tv
    search = explain_channel(out.iloc[2], labels)
    assert "No recent geo test" in search and "over-count" in search
