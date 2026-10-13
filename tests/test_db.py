"""Postgres round trips. Skipped unless DATABASE_URL points at a reachable database (CI provides one)."""

from datetime import date

import pandas as pd

from measurement.pipeline.weekly import load_panel


def test_inputs_round_trip(db_repo, small_seed):
    db_repo.load_inputs(small_seed)
    assert len(db_repo.load_markets()) == len(small_seed.markets)
    assert len(db_repo.load_media()) == len(small_seed.media)
    assert len(db_repo.load_outcomes()) == len(small_seed.outcomes)
    assert len(db_repo.load_airings()) == len(small_seed.airings)
    sessions = db_repo.load_minute_sessions()
    assert len(sessions) == len(small_seed.sessions)
    assert float(sessions.sum()) == float(small_seed.sessions.sum())
    panel = load_panel(db_repo)
    assert panel.shape == (20, 60, 5)
    assert panel.y.sum() == small_seed.outcomes.new_subscriptions.sum()


def test_geo_test_json_fields_round_trip(db_repo):
    t = db_repo.get_geo_test("gt-ctv-spend-up-2026q2")
    assert isinstance(t["treatment_markets"], list) and len(t["control_markets"]) >= 3
    assert t["readout"] is None and t["status"] == "completed"
    db_repo.update_geo_test_readout(
        t["test_id"], {"incremental_conversions": 42.0, "incremental_se": 9.0, "incremental_spend": 1234.5}
    )
    t2 = db_repo.get_geo_test(t["test_id"])
    assert t2["status"] == "read" and t2["readout"]["incremental_conversions"] == 42.0
    assert t2["incremental_spend"] == 1234.5
    assert [x["test_id"] for x in db_repo.load_geo_tests(status="designed")] == ["gt-paid-social-holdout-2026q4"]


def test_run_tables_round_trip(db_repo):
    channels = pd.DataFrame(
        [
            {
                "channel": "linear_tv",
                "spend": 100.0,
                "incremental_mean": 10.0,
                "incremental_low": 5.0,
                "incremental_high": 15.0,
                "roi_mean": 18.0,
                "decay": 0.5,
                "half_sat": 1.0,
                "slope": 1.2,
                "beta": 3.0,
                "spend_scale": 20.0,
            },
        ]
    )
    effects = pd.DataFrame([{"market_id": "m001", "channel": "linear_tv", "beta_market": 2.9}])
    recon = pd.DataFrame(
        [
            {
                "channel": "linear_tv",
                "spend": 100.0,
                "mmm_incremental": 10.0,
                "mmm_low": 5.0,
                "mmm_high": 15.0,
                "test_incremental": None,
                "test_low": None,
                "test_high": None,
                "attributed_conversions": 2.0,
                "harmonized_incremental": 10.0,
                "harmonized_source": "mmm",
                "harmonized_roi": 18.0,
                "mmm_vs_test_gap": None,
            }
        ]
    )
    tv = pd.DataFrame(
        [
            {
                "dimension": "network",
                "key": "Meridian Network",
                "airings": 3,
                "impressions": 1000,
                "cost": 30.0,
                "attributed_sessions": 4.0,
                "cost_per_response": 7.5,
            }
        ]
    )
    db_repo.save_run(
        "run-test-1",
        date(2026, 9, 28),
        "test",
        {"r_hat_max": 1.01, "n_markets": 16, "n_weeks": 60},
        channels,
        effects,
        recon,
        tv,
    )
    db_repo.save_run(
        "run-test-2",
        date(2026, 9, 28),
        "test",
        {"r_hat_max": 1.02, "n_markets": 16, "n_weeks": 60},
        channels,
        effects,
        recon,
        tv,
    )

    runs = db_repo.list_runs()
    assert [r["run_id"] for r in runs][:2] == ["run-test-2", "run-test-1"]
    assert db_repo.latest_run_id() == "run-test-2"
    assert runs[0]["diagnostics"]["r_hat_max"] == 1.02
    got = db_repo.get_channel_results("run-test-1")
    assert float(got.spend_scale.iloc[0]) == 20.0
    r = db_repo.get_reconciliation("run-test-1")
    assert r.harmonized_source.iloc[0] == "mmm" and pd.isna(r.test_incremental.iloc[0])
    assert db_repo.get_tv_rollup("run-test-1", "network").key.iloc[0] == "Meridian Network"
    assert db_repo.get_tv_rollup("run-test-1", "creative").empty
    assert len(db_repo.get_market_effects("run-test-2")) == 1
