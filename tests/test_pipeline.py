def test_refresh_publishes_a_complete_run(fitted_repo, refresh_summary):
    run_id = refresh_summary["run_id"]
    assert refresh_summary["channels"] == 5 and refresh_summary["markets"] == 20
    assert set(refresh_summary["tests_read"]) == {"gt-linear-tv-holdout-2026q1", "gt-ctv-spend-up-2026q2"}
    assert refresh_summary["tv_airings"] > 0

    run = fitted_repo.get_run(run_id)
    assert run["profile"] == "test"
    assert fitted_repo.latest_run_id() == run_id
    channels = fitted_repo.get_channel_results(run_id)
    recon = fitted_repo.get_reconciliation(run_id)
    tv = fitted_repo.get_tv_rollup(run_id)
    assert len(channels) == 5 and len(recon) == 5
    assert set(tv.dimension) == {"network", "daypart", "creative"}
    assert (recon.harmonized_incremental > 0).all()
    sources = recon.set_index("channel").harmonized_source
    assert sources["linear_tv"].endswith("gt-linear-tv-holdout-2026q1")
    assert sources["ctv"].endswith("gt-ctv-spend-up-2026q2")
    assert sources["paid_search"] == "mmm"


def test_geo_tests_are_read_once_and_designed_tests_left_alone(fitted_repo):
    tests = {t["test_id"]: t for t in fitted_repo.load_geo_tests()}
    assert tests["gt-linear-tv-holdout-2026q1"]["status"] == "read"
    assert tests["gt-linear-tv-holdout-2026q1"]["readout"]["incremental_conversions"] < 0
    assert tests["gt-linear-tv-holdout-2026q1"]["readout"]["p_value"] <= 0.1  # going dark is unmistakable
    assert tests["gt-ctv-spend-up-2026q2"]["readout"]["placebo_count"] >= 20
    assert tests["gt-ctv-spend-up-2026q2"]["readout"]["incremental_spend"] > 0
    assert tests["gt-paid-social-holdout-2026q4"]["status"] == "designed"
    assert tests["gt-paid-social-holdout-2026q4"]["readout"] is None


def test_second_refresh_reuses_readouts(small_seed):
    """A fresh repo with the readouts already stored should not re-read the tests."""
    from measurement.repository import InMemoryRepository

    repo = InMemoryRepository()
    repo.load_inputs(small_seed)
    repo.update_geo_test_readout(
        "gt-linear-tv-holdout-2026q1",
        {"incremental_conversions": -80.0, "incremental_se": 30.0, "incremental_spend": -20_000.0},
    )
    repo.update_geo_test_readout(
        "gt-ctv-spend-up-2026q2",
        {"incremental_conversions": 60.0, "incremental_se": 25.0, "incremental_spend": 15_000.0},
    )
    from measurement.pipeline.weekly import calibration_targets, load_panel, read_completed_tests

    panel = load_panel(repo)
    assert read_completed_tests(repo, panel, max(panel.weeks)) == []
    targets, tests = calibration_targets(repo, max(panel.weeks))
    assert [t.test_id for t in targets] == ["gt-linear-tv-holdout-2026q1", "gt-ctv-spend-up-2026q2"]
    assert targets[0].incremental_mean == -80.0 and targets[1].incremental_se == 25.0
