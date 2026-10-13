"""Weekly refresh: read any completed geo tests, fit the calibrated MMM, attribute TV spots, reconcile, publish.

In production this ran as an Airflow DAG every Monday morning in about 45 minutes. Here it is a single
function so it can be run from the CLI, from Compose or from a test.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta

import pandas as pd

from measurement.config import channel_names, control_names, value_per_conversion
from measurement.geo.readout import read_geo_test
from measurement.harmonize.reconcile import reconcile
from measurement.mmm.data import CalibrationTarget, Panel
from measurement.mmm.fit import fit_mmm
from measurement.tv.spot_lift import attribute_spots, rollup

TV_ROLLUP_COLUMNS = ["dimension", "key", "airings", "impressions", "cost", "attributed_sessions", "cost_per_response"]


def load_panel(repo) -> Panel:
    return Panel.from_frames(
        repo.load_markets(), repo.load_media(), repo.load_outcomes(), channel_names(), control_names()
    )


def read_completed_tests(repo, panel: Panel, as_of: date, seed: int = 0) -> list[str]:
    """Produce readouts for tests that have finished but have not been read yet."""
    read_ids = []
    for t in repo.load_geo_tests():
        end = pd.to_datetime(t["test_end"]).date()
        if t.get("readout") or end > as_of or t["status"] == "designed":
            continue
        readout = read_geo_test(panel, t, seed=seed)
        repo.update_geo_test_readout(t["test_id"], readout, status="read")
        read_ids.append(t["test_id"])
    return read_ids


def calibration_targets(repo, as_of: date) -> tuple[list[CalibrationTarget], list[dict]]:
    tests = [t for t in repo.load_geo_tests() if t.get("readout") and pd.to_datetime(t["test_end"]).date() <= as_of]
    return [CalibrationTarget.from_readout(t) for t in tests], tests


def tv_response(repo, as_of: date, weeks: int = 4) -> tuple[pd.DataFrame, int]:
    end = datetime.combine(as_of + timedelta(days=7), datetime.min.time())
    start = end - timedelta(weeks=weeks)
    airings = repo.load_airings(start, end)
    sessions = repo.load_minute_sessions(start, end)
    if len(airings) == 0 or len(sessions) == 0:
        return pd.DataFrame(columns=TV_ROLLUP_COLUMNS), 0
    attributed = attribute_spots(sessions, airings)
    return rollup(attributed), len(airings)


def run_refresh(repo, as_of: date | None = None, profile: str = "quick", seed: int = 0, tv_weeks: int = 4) -> dict:
    panel = load_panel(repo)
    as_of = as_of or max(panel.weeks)
    value = value_per_conversion()

    read_ids = read_completed_tests(repo, panel, as_of, seed=seed)
    targets, tests = calibration_targets(repo, as_of)
    result = fit_mmm(panel, targets, profile=profile, seed=seed, value_per_conversion=value)
    tv, n_airings = tv_response(repo, as_of, weeks=tv_weeks)

    attribution = repo.load_attribution()
    if len(attribution):
        attribution = attribution[pd.to_datetime(attribution.week_start).dt.date.isin(set(panel.weeks))]
    recon = reconcile(
        result.channel_summary,
        tests,
        attribution,
        value,
        calibration_checks=result.diagnostics.get("calibration"),
        as_of=as_of,
    )

    run_id = f"run-{as_of:%Y%m%d}-{datetime.now():%H%M%S}"
    diagnostics = dict(result.diagnostics)
    diagnostics["tv_airings_attributed"] = n_airings
    diagnostics["tests_read_this_run"] = read_ids
    diagnostics["calibration_tests"] = [t["test_id"] for t in tests]
    repo.save_run(run_id, as_of, profile, diagnostics, result.channel_summary, result.market_effects, recon, tv)
    return {
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "profile": profile,
        "channels": len(result.channels),
        "markets": len(result.markets),
        "weeks": len(result.weeks),
        "tests_read": read_ids,
        "calibration_tests": [t["test_id"] for t in tests],
        "tv_airings": n_airings,
        "diagnostics": {k: diagnostics[k] for k in ("divergences", "r_hat_max", "train_mape", "holdout_mape")},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the weekly measurement refresh")
    parser.add_argument("--as-of", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--profile", default="full", choices=["full", "quick", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)

    from measurement.repository import get_repository

    repo = get_repository(args.database_url)
    summary = run_refresh(repo, as_of=args.as_of, profile=args.profile, seed=args.seed)
    json.dump(summary, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
