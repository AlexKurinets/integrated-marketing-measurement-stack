import pytest
from fastapi.testclient import TestClient

from measurement.api.app import create_app
from measurement.service import MeasurementService


@pytest.fixture(scope="module")
def client(fitted_repo):
    return TestClient(create_app(MeasurementService(fitted_repo, value=180.0)))


def test_health_and_runs(client, refresh_summary):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["latest_run"] == refresh_summary["run_id"]
    runs = client.get("/runs").json()
    assert runs[0]["run_id"] == refresh_summary["run_id"]
    assert "r_hat_max" in runs[0]["diagnostics"]
    assert client.get("/runs/latest").json()["run_id"] == refresh_summary["run_id"]
    assert client.get("/runs/nope").status_code == 404


def test_channel_results_and_reconciliation(client, refresh_summary):
    rows = client.get(f"/runs/{refresh_summary['run_id']}/channels").json()
    assert len(rows) == 5 and rows[0]["label"]
    assert all(r["incremental_low"] <= r["incremental_mean"] <= r["incremental_high"] for r in rows)
    recon = client.get("/runs/latest/reconciliation").json()
    assert {r["channel"] for r in recon} == {r["channel"] for r in rows}
    assert all(r["harmonized_source"] for r in recon)
    explain = client.get("/runs/latest/explain", params={"channel": "linear_tv"}).json()
    assert len(explain) == 1 and explain[0]["text"].startswith("Linear TV:")
    assert client.get("/runs/latest/explain", params={"channel": "radio"}).status_code == 404


def test_geo_tests(client):
    tests = client.get("/geo-tests").json()
    assert {t["test_id"] for t in tests} >= {"gt-linear-tv-holdout-2026q1", "gt-ctv-spend-up-2026q2"}
    one = client.get("/geo-tests/gt-ctv-spend-up-2026q2").json()
    assert one["readout"]["incremental_conversions"] > 0
    assert client.get("/geo-tests", params={"status": "designed"}).json()[0]["status"] == "designed"
    assert client.get("/geo-tests/missing").status_code == 404


def test_tv_and_markets(client):
    nets = client.get("/runs/latest/tv", params={"dimension": "network"}).json()
    assert nets and all(r["dimension"] == "network" for r in nets)
    assert sum(r["airings"] for r in nets) > 0
    markets = client.get("/runs/latest/markets", params={"channel": "ctv"}).json()
    assert len(markets) == 20


def test_scenarios(client):
    sim = client.get(
        "/runs/latest/simulate", params={"from_channel": "linear_tv", "to_channel": "ctv", "weekly_amount": 500}
    ).json()
    assert "delta_conversions" in sim and sim["weekly_spend_after"]["ctv"] > sim["weekly_spend_before"]["ctv"]
    bad = client.get(
        "/runs/latest/simulate", params={"from_channel": "ctv", "to_channel": "linear_tv", "weekly_amount": 1e12}
    )
    assert bad.status_code == 400
    plan = client.get("/runs/latest/reallocation", params={"max_shift_pct": 0.1}).json()
    assert set(plan["plan"]) == {"linear_tv", "ctv", "online_video", "paid_social", "paid_search"}
    assert plan["total_after"] >= plan["total_before"] - 1e-6
