"""Results API: a thin HTTP layer over MeasurementService for dashboards and the client success team."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query

from measurement.service import MeasurementService


def create_app(service: MeasurementService | None = None) -> FastAPI:
    app = FastAPI(title="Integrated Marketing Measurement API", version="0.4.0")

    def svc() -> MeasurementService:
        nonlocal service
        if service is None:
            from measurement.repository import get_repository

            service = MeasurementService(get_repository())
        return service

    def guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def health():
        return svc().health()

    @app.get("/runs")
    def runs():
        return svc().runs()

    @app.get("/runs/latest")
    def latest_run():
        return guard(svc().run, None)

    @app.get("/runs/{run_id}")
    def run(run_id: str):
        return guard(svc().run, run_id)

    @app.get("/runs/{run_id}/channels")
    def channels(run_id: str):
        return guard(svc().channel_results, None if run_id == "latest" else run_id)

    @app.get("/runs/{run_id}/reconciliation")
    def reconciliation(run_id: str):
        return guard(svc().reconciliation, None if run_id == "latest" else run_id)

    @app.get("/runs/{run_id}/explain")
    def explain(run_id: str, channel: str | None = None):
        return guard(svc().explain, None if run_id == "latest" else run_id, channel)

    @app.get("/runs/{run_id}/tv")
    def tv(run_id: str, dimension: str | None = Query(default=None)):
        return guard(svc().tv_rollup, None if run_id == "latest" else run_id, dimension)

    @app.get("/runs/{run_id}/markets")
    def markets(run_id: str, channel: str | None = None):
        return guard(svc().market_effects, None if run_id == "latest" else run_id, channel)

    @app.get("/runs/{run_id}/simulate")
    def simulate(run_id: str, from_channel: str, to_channel: str, weekly_amount: float):
        rid = None if run_id == "latest" else run_id
        return guard(svc().simulate_shift, from_channel, to_channel, weekly_amount, rid)

    @app.get("/runs/{run_id}/reallocation")
    def reallocation(run_id: str, max_shift_pct: float = 0.2):
        return guard(svc().propose_reallocation, max_shift_pct, None if run_id == "latest" else run_id)

    @app.get("/geo-tests")
    def geo_tests(status: str | None = None):
        return svc().geo_tests(status)

    @app.get("/geo-tests/{test_id}")
    def geo_test(test_id: str):
        return guard(svc().geo_test, test_id)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("measurement.api.app:app", host="0.0.0.0", port=int(os.environ.get("API_PORT", "8000")))


if __name__ == "__main__":
    main()
