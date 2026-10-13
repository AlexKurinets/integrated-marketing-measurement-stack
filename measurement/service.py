"""Read model for published results. The API and the MCP server both call this and nothing else."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from measurement import __version__
from measurement.config import channel_labels, value_per_conversion
from measurement.harmonize.allocate import propose_reallocation, simulate_shift
from measurement.harmonize.reconcile import explain_channel


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for r in df.to_dict("records"):
        clean = {}
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        out.append(clean)
    return out


class MeasurementService:
    def __init__(self, repo, value: float | None = None) -> None:
        self.repo = repo
        self.value = value if value is not None else value_per_conversion()

    # ---- helpers ------------------------------------------------------------------------------------
    def _resolve_run(self, run_id: str | None) -> str:
        if run_id:
            if self.repo.get_run(run_id) is None:
                raise KeyError(f"Unknown run {run_id}")
            return run_id
        latest = self.repo.latest_run_id()
        if latest is None:
            raise KeyError("No model runs published yet")
        return latest

    def _response_inputs(self, run_id: str) -> tuple[dict[str, dict], dict[str, float]]:
        run = self.repo.get_run(run_id)
        diag = run["diagnostics"]
        n_markets, n_weeks = int(diag["n_markets"]), int(diag["n_weeks"])
        params, weekly = {}, {}
        for r in self.repo.get_channel_results(run_id).to_dict("records"):
            params[r["channel"]] = {
                "beta": float(r["beta"]),
                "half_sat": float(r["half_sat"]),
                "slope": float(r["slope"]),
                "spend_scale": float(r["spend_scale"]),
                "n_markets": n_markets,
            }
            weekly[r["channel"]] = float(r["spend"]) / n_weeks
        return params, weekly

    # ---- reads --------------------------------------------------------------------------------------
    def health(self) -> dict:
        return {"status": "ok", "version": __version__, "latest_run": self.repo.latest_run_id()}

    def runs(self) -> list[dict]:
        return self.repo.list_runs()

    def run(self, run_id: str | None = None) -> dict:
        return self.repo.get_run(self._resolve_run(run_id))

    def channel_results(self, run_id: str | None = None) -> list[dict]:
        rid = self._resolve_run(run_id)
        df = self.repo.get_channel_results(rid)
        labels = channel_labels()
        df.insert(1, "label", df.channel.map(lambda c: labels.get(c, c)))
        return _records(df)

    def reconciliation(self, run_id: str | None = None) -> list[dict]:
        rid = self._resolve_run(run_id)
        df = self.repo.get_reconciliation(rid)
        labels = channel_labels()
        df.insert(1, "label", df.channel.map(lambda c: labels.get(c, c)))
        return _records(df)

    def explain(self, run_id: str | None = None, channel: str | None = None) -> list[dict]:
        rows = self.reconciliation(run_id)
        if channel:
            rows = [r for r in rows if r["channel"] == channel]
            if not rows:
                raise KeyError(f"Unknown channel {channel}")
        return [{"channel": r["channel"], "text": explain_channel(r, channel_labels(), self.value)} for r in rows]

    def geo_tests(self, status: str | None = None) -> list[dict]:
        return self.repo.load_geo_tests(status)

    def geo_test(self, test_id: str) -> dict:
        t = self.repo.get_geo_test(test_id)
        if t is None:
            raise KeyError(f"Unknown geo test {test_id}")
        return t

    def tv_rollup(self, run_id: str | None = None, dimension: str | None = None) -> list[dict]:
        rid = self._resolve_run(run_id)
        return _records(self.repo.get_tv_rollup(rid, dimension))

    def market_effects(self, run_id: str | None = None, channel: str | None = None) -> list[dict]:
        rid = self._resolve_run(run_id)
        df = self.repo.get_market_effects(rid)
        if channel:
            df = df[df.channel == channel]
        return _records(df)

    # ---- scenarios ----------------------------------------------------------------------------------
    def simulate_shift(
        self, from_channel: str, to_channel: str, weekly_amount: float, run_id: str | None = None
    ) -> dict:
        rid = self._resolve_run(run_id)
        params, weekly = self._response_inputs(rid)
        out = simulate_shift(params, weekly, from_channel, to_channel, float(weekly_amount))
        out["run_id"] = rid
        return out

    def propose_reallocation(self, max_shift_pct: float = 0.2, run_id: str | None = None) -> dict:
        rid = self._resolve_run(run_id)
        params, weekly = self._response_inputs(rid)
        out = propose_reallocation(params, weekly, max_shift_pct=float(max_shift_pct))
        out["run_id"] = rid
        return out
