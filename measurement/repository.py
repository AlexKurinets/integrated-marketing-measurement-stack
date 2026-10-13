"""Storage for inputs and results.

PostgresRepository is what the pipeline, API and MCP server use. InMemoryRepository has the same interface
and backs the unit tests that do not need a database.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from measurement.config import database_url

SQL_DIR = Path(os.environ.get("IMMS_SQL_DIR", Path(__file__).resolve().parent.parent / "sql"))


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, date | datetime):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


class InMemoryRepository:
    def __init__(self) -> None:
        self.markets = pd.DataFrame(columns=["market_id", "name", "region", "population", "tv_households"])
        self.media = pd.DataFrame(columns=["market_id", "week_start", "channel", "spend", "impressions"])
        self.outcomes = pd.DataFrame()
        self.airings = pd.DataFrame()
        self.sessions = pd.Series(dtype=float)
        self.attribution = pd.DataFrame(columns=["week_start", "channel", "last_touch_conversions"])
        self.geo_tests: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}

    # ---- inputs -------------------------------------------------------------------------------------
    def migrate(self) -> None:
        return None

    def load_inputs(self, data) -> None:
        self.markets = data.markets
        self.media = data.media
        self.outcomes = data.outcomes
        self.airings = data.airings
        self.sessions = data.sessions
        self.attribution = data.attribution
        for t in data.geo_tests:
            self.save_geo_test(t)

    def load_markets(self) -> pd.DataFrame:
        return self.markets.copy()

    def load_media(self) -> pd.DataFrame:
        return self.media.copy()

    def load_outcomes(self) -> pd.DataFrame:
        return self.outcomes.copy()

    def load_airings(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        a = self.airings.copy()
        if len(a) and start is not None:
            a = a[pd.to_datetime(a.aired_at) >= start]
        if len(a) and end is not None:
            a = a[pd.to_datetime(a.aired_at) < end]
        return a

    def load_minute_sessions(self, start: datetime | None = None, end: datetime | None = None) -> pd.Series:
        s = self.sessions.copy()
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index < end]
        return s

    def load_attribution(self) -> pd.DataFrame:
        return self.attribution.copy()

    # ---- geo tests ----------------------------------------------------------------------------------
    def save_geo_test(self, test: dict) -> None:
        self.geo_tests[test["test_id"]] = _json_ready(dict(test))

    def update_geo_test_readout(self, test_id: str, readout: dict, status: str = "read") -> None:
        self.geo_tests[test_id]["readout"] = _json_ready(readout)
        self.geo_tests[test_id]["status"] = status
        if readout.get("incremental_spend") is not None:
            self.geo_tests[test_id]["incremental_spend"] = readout["incremental_spend"]

    def load_geo_tests(self, status: str | None = None) -> list[dict]:
        tests = [dict(t) for t in self.geo_tests.values()]
        if status:
            tests = [t for t in tests if t["status"] == status]
        return sorted(tests, key=lambda t: t["test_start"])

    def get_geo_test(self, test_id: str) -> dict | None:
        t = self.geo_tests.get(test_id)
        return dict(t) if t else None

    # ---- runs ---------------------------------------------------------------------------------------
    def save_run(
        self,
        run_id: str,
        as_of_week: date,
        profile: str,
        diagnostics: dict,
        channel_results: pd.DataFrame,
        market_effects: pd.DataFrame,
        reconciliation: pd.DataFrame,
        tv_rollup: pd.DataFrame,
    ) -> None:
        self.runs[run_id] = {
            "run_id": run_id,
            "as_of_week": as_of_week.isoformat(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "profile": profile,
            "diagnostics": _json_ready(diagnostics),
            "channel_results": channel_results.copy(),
            "market_effects": market_effects.copy(),
            "reconciliation": reconciliation.copy(),
            "tv_rollup": tv_rollup.copy(),
        }

    def list_runs(self) -> list[dict]:
        rows = [{k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)} for r in self.runs.values()]
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)

    def get_run(self, run_id: str) -> dict | None:
        r = self.runs.get(run_id)
        return {k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)} if r else None

    def latest_run_id(self) -> str | None:
        runs = self.list_runs()
        return runs[0]["run_id"] if runs else None

    def get_channel_results(self, run_id: str) -> pd.DataFrame:
        return self.runs[run_id]["channel_results"].copy()

    def get_market_effects(self, run_id: str) -> pd.DataFrame:
        return self.runs[run_id]["market_effects"].copy()

    def get_reconciliation(self, run_id: str) -> pd.DataFrame:
        return self.runs[run_id]["reconciliation"].copy()

    def get_tv_rollup(self, run_id: str, dimension: str | None = None) -> pd.DataFrame:
        df = self.runs[run_id]["tv_rollup"].copy()
        return df[df.dimension == dimension] if dimension else df


class PostgresRepository(InMemoryRepository):
    """Same interface, backed by SQLAlchemy Core against Postgres."""

    def __init__(self, url: str | None = None, engine: Engine | None = None) -> None:
        super().__init__()
        self.engine = engine or create_engine(url or database_url(), future=True)

    def migrate(self) -> None:
        with self.engine.begin() as conn:
            for path in sorted(SQL_DIR.glob("*.sql")):
                conn.execute(text(path.read_text(encoding="utf-8")))

    def reset(self) -> None:
        tables = [
            "tv_rollup",
            "reconciliation",
            "market_effects",
            "channel_results",
            "model_runs",
            "geo_tests",
            "attribution_weekly",
            "minute_sessions",
            "tv_airings",
            "weekly_outcomes",
            "weekly_media",
            "markets",
        ]
        with self.engine.begin() as conn:
            for t in tables:
                conn.execute(text(f"DELETE FROM {t}"))

    def _read(self, sql: str, **params) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)

    def _write_frame(self, df: pd.DataFrame, table: str) -> None:
        if df.empty:
            return
        df = df.copy()
        for col in df.columns:
            # Columns that are entirely None (an optional numeric with no values yet) must not be sent as text.
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        with self.engine.begin() as conn:
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=2000)

    # ---- inputs -------------------------------------------------------------------------------------
    def load_inputs(self, data) -> None:
        self.reset()
        self._write_frame(data.markets, "markets")
        self._write_frame(data.media, "weekly_media")
        self._write_frame(data.outcomes, "weekly_outcomes")
        self._write_frame(data.airings, "tv_airings")
        sessions = data.sessions.rename("sessions").reset_index().rename(columns={"index": "minute_ts"})
        sessions.columns = ["minute_ts", "sessions"]
        self._write_frame(sessions, "minute_sessions")
        self._write_frame(data.attribution, "attribution_weekly")
        for t in data.geo_tests:
            self.save_geo_test(t)

    def load_markets(self) -> pd.DataFrame:
        return self._read("SELECT market_id, name, region, population, tv_households FROM markets ORDER BY market_id")

    def load_media(self) -> pd.DataFrame:
        df = self._read("SELECT market_id, week_start, channel, spend, impressions FROM weekly_media")
        df["spend"] = df["spend"].astype(float)
        return df

    def load_outcomes(self) -> pd.DataFrame:
        df = self._read(
            "SELECT market_id, week_start, new_subscriptions, promo_depth, competitor_grps, price_index "
            "FROM weekly_outcomes"
        )
        for c in ("promo_depth", "competitor_grps", "price_index"):
            df[c] = df[c].astype(float)
        return df

    def load_airings(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        sql = "SELECT airing_id, aired_at, network, daypart, creative, impressions, cost FROM tv_airings WHERE 1=1"
        params: dict[str, Any] = {}
        if start is not None:
            sql += " AND aired_at >= :start"
            params["start"] = start
        if end is not None:
            sql += " AND aired_at < :end"
            params["end"] = end
        df = self._read(sql + " ORDER BY aired_at", **params)
        if len(df):
            df["cost"] = df["cost"].astype(float)
        return df

    def load_minute_sessions(self, start: datetime | None = None, end: datetime | None = None) -> pd.Series:
        sql = "SELECT minute_ts, sessions FROM minute_sessions WHERE 1=1"
        params: dict[str, Any] = {}
        if start is not None:
            sql += " AND minute_ts >= :start"
            params["start"] = start
        if end is not None:
            sql += " AND minute_ts < :end"
            params["end"] = end
        df = self._read(sql + " ORDER BY minute_ts", **params)
        return pd.Series(df.sessions.to_numpy(dtype=float), index=pd.DatetimeIndex(df.minute_ts))

    def load_attribution(self) -> pd.DataFrame:
        return self._read("SELECT week_start, channel, last_touch_conversions FROM attribution_weekly")

    # ---- geo tests ----------------------------------------------------------------------------------
    def save_geo_test(self, test: dict) -> None:
        t = _json_ready(dict(test))
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO geo_tests (test_id, name, channel, treatment_markets, control_markets, pre_start,
                                           test_start, test_end, incremental_spend, status, readout)
                    VALUES (:test_id, :name, :channel, :treatment, :control, :pre_start, :test_start, :test_end,
                            :incremental_spend, :status, :readout)
                    ON CONFLICT (test_id) DO UPDATE SET
                        name = EXCLUDED.name, channel = EXCLUDED.channel,
                        treatment_markets = EXCLUDED.treatment_markets, control_markets = EXCLUDED.control_markets,
                        pre_start = EXCLUDED.pre_start, test_start = EXCLUDED.test_start, test_end = EXCLUDED.test_end,
                        incremental_spend = EXCLUDED.incremental_spend, status = EXCLUDED.status,
                        readout = EXCLUDED.readout
                    """
                ),
                {
                    "test_id": t["test_id"],
                    "name": t["name"],
                    "channel": t["channel"],
                    "treatment": json.dumps(t["treatment_markets"]),
                    "control": json.dumps(t["control_markets"]),
                    "pre_start": t["pre_start"],
                    "test_start": t["test_start"],
                    "test_end": t["test_end"],
                    "incremental_spend": float(t.get("incremental_spend") or 0.0),
                    "status": t.get("status", "designed"),
                    "readout": json.dumps(t["readout"]) if t.get("readout") else None,
                },
            )

    def update_geo_test_readout(self, test_id: str, readout: dict, status: str = "read") -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE geo_tests SET readout = :readout, status = :status, "
                    "incremental_spend = COALESCE(:spend, incremental_spend) WHERE test_id = :test_id"
                ),
                {
                    "readout": json.dumps(_json_ready(readout)),
                    "status": status,
                    "spend": readout.get("incremental_spend"),
                    "test_id": test_id,
                },
            )

    @staticmethod
    def _row_to_test(r: dict) -> dict:
        out = dict(r)
        for k in ("treatment_markets", "control_markets", "readout"):
            if isinstance(out.get(k), str):
                out[k] = json.loads(out[k])
        for k in ("pre_start", "test_start", "test_end"):
            if isinstance(out.get(k), date | datetime):
                out[k] = out[k].isoformat()
        if out.get("incremental_spend") is not None:
            out["incremental_spend"] = float(out["incremental_spend"])
        return out

    def load_geo_tests(self, status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM geo_tests"
        params: dict[str, Any] = {}
        if status:
            sql += " WHERE status = :status"
            params["status"] = status
        df = self._read(sql + " ORDER BY test_start", **params)
        return [self._row_to_test(r) for r in df.to_dict("records")]

    def get_geo_test(self, test_id: str) -> dict | None:
        df = self._read("SELECT * FROM geo_tests WHERE test_id = :test_id", test_id=test_id)
        return self._row_to_test(df.to_dict("records")[0]) if len(df) else None

    # ---- runs ---------------------------------------------------------------------------------------
    def save_run(
        self, run_id, as_of_week, profile, diagnostics, channel_results, market_effects, reconciliation, tv_rollup
    ):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO model_runs (run_id, as_of_week, profile, diagnostics) "
                    "VALUES (:run_id, :as_of, :profile, :diag)"
                ),
                {
                    "run_id": run_id,
                    "as_of": as_of_week,
                    "profile": profile,
                    "diag": json.dumps(_json_ready(diagnostics)),
                },
            )
        for df, table in (
            (channel_results, "channel_results"),
            (market_effects, "market_effects"),
            (reconciliation, "reconciliation"),
            (tv_rollup, "tv_rollup"),
        ):
            if len(df):
                self._write_frame(df.assign(run_id=run_id), table)

    def list_runs(self) -> list[dict]:
        df = self._read(
            "SELECT run_id, as_of_week, created_at, profile, diagnostics FROM model_runs ORDER BY created_at DESC"
        )
        rows = []
        for r in df.to_dict("records"):
            r["as_of_week"] = pd.to_datetime(r["as_of_week"]).date().isoformat()
            r["created_at"] = pd.to_datetime(r["created_at"]).isoformat(timespec="seconds")
            if isinstance(r["diagnostics"], str):
                r["diagnostics"] = json.loads(r["diagnostics"])
            rows.append(r)
        return rows

    def get_run(self, run_id: str) -> dict | None:
        return next((r for r in self.list_runs() if r["run_id"] == run_id), None)

    def _run_frame(self, table: str, run_id: str, extra: str = "", **params) -> pd.DataFrame:
        df = self._read(f"SELECT * FROM {table} WHERE run_id = :run_id {extra}", run_id=run_id, **params)
        df = df.drop(columns=["run_id"])
        for c in df.columns:
            if df[c].dtype == object and c not in ("channel", "market_id", "dimension", "key", "harmonized_source"):
                try:
                    df[c] = pd.to_numeric(df[c])
                except (ValueError, TypeError):
                    pass
        return df

    def get_channel_results(self, run_id: str) -> pd.DataFrame:
        return self._run_frame("channel_results", run_id)

    def get_market_effects(self, run_id: str) -> pd.DataFrame:
        return self._run_frame("market_effects", run_id)

    def get_reconciliation(self, run_id: str) -> pd.DataFrame:
        return self._run_frame("reconciliation", run_id)

    def get_tv_rollup(self, run_id: str, dimension: str | None = None) -> pd.DataFrame:
        if dimension:
            return self._run_frame("tv_rollup", run_id, "AND dimension = :dim", dim=dimension)
        return self._run_frame("tv_rollup", run_id)


def get_repository(url: str | None = None) -> PostgresRepository:
    return PostgresRepository(url)
