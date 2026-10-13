"""Panel construction from the warehouse tables and calibration targets from geo test readouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class Panel:
    markets: list[str]
    weeks: list[date]
    channels: list[str]
    controls: list[str]
    spend: np.ndarray  # (M, T, C)
    y: np.ndarray  # (M, T)
    controls_x: np.ndarray  # (M, T, K)
    population: np.ndarray  # (M,)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.spend.shape

    def market_index(self, market_id: str) -> int:
        return self.markets.index(market_id)

    def week_mask(self, start: date, end: date) -> np.ndarray:
        return np.array([(w >= start) and (w <= end) for w in self.weeks])

    def outcome_wide(self) -> pd.DataFrame:
        """Weeks as rows, markets as columns. The shape the geo test code works with."""
        return pd.DataFrame(self.y.T, index=pd.Index(self.weeks, name="week_start"), columns=self.markets)

    def spend_wide(self, channel: str) -> pd.DataFrame:
        c = self.channels.index(channel)
        return pd.DataFrame(self.spend[:, :, c].T, index=pd.Index(self.weeks, name="week_start"), columns=self.markets)

    @classmethod
    def from_frames(
        cls,
        markets: pd.DataFrame,
        media: pd.DataFrame,
        outcomes: pd.DataFrame,
        channels: list[str],
        controls: list[str],
    ) -> Panel:
        market_ids = sorted(markets.market_id.tolist())
        weeks = sorted(pd.to_datetime(outcomes.week_start).dt.date.unique().tolist())
        m_idx = {m: i for i, m in enumerate(market_ids)}
        w_idx = {w: i for i, w in enumerate(weeks)}
        M, T, C, K = len(market_ids), len(weeks), len(channels), len(controls)

        spend = np.zeros((M, T, C))
        media = media.copy()
        media["week_start"] = pd.to_datetime(media.week_start).dt.date
        for c_i, ch in enumerate(channels):
            sub = media[media.channel == ch]
            mi = sub.market_id.map(m_idx).to_numpy()
            wi = sub.week_start.map(w_idx).to_numpy()
            spend[mi, wi, c_i] = sub.spend.to_numpy(dtype=float)

        y = np.zeros((M, T))
        controls_x = np.zeros((M, T, K))
        outcomes = outcomes.copy()
        outcomes["week_start"] = pd.to_datetime(outcomes.week_start).dt.date
        mi = outcomes.market_id.map(m_idx).to_numpy()
        wi = outcomes.week_start.map(w_idx).to_numpy()
        y[mi, wi] = outcomes.new_subscriptions.to_numpy(dtype=float)
        for k_i, ctrl in enumerate(controls):
            controls_x[mi, wi, k_i] = outcomes[ctrl].to_numpy(dtype=float)

        pop = markets.set_index("market_id").loc[market_ids, "population"].to_numpy(dtype=float)
        return cls(market_ids, weeks, list(channels), list(controls), spend, y, controls_x, pop)


@dataclass
class CalibrationTarget:
    """A completed geo test expressed as a constraint on the model-implied incremental conversions."""

    test_id: str
    channel: str
    markets: list[str]
    week_start: date
    week_end: date
    incremental_mean: float
    incremental_se: float
    source: str = "geo_test"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_readout(cls, test: dict) -> CalibrationTarget:
        r = test["readout"]
        mean = float(r["incremental_conversions"])
        se = r.get("incremental_se")
        if se is None or not np.isfinite(float(se)) or float(se) <= 0:
            se = 0.2 * abs(mean) + 1.0  # no placebo distribution available: fall back to a wide read
        return cls(
            test_id=test["test_id"],
            channel=test["channel"],
            markets=list(test["treatment_markets"]),
            week_start=pd.to_datetime(test["test_start"]).date(),
            week_end=pd.to_datetime(test["test_end"]).date(),
            incremental_mean=mean,
            incremental_se=float(se),
        )
