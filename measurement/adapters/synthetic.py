"""Synthetic stand-ins for the syndicated data feed and the attribution platform."""

from __future__ import annotations

import zlib
from datetime import date

import numpy as np
import pandas as pd

from measurement.config import channel_names


class SyntheticSyndicatedData:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def competitor_grps(self, market_ids: list[str], weeks: list[date]) -> pd.DataFrame:
        rows = []
        base = {m: self.rng.uniform(40, 120) for m in market_ids}
        for w_i, week in enumerate(weeks):
            season = 1.0 + 0.15 * np.sin(2 * np.pi * w_i / 52)
            for m in market_ids:
                rows.append((m, week, round(float(base[m] * season * self.rng.uniform(0.8, 1.2)), 2)))
        return pd.DataFrame(rows, columns=["market_id", "week_start", "competitor_grps"])

    def audience_composition(self, network: str, daypart: str) -> dict[str, float]:
        h = (zlib.crc32(f"{network}|{daypart}".encode()) % 1000) / 1000.0
        return {"adults_25_54": round(0.35 + 0.3 * h, 3), "women_25_54": round(0.18 + 0.2 * h, 3)}


class SyntheticAttribution:
    """Last-touch counts that over-credit search and under-credit TV, the way last-touch does in practice."""

    LAST_TOUCH_BIAS = {
        "linear_tv": 0.15,
        "ctv": 0.35,
        "online_video": 0.5,
        "paid_social": 0.9,
        "paid_search": 2.4,
    }

    def __init__(self, seed: int = 11):
        self.rng = np.random.default_rng(seed)

    def last_touch_weekly(self, weeks: list[date], true_incremental: pd.DataFrame | None = None) -> pd.DataFrame:
        rows = []
        for week in weeks:
            for ch in channel_names():
                truth = 0.0
                if true_incremental is not None:
                    sel = true_incremental[(true_incremental.week_start == week) & (true_incremental.channel == ch)]
                    truth = float(sel.incremental.sum()) if len(sel) else 0.0
                count = truth * self.LAST_TOUCH_BIAS.get(ch, 1.0) * self.rng.uniform(0.9, 1.1)
                rows.append((week, ch, int(round(count))))
        return pd.DataFrame(rows, columns=["week_start", "channel", "last_touch_conversions"])
