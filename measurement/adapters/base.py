"""Protocols for external data sources.

In production, syndicated competitor GRPs and audience composition came from a Nielsen-style feed, and
last-touch conversions came from the brand's attribution platform. Both are replaced here by synthetic
implementations with the same interface, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class SyndicatedDataAdapter(Protocol):
    def competitor_grps(self, market_ids: list[str], weeks: list[date]) -> pd.DataFrame:
        """Return columns market_id, week_start, competitor_grps."""
        ...

    def audience_composition(self, network: str, daypart: str) -> dict[str, float]:
        """Return the share of the target audience reached by a network and daypart."""
        ...


class AttributionAdapter(Protocol):
    def last_touch_weekly(self, weeks: list[date]) -> pd.DataFrame:
        """Return columns week_start, channel, last_touch_conversions."""
        ...
