"""Spot-level TV response: baseline-and-lift attribution of site sessions to individual airings.

For each airing, the baseline is the median of site sessions in the minutes before it aired, excluding any
minute that falls inside an earlier airing's response window. Sessions above that baseline in the minutes
after the airing are the response. Where response windows overlap, each minute's lift is split between the
airings covering it in proportion to their impressions.

Clipping at zero means pure noise also produces a small positive "lift". The same procedure is therefore
run at a sample of random clean minutes, and the mean placebo lift is subtracted from every airing, so the
aggregate attribution is not biased upward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from measurement.config import tv_config

ROLLUP_DIMENSIONS = ("network", "daypart", "creative")


def _minute_index(sessions: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    s = sessions.sort_index()
    idx = pd.DatetimeIndex(s.index)
    return idx, s.to_numpy(dtype=float)


def _window_lift(values: np.ndarray, p: int, base_w: int, resp_w: int, contaminated: np.ndarray) -> tuple[float, float]:
    n = len(values)
    lo, hi = max(p - base_w, 0), max(min(p, n), 0)
    window = values[lo:hi]
    clean = window[~contaminated[lo:hi]]
    if len(clean) >= 3:
        base = float(np.median(clean))
    elif len(window):
        base = float(np.median(window))
    else:
        return float("nan"), 0.0
    post = values[max(p, 0) : min(p + resp_w, n)]
    return base, float(np.clip(post - base, 0.0, None).sum())


def placebo_lift(
    values: np.ndarray, contaminated: np.ndarray, base_w: int, resp_w: int, samples: int, seed: int
) -> float:
    """Mean lift the method finds at random minutes with no airing nearby."""
    n = len(values)
    candidates = np.arange(base_w, n - resp_w)
    if len(candidates) == 0:
        return 0.0
    bad = np.convolve(contaminated.astype(int), np.ones(base_w + resp_w, dtype=int), mode="full")
    # candidate p is clean when no contaminated minute falls in [p - base_w, p + resp_w)
    clean = candidates[bad[candidates + resp_w - 1] == 0]
    if len(clean) < 20:
        return 0.0
    rng = np.random.default_rng(seed)
    picks = rng.choice(clean, size=min(samples, len(clean)), replace=False)
    lifts = [_window_lift(values, int(p), base_w, resp_w, contaminated)[1] for p in picks]
    return float(np.mean(lifts))


def attribute_spots(sessions: pd.Series, airings: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or tv_config()
    base_w = int(cfg["baseline_window_minutes"])
    resp_w = int(cfg["response_window_minutes"])
    contam_w = int(cfg["contamination_minutes"])
    n_placebo = int(cfg.get("placebo_samples", 0))
    placebo_seed = int(cfg.get("placebo_seed", 0))

    idx, values = _minute_index(sessions)
    n = len(idx)
    if n == 0 or airings.empty:
        return airings.assign(baseline=np.nan, response_sessions=0.0, placebo_sessions=0.0, attributed_sessions=0.0)

    start = idx[0]
    pos = {t: i for i, t in enumerate(idx)}
    air = airings.copy().reset_index(drop=True)
    air["aired_at"] = pd.to_datetime(air["aired_at"]).dt.floor("min")
    air["_pos"] = air["aired_at"].map(lambda t: pos.get(t, int((t - start).total_seconds() // 60)))

    contaminated = np.zeros(n, dtype=bool)
    for p in air["_pos"]:
        lo, hi = max(int(p), 0), min(int(p) + contam_w, n)
        if lo < n:
            contaminated[lo:hi] = True

    baselines = np.full(len(air), np.nan)
    raw = np.zeros(len(air))
    for i, p in enumerate(air["_pos"]):
        baselines[i], raw[i] = _window_lift(values, int(p), base_w, resp_w, contaminated)
    air["baseline"] = baselines
    air["response_sessions"] = raw

    # Minute-level lift split across the airings whose response windows cover that minute.
    cover: dict[int, list[int]] = {}
    for i, p in enumerate(air["_pos"]):
        for m in range(int(p), min(int(p) + resp_w, n)):
            if m >= 0:
                cover.setdefault(m, []).append(i)
    attributed = np.zeros(len(air))
    share_sum = np.zeros(len(air))
    imps = air["impressions"].to_numpy(dtype=float)
    for m, ids in cover.items():
        base = np.nanmean(baselines[ids])
        lift = max(values[m] - base, 0.0) if np.isfinite(base) else 0.0
        share = imps[ids] / max(imps[ids].sum(), 1e-9)
        for j, i in enumerate(ids):
            attributed[i] += lift * share[j]
            share_sum[i] += share[j]

    placebo = placebo_lift(values, contaminated, base_w, resp_w, n_placebo, placebo_seed) if n_placebo else 0.0
    expected_noise = placebo * share_sum / resp_w
    air["placebo_sessions"] = expected_noise
    air["attributed_sessions"] = np.clip(attributed - expected_noise, 0.0, None)
    att = air["attributed_sessions"].to_numpy(dtype=float)
    air["cost_per_response"] = np.where(att > 0, air["cost"].to_numpy(dtype=float) / np.maximum(att, 1e-9), np.nan)
    return air.drop(columns=["_pos"])


def rollup(attributed: pd.DataFrame, dimensions: tuple[str, ...] = ROLLUP_DIMENSIONS) -> pd.DataFrame:
    rows = []
    for dim in dimensions:
        g = attributed.groupby(dim).agg(
            airings=("airing_id", "count"),
            impressions=("impressions", "sum"),
            cost=("cost", "sum"),
            attributed_sessions=("attributed_sessions", "sum"),
        )
        for key, r in g.iterrows():
            cpr = float(r.cost) / float(r.attributed_sessions) if r.attributed_sessions > 0 else None
            rows.append(
                {
                    "dimension": dim,
                    "key": str(key),
                    "airings": int(r.airings),
                    "impressions": int(r.impressions),
                    "cost": float(r.cost),
                    "attributed_sessions": float(r.attributed_sessions),
                    "cost_per_response": cpr,
                }
            )
    return pd.DataFrame(rows)


def weekly_response(attributed: pd.DataFrame) -> pd.DataFrame:
    """Attributed sessions by week, the series the MMM team uses as a sanity check on the TV channel."""
    a = attributed.copy()
    a["week_start"] = pd.to_datetime(a["aired_at"]).dt.to_period("W-SUN").dt.start_time.dt.date
    return a.groupby("week_start", as_index=False).agg(
        airings=("airing_id", "count"), cost=("cost", "sum"), attributed_sessions=("attributed_sessions", "sum")
    )
