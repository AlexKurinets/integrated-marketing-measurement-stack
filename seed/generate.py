"""Synthetic data generator.

Everything here is made up: markets, networks, creatives, spend, subscriptions and sessions. The data is
generated from a known response process (the same adstock and Hill forms the MMM uses, with known
parameters) so the tests can check that the model recovers what was put in. Two geo tests are embedded in
the spend history and one is designed for the future.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from measurement.adapters.synthetic import SyntheticAttribution, SyntheticSyndicatedData
from measurement.config import channel_names
from measurement.geo.matching import control_pool, match_markets
from measurement.mmm.transforms import geometric_adstock, hill_saturation
from seed.markets import CREATIVES, MARKETS, NETWORKS

ANNUAL_MEDIA_BUDGET = 40_000_000.0  # production footprint: about 120M people across 210 DMAs
FOOTPRINT_POPULATION = 120_000_000
BASE_WEEKLY_RATE = 0.00006  # organic new subscriptions per capita per week
BUDGET_SHARE = {"linear_tv": 0.35, "ctv": 0.20, "online_video": 0.15, "paid_social": 0.20, "paid_search": 0.10}
CPM = {"linear_tv": 18.0, "ctv": 32.0, "online_video": 14.0, "paid_social": 9.0, "paid_search": 45.0}

TRUE_PARAMS = {
    "decay": {"linear_tv": 0.60, "ctv": 0.45, "online_video": 0.35, "paid_social": 0.25, "paid_search": 0.15},
    "half_sat": {"linear_tv": 1.0, "ctv": 0.9, "online_video": 1.1, "paid_social": 0.8, "paid_search": 0.7},
    "slope": {"linear_tv": 1.3, "ctv": 1.2, "online_video": 1.1, "paid_social": 1.0, "paid_search": 1.0},
    "beta": {"linear_tv": 0.60, "ctv": 0.40, "online_video": 0.20, "paid_social": 0.35, "paid_search": 0.30},
    "max_lag": 8,
}

DAYPARTS = {
    "early_morning": (5, 9, 0.5),
    "daytime": (9, 16, 0.6),
    "early_fringe": (16, 19, 0.8),
    "prime": (19, 23, 1.0),
    "late_night": (23, 29, 0.45),
}


@dataclass
class SeedData:
    markets: pd.DataFrame
    media: pd.DataFrame
    outcomes: pd.DataFrame
    airings: pd.DataFrame
    sessions: pd.Series
    attribution: pd.DataFrame
    geo_tests: list[dict]
    truth: pd.DataFrame
    true_params: dict


def _market_frame(n_markets: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(n_markets):
        if i < len(MARKETS):
            name, region = MARKETS[i]
        else:
            name, region = f"Market {i + 1}", "Other"
        pop = int(np.clip(rng.lognormal(mean=np.log(420_000), sigma=0.75), 80_000, 3_500_000))
        rows.append(
            {
                "market_id": f"m{i + 1:03d}",
                "name": name,
                "region": region,
                "population": pop,
                "tv_households": int(pop * 0.38),
            }
        )
    return pd.DataFrame(rows)


def weekly_media_budget(markets: pd.DataFrame) -> float:
    """The synthetic footprint spends at the same per-capita rate as the production footprint."""
    return ANNUAL_MEDIA_BUDGET / 52.0 * float(markets.population.sum()) / FOOTPRINT_POPULATION


def _weekly_spend(markets: pd.DataFrame, weeks: list[date], rng: np.random.Generator) -> np.ndarray:
    """Spend array (M, T, C) in dollars with population-proportional allocation, flighting and noise."""
    M, T, C = len(markets), len(weeks), len(channel_names())
    pop_share = markets.population.to_numpy(dtype=float) / markets.population.sum()
    market_skew = rng.uniform(0.85, 1.15, size=M)
    weekly_budget = weekly_media_budget(markets)
    spend = np.zeros((M, T, C))
    for c_i, ch in enumerate(channel_names()):
        for t in range(T):
            season = 1.0 + 0.2 * np.sin(2 * np.pi * (t + 6) / 52.0)
            flight = 1.0
            if ch in ("linear_tv", "ctv"):
                flight = 0.35 if (t % 4 == 3) else 1.15
            if ch == "linear_tv" and t % 13 == 6:
                flight *= 1.6  # periodic heavy-up
            national = weekly_budget * BUDGET_SHARE[ch] * season * flight * rng.uniform(0.85, 1.15)
            spend[:, t, c_i] = national * pop_share * market_skew * rng.uniform(0.9, 1.1, size=M)
        # Regional flighting: individual markets go light or heavy in some weeks, which is what identifies
        # the response curve separately from national seasonality.
        pulse = rng.choice([0.5, 1.0, 1.5], size=(M, T), p=[0.08, 0.84, 0.08])
        spend[:, :, c_i] *= pulse
    return spend


def _embed_geo_tests(
    spend: np.ndarray, markets: pd.DataFrame, weeks: list[date], rng: np.random.Generator
) -> list[dict]:
    """Modify spend for two completed tests and return their records (controls are matched later)."""
    T = len(weeks)
    chans = channel_names()
    pop = markets.population.to_numpy(dtype=float)
    mid = np.argsort(np.abs(np.log(pop / np.median(pop))))  # mid-sized markets first
    M = len(markets)
    n_tv, n_ctv, n_social = max(3, M // 8), max(3, M // 8), max(3, M // 8)
    treat_tv = sorted(markets.market_id.iloc[mid[:n_tv]].tolist())
    treat_ctv = sorted(markets.market_id.iloc[mid[n_tv : n_tv + n_ctv]].tolist())
    ids = markets.market_id.tolist()

    tests = []
    # Linear TV go-dark holdout.
    s1, e1 = T - 40, T - 35
    for m in treat_tv:
        spend[ids.index(m), s1 : e1 + 1, chans.index("linear_tv")] = 0.0
    tests.append(
        {
            "test_id": "gt-linear-tv-holdout-2026q1",
            "name": "Linear TV go-dark holdout",
            "channel": "linear_tv",
            "treatment_markets": treat_tv,
            "control_markets": [],
            "pre_start": weeks[max(s1 - 26, 0)],
            "test_start": weeks[s1],
            "test_end": weeks[e1],
            "incremental_spend": 0.0,
            "status": "completed",
            "readout": None,
        }
    )
    # CTV spend-up.
    s2, e2 = T - 20, T - 15
    for m in treat_ctv:
        spend[ids.index(m), s2 : e2 + 1, chans.index("ctv")] *= 2.0
    tests.append(
        {
            "test_id": "gt-ctv-spend-up-2026q2",
            "name": "CTV spend doubling",
            "channel": "ctv",
            "treatment_markets": treat_ctv,
            "control_markets": [],
            "pre_start": weeks[max(s2 - 26, 0)],
            "test_start": weeks[s2],
            "test_end": weeks[e2],
            "incremental_spend": 0.0,
            "status": "completed",
            "readout": None,
        }
    )
    # A designed test that has not started.
    treat_social = sorted(markets.market_id.iloc[mid[n_tv + n_ctv : n_tv + n_ctv + n_social]].tolist())
    start = weeks[-1] + timedelta(weeks=2)
    tests.append(
        {
            "test_id": "gt-paid-social-holdout-2026q4",
            "name": "Paid social holdout (designed)",
            "channel": "paid_social",
            "treatment_markets": treat_social,
            "control_markets": [],
            "pre_start": weeks[max(len(weeks) - 26, 0)],
            "test_start": start,
            "test_end": start + timedelta(weeks=5),
            "incremental_spend": 0.0,
            "status": "designed",
            "readout": None,
        }
    )
    return tests


def _outcomes(spend, markets, weeks, rng, syndicated: SyntheticSyndicatedData):
    M, T, C = spend.shape
    chans = channel_names()
    pop = markets.population.to_numpy(dtype=float)
    base = pop * BASE_WEEKLY_RATE
    t = np.arange(T)
    season = 1.0 + 0.12 * np.sin(2 * np.pi * (t + 10) / 52.0) + 0.05 * np.cos(4 * np.pi * t / 52.0)
    trend = 1.0 + 0.002 * t

    promo_national = np.where(t % 6 == 2, rng.uniform(0.15, 0.3, size=T), 0.0)
    promo = np.clip(promo_national[None, :] + rng.normal(0, 0.01, size=(M, T)), 0, None)
    comp = syndicated.competitor_grps(markets.market_id.tolist(), weeks)
    comp_arr = (
        comp.pivot(index="market_id", columns="week_start", values="competitor_grps").loc[markets.market_id].to_numpy()
    )
    price = 1.0 + np.cumsum(rng.normal(0, 0.004, size=T))
    price = np.clip(price, 0.9, 1.1)
    price_arr = np.repeat(price[None, :], M, axis=0) * rng.uniform(0.995, 1.005, size=(M, T))
    ctrl_effect = np.exp(0.8 * promo - 0.0015 * (comp_arr - 80.0) - 0.6 * (price_arr - 1.0))

    scale = spend.reshape(-1, C).mean(axis=0)
    x = spend / scale[None, None, :]
    decay = np.array([TRUE_PARAMS["decay"][c] for c in chans])
    half = np.array([TRUE_PARAMS["half_sat"][c] for c in chans])
    slope = np.array([TRUE_PARAMS["slope"][c] for c in chans])
    beta = np.array([TRUE_PARAMS["beta"][c] for c in chans])
    mult = np.exp(rng.normal(0, 0.2, size=(M, C)))
    sat = hill_saturation(geometric_adstock(x, decay, TRUE_PARAMS["max_lag"]), half, slope)
    media = beta[None, None, :] * mult[:, None, :] * sat  # (M, T, C) in units of base
    mu = base[:, None] * season[None, :] * trend[None, :] * ctrl_effect + base[:, None] * media.sum(axis=-1)
    y = rng.poisson(np.clip(mu, 1.0, None))

    rows, truth_rows = [], []
    ids = markets.market_id.tolist()
    for m_i, m in enumerate(ids):
        for t_i, w in enumerate(weeks):
            rows.append(
                {
                    "market_id": m,
                    "week_start": w,
                    "new_subscriptions": int(y[m_i, t_i]),
                    "promo_depth": round(float(promo[m_i, t_i]), 4),
                    "competitor_grps": round(float(comp_arr[m_i, t_i]), 2),
                    "price_index": round(float(price_arr[m_i, t_i]), 4),
                }
            )
    inc = base[:, None, None] * media  # conversions
    for t_i, w in enumerate(weeks):
        for c_i, ch in enumerate(chans):
            truth_rows.append({"week_start": w, "channel": ch, "incremental": float(inc[:, t_i, c_i].sum())})
    return pd.DataFrame(rows), pd.DataFrame(truth_rows), {"scale": scale.tolist(), "market_multipliers": mult.tolist()}


def _tv(weeks: list[date], rng: np.random.Generator, tv_weeks: int, weekly_tv_spend: float):
    """Airings and minute-level sessions for the last tv_weeks weeks. Airing costs sum to the weekly TV budget."""
    start = datetime.combine(weeks[-tv_weeks], datetime.min.time())
    minutes = pd.date_range(start, periods=tv_weeks * 7 * 24 * 60, freq="min")
    hour = minutes.hour.to_numpy() + minutes.minute.to_numpy() / 60.0
    diurnal = 0.35 + 0.65 * np.clip(np.sin(np.pi * (hour - 6) / 18.0), 0, None)
    weekend = np.where(minutes.dayofweek >= 5, 1.1, 1.0)
    baseline = 12.0 + 36.0 * diurnal * weekend
    sessions = rng.poisson(baseline).astype(float)

    per_week = max(60, int(weekly_tv_spend / 330.0))
    airings = []
    k = 0
    for w in range(tv_weeks):
        for _ in range(per_week):
            dp = rng.choice(list(DAYPARTS.keys()), p=[0.12, 0.22, 0.18, 0.33, 0.15])
            h0, h1, dp_factor = DAYPARTS[dp]
            day = int(rng.integers(0, 7))
            minute_of_day = int(rng.uniform(h0, h1) * 60) % (24 * 60)
            aired = start + timedelta(days=w * 7 + day, minutes=minute_of_day)
            net, net_size = NETWORKS[int(rng.integers(0, len(NETWORKS)))]
            creative, cr_factor = CREATIVES[int(rng.integers(0, len(CREATIVES)))]
            impressions = rng.lognormal(np.log(25_000 * net_size * dp_factor), 0.35)
            cpm = rng.uniform(14.0, 38.0) * (1.15 if dp == "prime" else 1.0)
            k += 1
            airings.append(
                {
                    "airing_id": f"a{k:05d}",
                    "aired_at": aired,
                    "network": net,
                    "daypart": dp,
                    "creative": creative,
                    "impressions": impressions,
                    "cost": impressions / 1000.0 * cpm,
                    "cr_factor": cr_factor,
                }
            )
    airings_df = pd.DataFrame(airings)
    scale = weekly_tv_spend * tv_weeks / max(airings_df.cost.sum(), 1.0)
    airings_df["impressions"] = (airings_df.impressions * scale).round().astype(int)
    airings_df["cost"] = (airings_df.cost * scale).round(2)

    response_profile = np.array([0.35, 0.30, 0.20, 0.10, 0.05])
    for r in airings_df.itertuples(index=False):
        pos = int((r.aired_at - start).total_seconds() // 60)
        resp = r.impressions / 1_000.0 * r.cr_factor * rng.uniform(0.7, 1.3)
        for j, share in enumerate(response_profile):
            if pos + j < len(sessions):
                sessions[pos + j] += rng.poisson(resp * share)
    airings_df = airings_df.drop(columns=["cr_factor"]).sort_values("aired_at").reset_index(drop=True)
    return airings_df, pd.Series(sessions, index=minutes)


def generate(
    n_markets: int = 40, n_weeks: int = 104, seed: int = 42, as_of: date = date(2026, 9, 28), tv_weeks: int = 4
) -> SeedData:
    rng = np.random.default_rng(seed)
    markets = _market_frame(n_markets, rng)
    weeks = [as_of - timedelta(weeks=n_weeks - 1 - i) for i in range(n_weeks)]
    spend = _weekly_spend(markets, weeks, rng)
    tests = _embed_geo_tests(spend, markets, weeks, rng)
    syndicated = SyntheticSyndicatedData(seed=seed)
    outcomes, truth, extra = _outcomes(spend, markets, weeks, rng, syndicated)

    # Match controls on the pre-period of each test now that outcomes exist.
    wide = outcomes.pivot(index="week_start", columns="market_id", values="new_subscriptions").astype(float)
    for t in tests:
        idx = pd.to_datetime(wide.index).date
        pre = wide[(idx >= t["pre_start"]) & (idx < t["test_start"])]
        # Markets treated by another test that overlaps this one's window cannot serve as controls.
        overlapping = [
            o for o in tests if o is not t and o["test_start"] <= t["test_end"] and o["test_end"] >= t["pre_start"]
        ]
        exclude = tuple(m for o in overlapping for m in o["treatment_markets"])
        matches = match_markets(pre, t["treatment_markets"], n_controls=3, exclude=exclude)
        if len(control_pool(matches)) < 3:
            matches = match_markets(pre, t["treatment_markets"], n_controls=3)
        t["control_markets"] = control_pool(matches)

    chans = channel_names()
    media_rows = []
    ids = markets.market_id.tolist()
    for m_i, m in enumerate(ids):
        for t_i, w in enumerate(weeks):
            for c_i, ch in enumerate(chans):
                s = float(spend[m_i, t_i, c_i])
                media_rows.append(
                    {
                        "market_id": m,
                        "week_start": w,
                        "channel": ch,
                        "spend": round(s, 2),
                        "impressions": int(s / CPM[ch] * 1000),
                    }
                )
    media = pd.DataFrame(media_rows)

    weekly_tv = weekly_media_budget(markets) * BUDGET_SHARE["linear_tv"]
    airings, sessions = _tv(weeks, rng, tv_weeks, weekly_tv)
    attribution = SyntheticAttribution(seed=seed).last_touch_weekly(weeks, truth)
    params = {**TRUE_PARAMS, **extra}
    return SeedData(markets, media, outcomes, airings, sessions, attribution, tests, truth, params)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data and load it into the warehouse")
    parser.add_argument("--markets", type=int, default=40)
    parser.add_argument("--weeks", type=int, default=104)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)

    from measurement.repository import get_repository

    repo = get_repository(args.database_url)
    repo.migrate()
    data = generate(n_markets=args.markets, n_weeks=args.weeks, seed=args.seed)
    repo.load_inputs(data)
    summary = {
        "markets": len(data.markets),
        "weeks": len(data.outcomes.week_start.unique()),
        "media_rows": len(data.media),
        "airings": len(data.airings),
        "minute_rows": len(data.sessions),
        "geo_tests": [t["test_id"] for t in data.geo_tests],
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
