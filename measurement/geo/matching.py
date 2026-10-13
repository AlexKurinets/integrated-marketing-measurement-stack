"""Matched-market selection for geo tests.

A control market is a good match for a treatment market when their outcome series move together in the
pre-period and when they are of similar size, so that the synthetic control does not have to stretch to
reproduce the treated series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize(series: pd.Series) -> np.ndarray:
    v = series.to_numpy(dtype=float)
    return v / max(float(np.mean(v)), 1e-9)


def market_distance(a: pd.Series, b: pd.Series) -> dict:
    """Correlation of week-over-week changes plus a size penalty. Lower score is a better match."""
    na, nb = _normalize(a), _normalize(b)
    da, db = np.diff(na), np.diff(nb)
    if da.std() < 1e-12 or db.std() < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(da, db)[0, 1])
    scale_ratio = float(a.mean() / max(b.mean(), 1e-9))
    score = (1.0 - corr) + 0.5 * abs(np.log(max(scale_ratio, 1e-9)))
    return {"correlation": corr, "scale_ratio": scale_ratio, "score": score}


def match_markets(
    outcome_wide: pd.DataFrame,
    treatment: list[str],
    n_controls: int = 3,
    exclude: tuple[str, ...] = (),
    min_correlation: float = 0.0,
) -> pd.DataFrame:
    """Rank candidate controls for each treatment market. Returns one row per (treatment, control) pair."""
    rows = []
    candidates = [m for m in outcome_wide.columns if m not in treatment and m not in exclude]
    for t in treatment:
        scored = []
        for c in candidates:
            d = market_distance(outcome_wide[t], outcome_wide[c])
            if d["correlation"] >= min_correlation:
                scored.append((c, d))
        scored.sort(key=lambda item: item[1]["score"])
        for rank, (c, d) in enumerate(scored[:n_controls], start=1):
            rows.append({"treatment": t, "control": c, "rank": rank, **d})
    return pd.DataFrame(rows, columns=["treatment", "control", "rank", "correlation", "scale_ratio", "score"])


def control_pool(matches: pd.DataFrame) -> list[str]:
    """Distinct controls across all treatment markets, best matches first."""
    if matches.empty:
        return []
    ordered = matches.sort_values(["score"]).control.tolist()
    seen: list[str] = []
    for c in ordered:
        if c not in seen:
            seen.append(c)
    return seen


def select_treatment_markets(
    outcome_wide: pd.DataFrame,
    population: pd.Series,
    n_treat: int,
    seed: int = 0,
    exclude: tuple[str, ...] = (),
) -> list[str]:
    """Pick treatment markets that are easy to match: mid-sized and well correlated with the rest of the panel.

    Markets are scored by their average correlation with all other markets, and the top candidates are
    sampled with a fixed seed so a design is reproducible.
    """
    rng = np.random.default_rng(seed)
    markets = [m for m in outcome_wide.columns if m not in exclude]
    diffs = np.diff(np.stack([_normalize(outcome_wide[m]) for m in markets], axis=1), axis=0)
    corr = np.corrcoef(diffs.T)
    np.fill_diagonal(corr, np.nan)
    avg_corr = np.nanmean(corr, axis=1)
    pop = population.reindex(markets).to_numpy(dtype=float)
    size_penalty = np.abs(np.log(pop / np.median(pop)))
    score = avg_corr - 0.25 * size_penalty
    order = np.argsort(-score)
    pool = [markets[i] for i in order[: max(n_treat * 2, n_treat)]]
    chosen = rng.choice(pool, size=n_treat, replace=False)
    return sorted(chosen.tolist())
