"""Harmonization: one table per channel that puts the MMM, the geo tests and attribution side by side.

Precedence rule, in plain English:
  1. Geo tests calibrate. A completed test on a channel is the ground truth for the lift it measured, and the
     MMM is fitted with that read as a likelihood term. The model-implied lift for the same markets and weeks
     is compared with the read. If the calibrated MMM still sits outside the tolerance, the test wins: the
     channel total is scaled to the test and the channel is flagged.
  2. The MMM allocates across channels. For channels without a recent test, the MMM figure stands.
  3. Attribution allocates within a channel. Last-touch counts never set a channel total; they are
     reported beside the harmonized figure so the over- and under-crediting is visible.

Like is compared with like: the test measured a lift in specific markets and weeks, so the comparison uses
the MMM's implied lift for those markets and weeks, not the channel's average rate over the whole horizon.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

RECONCILIATION_COLUMNS = [
    "channel",
    "spend",
    "mmm_incremental",
    "mmm_low",
    "mmm_high",
    "test_incremental",
    "test_low",
    "test_high",
    "attributed_conversions",
    "harmonized_incremental",
    "harmonized_source",
    "harmonized_roi",
    "mmm_vs_test_gap",
]

Z90 = 1.6449


def _latest_test_by_channel(test_reads: list[dict], as_of: date | None, max_age_weeks: int) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for t in test_reads:
        if not t.get("readout"):
            continue
        end = pd.to_datetime(t["test_end"]).date()
        if as_of is not None and end > as_of:
            continue
        if as_of is not None and (as_of - end) > timedelta(weeks=max_age_weeks):
            continue
        cur = latest.get(t["channel"])
        if cur is None or pd.to_datetime(cur["test_end"]).date() < end:
            latest[t["channel"]] = t
    return latest


def _test_view(mmm_incremental: float, test: dict, check: dict | None) -> tuple[float, float, float, float] | None:
    """What the channel total would be if the model matched the test exactly, with a 90% band, plus the gap.

    Returns (test_incremental, test_low, test_high, gap) where gap = (mmm - test) / test on a like-for-like basis.
    """
    ro = test["readout"]
    if check is not None and abs(float(check["implied_mean"])) > 1e-9 and abs(float(check["target"])) > 1e-9:
        implied = float(check["implied_mean"])
        target = float(check["target"])
        se = float(check.get("target_se") or 0.0)
        if np.sign(implied) != np.sign(target):
            return None  # the read and the model disagree on direction: keep the MMM figure, flag below
        ratio = target / implied
        low = (abs(target) - Z90 * se) / abs(implied)
        high = (abs(target) + Z90 * se) / abs(implied)
        return mmm_incremental * ratio, mmm_incremental * max(low, 0.0), mmm_incremental * high, implied / target - 1.0
    # Fallback when the fit carries no calibration record for this test: compare rates per dollar.
    inc = abs(float(ro["incremental_conversions"]))
    se = float(ro.get("incremental_se") or 0.0)
    dspend = abs(float(ro.get("incremental_spend") or 0.0))
    if dspend <= 0 or inc <= 0:
        return None
    rate, rate_low, rate_high = inc / dspend, max(inc - Z90 * se, 0.0) / dspend, (inc + Z90 * se) / dspend
    spend = float(test.get("_spend", 0.0))
    if spend <= 0:
        return None
    mmm_rate = mmm_incremental / spend
    return rate * spend, rate_low * spend, rate_high * spend, (mmm_rate - rate) / rate


def reconcile(
    channel_summary: pd.DataFrame,
    test_reads: list[dict],
    attribution: pd.DataFrame,
    value_per_conversion: float,
    calibration_checks: list[dict] | None = None,
    as_of: date | None = None,
    max_test_age_weeks: int = 52,
    tolerance: float = 0.25,
) -> pd.DataFrame:
    tests = _latest_test_by_channel(test_reads, as_of, max_test_age_weeks)
    checks = {c["test_id"]: c for c in (calibration_checks or [])}
    attr = attribution.groupby("channel").last_touch_conversions.sum() if len(attribution) else pd.Series(dtype=float)
    rows = []
    for _, r in channel_summary.iterrows():
        ch = r.channel
        spend = float(r.spend)
        mmm = float(r.incremental_mean)
        row = {
            "channel": ch,
            "spend": spend,
            "mmm_incremental": mmm,
            "mmm_low": float(r.incremental_low),
            "mmm_high": float(r.incremental_high),
            "test_incremental": None,
            "test_low": None,
            "test_high": None,
            "attributed_conversions": float(attr.get(ch, 0.0)),
            "harmonized_incremental": mmm,
            "harmonized_source": "mmm",
            "harmonized_roi": None,
            "mmm_vs_test_gap": None,
        }
        t = tests.get(ch)
        if t is not None:
            t = {**t, "_spend": spend}
            view = _test_view(mmm, t, checks.get(t["test_id"]))
            if view is None:
                row["harmonized_source"] = f"mmm_unreconciled:{t['test_id']}"
            else:
                test_inc, test_low, test_high, gap = view
                row["test_incremental"], row["test_low"], row["test_high"] = test_inc, test_low, test_high
                row["mmm_vs_test_gap"] = float(gap) if np.isfinite(gap) else None
                if np.isfinite(gap) and abs(gap) <= tolerance:
                    row["harmonized_source"] = f"mmm_calibrated:{t['test_id']}"
                else:
                    row["harmonized_incremental"] = test_inc
                    row["harmonized_source"] = f"geo_test:{t['test_id']}"
        row["harmonized_roi"] = row["harmonized_incremental"] * value_per_conversion / max(spend, 1e-9)
        rows.append(row)
    return pd.DataFrame(rows, columns=RECONCILIATION_COLUMNS)


def explain_channel(
    row: pd.Series | dict, labels: dict[str, str] | None = None, value_per_conversion: float = 180.0
) -> str:
    """A plain-English sentence or two for a reconciliation row, the kind that goes into a client readout."""
    r = dict(row)
    label = (labels or {}).get(r["channel"], r["channel"])
    spend = float(r["spend"])
    mmm = float(r["mmm_incremental"])
    text = (
        f"{label}: the MMM attributes about {mmm:,.0f} incremental conversions to ${spend:,.0f} of spend "
        f"(ROI {float(r['harmonized_roi']):.2f} at ${value_per_conversion:,.0f} per conversion)."
    )
    gap = r.get("mmm_vs_test_gap")
    src = str(r.get("harmonized_source", "mmm"))
    if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
        direction = "above" if gap > 0 else "below"
        test_id = src.split(":", 1)[1] if ":" in src else "the latest geo test"
        if src.startswith("mmm_calibrated"):
            text += (
                f" The calibrated MMM sits {abs(gap) * 100:.0f}% {direction} the lift read by {test_id}, "
                f"within tolerance, so the MMM figure stands."
            )
        else:
            text += (
                f" The MMM sits {abs(gap) * 100:.0f}% {direction} the lift read by {test_id}, outside tolerance, "
                f"so the channel total is scaled to the test and flagged for review."
            )
    elif src.startswith("mmm_unreconciled"):
        text += (
            f" The latest geo test ({src.split(':', 1)[1]}) read in the opposite direction to the model, which usually"
            f" means the test was under-powered; the MMM figure stands and the test is flagged for review."
        )
    else:
        text += " No recent geo test covers this channel, so the MMM figure stands."
    attributed = float(r.get("attributed_conversions") or 0.0)
    harmonized = float(r["harmonized_incremental"])
    if harmonized > 0:
        ratio = attributed / harmonized
        if ratio < 0.7:
            text += (
                f" Last-touch attribution credits only {attributed:,.0f}, an under-count typical of upper-funnel media."
            )
        elif ratio > 1.3:
            text += f" Last-touch attribution credits {attributed:,.0f}, an over-count typical of lower-funnel media."
        else:
            text += f" Last-touch attribution credits {attributed:,.0f}, close to the harmonized figure."
    return text
