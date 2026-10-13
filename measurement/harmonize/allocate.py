"""Budget scenarios on the fitted response curves.

The MMM gives each channel a Hill curve. At steady state (constant weekly spend) the normalized adstock
equals the normalized spend, so expected weekly incremental conversions for a channel are

    n_markets * beta * hill(weekly_spend / (n_markets * spend_scale), half_sat, slope)

which treats the market mix as fixed. That is good enough for a what-if; the production build re-ran the
posterior for the final plan.
"""

from __future__ import annotations

import numpy as np

from measurement.mmm.transforms import hill_saturation


def expected_weekly_conversions(params: dict, weekly_spend: float) -> float:
    n = float(params["n_markets"])
    u = weekly_spend / max(n * float(params["spend_scale"]), 1e-9)
    return float(n * float(params["beta"]) * hill_saturation(np.array([u]), params["half_sat"], params["slope"])[0])


def marginal_conversions(params: dict, weekly_spend: float, delta: float = 1000.0) -> float:
    """Incremental conversions from the next `delta` of weekly spend."""
    return expected_weekly_conversions(params, weekly_spend + delta) - expected_weekly_conversions(params, weekly_spend)


def simulate_shift(
    response_params: dict[str, dict],
    current_weekly_spend: dict[str, float],
    from_channel: str,
    to_channel: str,
    amount: float,
) -> dict:
    if from_channel not in response_params or to_channel not in response_params:
        raise KeyError("Unknown channel")
    if amount < 0 or amount > current_weekly_spend.get(from_channel, 0.0):
        raise ValueError("Amount must be between zero and the current weekly spend of the source channel")
    before = {
        ch: expected_weekly_conversions(p, current_weekly_spend.get(ch, 0.0)) for ch, p in response_params.items()
    }
    new_spend = dict(current_weekly_spend)
    new_spend[from_channel] -= amount
    new_spend[to_channel] = new_spend.get(to_channel, 0.0) + amount
    after = {ch: expected_weekly_conversions(p, new_spend.get(ch, 0.0)) for ch, p in response_params.items()}
    total_before, total_after = sum(before.values()), sum(after.values())
    return {
        "from_channel": from_channel,
        "to_channel": to_channel,
        "weekly_amount": amount,
        "weekly_spend_before": current_weekly_spend,
        "weekly_spend_after": new_spend,
        "conversions_before": before,
        "conversions_after": after,
        "total_before": total_before,
        "total_after": total_after,
        "delta_conversions": total_after - total_before,
        "delta_pct": (total_after - total_before) / max(total_before, 1e-9),
    }


def propose_reallocation(
    response_params: dict[str, dict],
    current_weekly_spend: dict[str, float],
    max_shift_pct: float = 0.2,
    step_pct: float = 0.02,
    max_steps: int = 500,
) -> dict:
    """Greedy hill climb: move small slices from the channel with the lowest marginal return to the highest,
    never moving more than max_shift_pct of any channel's current budget."""
    spend = dict(current_weekly_spend)
    floor = {ch: s * (1 - max_shift_pct) for ch, s in spend.items()}
    cap = {ch: s * (1 + max_shift_pct) for ch, s in spend.items()}
    total_budget = sum(spend.values())
    step = total_budget * step_pct / max(len(spend), 1)
    moves: list[dict] = []
    for _ in range(max_steps):
        marg = {ch: marginal_conversions(response_params[ch], spend[ch], step) for ch in spend}
        donors = [ch for ch in spend if spend[ch] - step >= floor[ch]]
        receivers = [ch for ch in spend if spend[ch] + step <= cap[ch]]
        if not donors or not receivers:
            break
        low = min(donors, key=lambda ch: marg[ch])
        high = max(receivers, key=lambda ch: marg[ch])
        if low == high or marg[high] <= marg[low] * 1.02:
            break
        spend[low] -= step
        spend[high] += step
        moves.append({"from": low, "to": high, "amount": step})
    before = sum(expected_weekly_conversions(response_params[ch], current_weekly_spend[ch]) for ch in spend)
    after = sum(expected_weekly_conversions(response_params[ch], spend[ch]) for ch in spend)
    summary = {}
    for ch in spend:
        summary[ch] = {
            "current": current_weekly_spend[ch],
            "proposed": spend[ch],
            "change_pct": (spend[ch] - current_weekly_spend[ch]) / max(current_weekly_spend[ch], 1e-9),
        }
    return {
        "plan": summary,
        "moves": len(moves),
        "total_before": before,
        "total_after": after,
        "delta_pct": (after - before) / max(before, 1e-9),
    }
