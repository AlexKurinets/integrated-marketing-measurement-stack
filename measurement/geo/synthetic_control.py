"""Synthetic control counterfactual with placebo inference.

The treated series is the sum of the treatment markets. Each control market is normalized by its own
pre-period mean before fitting, so the non-negative, sum-to-one weights describe the shape of the
counterfactual and the pre-period level of the treated series sets its scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SCResult:
    treatment: list[str]
    controls: list[str]
    weights: dict[str, float]
    pre_weeks: list[date]
    post_weeks: list[date]
    actual_pre: np.ndarray
    counterfactual_pre: np.ndarray
    actual_post: np.ndarray
    counterfactual_post: np.ndarray

    @property
    def lift_abs(self) -> float:
        return float((self.actual_post - self.counterfactual_post).sum())

    @property
    def lift_pct(self) -> float:
        return float(self.lift_abs / max(self.counterfactual_post.sum(), 1e-9))

    @property
    def pre_rmspe(self) -> float:
        return float(np.sqrt(np.mean(((self.actual_pre - self.counterfactual_pre) / self.actual_pre.mean()) ** 2)))

    @property
    def post_rmspe(self) -> float:
        return float(np.sqrt(np.mean(((self.actual_post - self.counterfactual_post) / self.actual_pre.mean()) ** 2)))


def fit_weights(y_pre: np.ndarray, x_pre: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one that minimize squared pre-period error."""
    J = x_pre.shape[1]
    if J == 0:
        raise ValueError("No control markets to fit weights on")
    w0 = np.full(J, 1.0 / J)

    def loss(w):
        r = y_pre - x_pre @ w
        return float(r @ r)

    def grad(w):
        return -2.0 * x_pre.T @ (y_pre - x_pre @ w)

    res = minimize(
        loss,
        w0,
        jac=grad,
        bounds=[(0.0, 1.0)] * J,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(J)}],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, None)
    return w / max(w.sum(), 1e-12)


def synthetic_control(
    outcome_wide: pd.DataFrame,
    treatment: list[str],
    controls: list[str],
    pre_start: date,
    test_start: date,
    test_end: date,
) -> SCResult:
    idx = pd.to_datetime(outcome_wide.index).date
    pre = (idx >= pre_start) & (idx < test_start)
    post = (idx >= test_start) & (idx <= test_end)
    if pre.sum() < 4 or post.sum() < 1:
        raise ValueError("Need at least four pre-period weeks and one test week")

    treated = outcome_wide[treatment].sum(axis=1).to_numpy(dtype=float)
    ctrl = outcome_wide[controls].to_numpy(dtype=float)
    ctrl_mean = np.maximum(ctrl[pre].mean(axis=0), 1e-9)
    ctrl_norm = ctrl / ctrl_mean[None, :]
    level = float(treated[pre].mean())
    w = fit_weights(treated[pre] / level, ctrl_norm[pre])
    cf = level * (ctrl_norm @ w)
    return SCResult(
        treatment=list(treatment),
        controls=list(controls),
        weights={c: float(wi) for c, wi in zip(controls, w, strict=True)},
        pre_weeks=[d for d, p in zip(idx, pre, strict=True) if p],
        post_weeks=[d for d, p in zip(idx, post, strict=True) if p],
        actual_pre=treated[pre],
        counterfactual_pre=cf[pre],
        actual_post=treated[post],
        counterfactual_post=cf[post],
    )


def placebo_inference(
    outcome_wide: pd.DataFrame,
    treatment: list[str],
    controls: list[str],
    pre_start: date,
    test_start: date,
    test_end: date,
    observed: SCResult,
    n_permutations: int = 50,
    confidence: float = 0.9,
    seed: int = 0,
) -> dict:
    """Permutation inference: pretend each control (or random control subset) was treated and read its lift.

    The spread of placebo lifts gives a standard error for the observed lift and a p-value for the share of
    placebos at least as large in absolute terms.
    """
    rng = np.random.default_rng(seed)
    k = len(treatment)
    # Placebos draw from every untreated market in the panel, not only the matched controls, so that a
    # small control set still yields a usable null distribution.
    pool = [m for m in outcome_wide.columns if m not in treatment]
    placebo_pcts: list[float] = []
    attempts = 0
    while len(placebo_pcts) < n_permutations and attempts < n_permutations * 3:
        attempts += 1
        if len(pool) < k + 2:
            break
        fake_treat = rng.choice(pool, size=k, replace=False).tolist()
        fake_controls = [c for c in pool if c not in fake_treat]
        try:
            res = synthetic_control(outcome_wide, fake_treat, fake_controls, pre_start, test_start, test_end)
        except ValueError:
            continue
        placebo_pcts.append(res.lift_pct)
    placebo = np.array(placebo_pcts) if placebo_pcts else np.array([0.0])
    obs_pct = observed.lift_pct
    p_value = float((np.abs(placebo) >= abs(obs_pct)).mean()) if len(placebo_pcts) else 1.0
    se_pct = float(placebo.std(ddof=1)) if len(placebo_pcts) > 1 else float("nan")
    cf_total = float(observed.counterfactual_post.sum())
    se_abs = se_pct * cf_total if np.isfinite(se_pct) else float("nan")
    z = {0.8: 1.2816, 0.9: 1.6449, 0.95: 1.96}.get(confidence, 1.6449)
    return {
        "p_value": p_value,
        "placebo_count": len(placebo_pcts),
        "se_pct": se_pct,
        "incremental_se": se_abs,
        "ci_low": observed.lift_abs - z * se_abs if np.isfinite(se_abs) else float("nan"),
        "ci_high": observed.lift_abs + z * se_abs if np.isfinite(se_abs) else float("nan"),
        "confidence": confidence,
    }
