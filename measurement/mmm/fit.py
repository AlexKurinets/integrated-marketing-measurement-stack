"""Fit the MMM and turn posterior samples into the tables the rest of the stack consumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.diagnostics import split_gelman_rubin
from numpyro.infer import MCMC, NUTS

from measurement.config import channels as channel_specs
from measurement.config import model_config
from measurement.mmm.data import CalibrationTarget, Panel
from measurement.mmm.model import mmm_model
from measurement.mmm.transforms import fourier_terms


@dataclass
class MMMResult:
    channels: list[str]
    markets: list[str]
    weeks: list[date]
    channel_summary: pd.DataFrame
    market_effects: pd.DataFrame
    diagnostics: dict
    response_params: dict
    national_fit: pd.DataFrame
    samples: dict = field(default_factory=dict, repr=False)


def _prior_arrays(panel: Panel, spend_scale: np.ndarray, y_scale: np.ndarray) -> dict:
    specs = {c.name: c for c in channel_specs()}
    decay_mean, beta_mean, max_lag = [], [], 1
    for c_i, ch in enumerate(panel.channels):
        spec = specs[ch]
        decay_mean.append(spec.prior_decay_mean)
        # prior_roi_mean is incremental conversions per 1,000 of spend; at mean spend the Hill term is about 0.5.
        conv_per_week = spec.prior_roi_mean / 1000.0 * spend_scale[c_i]
        beta_mean.append(max(conv_per_week / float(np.mean(y_scale)) / 0.5, 1e-4))
        max_lag = max(max_lag, spec.adstock_max_lag)
    return {
        "decay_mean": jnp.array(decay_mean),
        "beta_mean": jnp.array(beta_mean),
        "beta_sd_log": 0.5,
        "baseline_mean": float(model_config()["baseline"]["share_mean"]),
        "baseline_sd": float(model_config()["baseline"]["share_sd"]),
        "market_sd": float(model_config()["hierarchy"]["market_sd_prior"]),
        "max_lag": max_lag,
    }


def counterfactual_spend(panel: Panel, target: CalibrationTarget) -> np.ndarray:
    """Spend the treatment markets would have had without the test: the per-capita spend of all other markets."""
    c = panel.channels.index(target.channel)
    treat = np.array([m in target.markets for m in panel.markets])
    weeks = panel.week_mask(target.week_start, target.week_end)
    x_cf = panel.spend.copy()
    others = ~treat
    per_capita = panel.spend[others][:, :, c].sum(axis=0) / max(panel.population[others].sum(), 1.0)
    for m_i in np.where(treat)[0]:
        x_cf[m_i, weeks, c] = per_capita[weeks] * panel.population[m_i]
    return x_cf


def _build_calibration(panel: Panel, targets: list[CalibrationTarget], spend_scale: np.ndarray) -> list[dict]:
    inflate = float(model_config()["calibration"]["se_inflation"])
    out = []
    for t in targets:
        if t.channel not in panel.channels:
            continue
        c_i = panel.channels.index(t.channel)
        x_cf = counterfactual_spend(panel, t)[:, :, c_i] / spend_scale[c_i]
        out.append(
            {
                "test_id": t.test_id,
                "channel_idx": panel.channels.index(t.channel),
                "market_mask": jnp.array([1.0 if m in t.markets else 0.0 for m in panel.markets]),
                "week_mask": jnp.array(panel.week_mask(t.week_start, t.week_end).astype(float)),
                "x_cf": jnp.array(x_cf),
                "mean": float(t.incremental_mean),
                "se": float(max(t.incremental_se * inflate, 1e-3)),
            }
        )
    return out


def fit_mmm(
    panel: Panel,
    calibration: list[CalibrationTarget] | None = None,
    profile: str = "quick",
    seed: int = 0,
    value_per_conversion: float = 180.0,
) -> MMMResult:
    cfg = model_config()
    prof = cfg["profiles"][profile]
    M, T, C = panel.shape
    H = int(prof["holdout_weeks"])

    spend_scale = np.maximum(panel.spend.reshape(-1, C).mean(axis=0), 1e-6)
    x = panel.spend / spend_scale[None, None, :]
    y_scale = np.maximum(panel.y.mean(axis=1), 1e-6)
    y_norm = panel.y / y_scale[:, None]
    flat_controls = panel.controls_x.reshape(-1, panel.controls_x.shape[-1])
    ctrl_mean = flat_controls.mean(axis=0)
    ctrl_sd = np.maximum(flat_controls.std(axis=0), 1e-6)
    controls = (panel.controls_x - ctrl_mean) / ctrl_sd
    fourier = fourier_terms(T, cfg["seasonality"]["period_weeks"], cfg["seasonality"]["fourier_order"])
    t_lin = (np.arange(T) - T / 2.0) / T
    train_mask = np.arange(T) < (T - H)
    prior = _prior_arrays(panel, spend_scale, y_scale)
    calib = _build_calibration(panel, list(calibration or []), spend_scale)

    kernel = NUTS(mmm_model, target_accept_prob=0.85, max_tree_depth=7)
    mcmc = MCMC(
        kernel,
        num_warmup=int(prof["num_warmup"]),
        num_samples=int(prof["num_samples"]),
        num_chains=int(prof["num_chains"]),
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        x=jnp.array(x),
        controls=jnp.array(controls),
        fourier=jnp.array(fourier),
        t_lin=jnp.array(t_lin),
        y_scale=jnp.array(y_scale),
        train_mask=jnp.array(train_mask),
        prior=prior,
        calib=calib,
        y=jnp.array(y_norm),
        extra_fields=("diverging",),
    )
    grouped = mcmc.get_samples(group_by_chain=True)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    diverging = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())

    rhat = {}
    for name in ("decay", "half_sat", "slope", "beta_national", "sigma"):
        rhat[name] = float(np.max(np.asarray(split_gelman_rubin(grouped[name]))))

    media = samples["media_contrib"] * y_scale[None, :, None, None]  # (S, M, T, C) in conversions
    inc = media.sum(axis=(1, 2))  # (S, C)
    spend_total = panel.spend.sum(axis=(0, 1))
    beta_conv = (samples["beta_market"] * y_scale[None, :, None]).mean(axis=(0, 1))  # (C,)

    rows = []
    response_params = {}
    for c_i, ch in enumerate(panel.channels):
        mean = float(inc[:, c_i].mean())
        low, high = np.percentile(inc[:, c_i], [5, 95]).tolist()
        roi = mean * value_per_conversion / max(spend_total[c_i], 1e-6)
        params = {
            "decay": float(samples["decay"][:, c_i].mean()),
            "half_sat": float(samples["half_sat"][:, c_i].mean()),
            "slope": float(samples["slope"][:, c_i].mean()),
            "beta": float(beta_conv[c_i]),
            "spend_scale": float(spend_scale[c_i]),
            "n_markets": M,
        }
        response_params[ch] = params
        rows.append(
            {
                "channel": ch,
                "spend": float(spend_total[c_i]),
                "incremental_mean": mean,
                "incremental_low": float(low),
                "incremental_high": float(high),
                "roi_mean": float(roi),
                **{k: params[k] for k in ("decay", "half_sat", "slope", "beta", "spend_scale")},
            }
        )
    channel_summary = pd.DataFrame(rows)

    bm = samples["beta_market"].mean(axis=0)  # (M, C)
    market_effects = pd.DataFrame(
        [(m, ch, float(bm[m_i, c_i])) for m_i, m in enumerate(panel.markets) for c_i, ch in enumerate(panel.channels)],
        columns=["market_id", "channel", "beta_market"],
    )

    mu_conv = (samples["mu"] * y_scale[None, :, None]).mean(axis=0)  # (M, T)
    national_actual = panel.y.sum(axis=0)
    national_fitted = mu_conv.sum(axis=0)
    ape = np.abs(national_fitted - national_actual) / np.maximum(national_actual, 1.0)
    national_df = pd.DataFrame(
        {"week_start": panel.weeks, "actual": national_actual, "fitted": national_fitted, "holdout": ~train_mask}
    )

    calib_check = []
    for i, c in enumerate(calib):
        implied = samples[f"calib_implied_{i}"]
        calib_check.append(
            {
                "test_id": c["test_id"],
                "target": c["mean"],
                "target_se": c["se"],
                "implied_mean": float(implied.mean()),
                "implied_low": float(np.percentile(implied, 5)),
                "implied_high": float(np.percentile(implied, 95)),
            }
        )

    diagnostics = {
        "profile": profile,
        "n_markets": M,
        "n_weeks": T,
        "holdout_weeks": H,
        "num_samples": int(prof["num_samples"]) * int(prof["num_chains"]),
        "divergences": diverging,
        "r_hat_max": max(rhat.values()),
        "r_hat": rhat,
        "train_mape": float(ape[train_mask].mean()),
        "holdout_mape": float(ape[~train_mask].mean()),
        "calibration": calib_check,
    }
    return MMMResult(
        channels=list(panel.channels),
        markets=list(panel.markets),
        weeks=list(panel.weeks),
        channel_summary=channel_summary,
        market_effects=market_effects,
        diagnostics=diagnostics,
        response_params=response_params,
        national_fit=national_df,
        samples=samples,
    )
