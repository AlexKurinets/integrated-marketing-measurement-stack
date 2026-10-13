"""NumPyro model: hierarchical Bayesian MMM with geo test calibration.

Structure
---------
* Channel-level adstock decay, Hill half-saturation and slope, shared across markets.
* A national channel coefficient with market-level coefficients partially pooled around it on the log
  scale, so small markets borrow strength from the panel.
* Market intercepts, control effects, Fourier seasonality and a linear trend.
* Completed geo tests enter as likelihood terms: the model-implied lift for the tested channel, markets and
  weeks (actual spend minus counterfactual spend) is observed with the standard error of the test readout.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from measurement.mmm.transforms import geometric_adstock_jax, hill_saturation_jax

DECAY_CONCENTRATION = 12.0


def _media_response(x, decay, half_sat, slope, beta_market, max_lag):
    ad = geometric_adstock_jax(x, decay, max_lag)
    sat = hill_saturation_jax(ad, half_sat[None, None, :], slope[None, None, :])
    return beta_market[:, None, :] * sat


def mmm_model(x, controls, fourier, t_lin, y_scale, train_mask, prior, calib, y=None):
    M, T, C = x.shape
    K = controls.shape[-1]
    F = fourier.shape[-1]
    max_lag = int(prior["max_lag"])

    decay_mean = prior["decay_mean"]
    decay = numpyro.sample(
        "decay", dist.Beta(decay_mean * DECAY_CONCENTRATION, (1.0 - decay_mean) * DECAY_CONCENTRATION)
    )
    half_sat = numpyro.sample("half_sat", dist.LogNormal(jnp.zeros(C), 0.5))
    slope = numpyro.sample("slope", dist.LogNormal(jnp.zeros(C), 0.3))
    beta_national = numpyro.sample("beta_national", dist.LogNormal(jnp.log(prior["beta_mean"]), prior["beta_sd_log"]))
    market_sd = numpyro.sample("market_sd", dist.HalfNormal(jnp.full(C, prior["market_sd"])))
    with numpyro.plate("market", M):
        z = numpyro.sample("z_market", dist.Normal(jnp.zeros(C), 1.0).to_event(1))
    beta_market = numpyro.deterministic("beta_market", beta_national[None, :] * jnp.exp(z * market_sd[None, :]))

    intercept_mu = numpyro.sample("intercept_mu", dist.Normal(prior["baseline_mean"], prior["baseline_sd"]))
    intercept_sd = numpyro.sample("intercept_sd", dist.HalfNormal(0.3))
    with numpyro.plate("market_intercept", M):
        z_int = numpyro.sample("z_intercept", dist.Normal(0.0, 1.0))
    intercept = numpyro.deterministic("intercept", intercept_mu + intercept_sd * z_int)
    gamma = numpyro.sample("gamma", dist.Normal(jnp.zeros(K), 0.3))
    season = numpyro.sample("season", dist.Normal(jnp.zeros(F), 0.2))
    trend = numpyro.sample("trend", dist.Normal(0.0, 0.1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))

    media = _media_response(x, decay, half_sat, slope, beta_market, max_lag)
    numpyro.deterministic("media_contrib", media)
    mu = (
        intercept[:, None]
        + media.sum(axis=-1)
        + controls @ gamma
        + (fourier @ season)[None, :]
        + trend * t_lin[None, :]
    )
    numpyro.deterministic("mu", mu)

    with numpyro.handlers.mask(mask=jnp.broadcast_to(train_mask[None, :], (M, T))):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    for i, c in enumerate(calib):
        ci = c["channel_idx"]
        # Response of the tested channel under counterfactual spend, computed for that channel only.
        ad_cf = geometric_adstock_jax(c["x_cf"][:, :, None], decay[ci : ci + 1], max_lag)[:, :, 0]
        sat_cf = hill_saturation_jax(ad_cf, half_sat[ci], slope[ci])
        media_cf = beta_market[:, ci][:, None] * sat_cf
        diff = (media[:, :, ci] - media_cf) * y_scale[:, None]
        implied = (diff * c["market_mask"][:, None] * c["week_mask"][None, :]).sum()
        numpyro.deterministic(f"calib_implied_{i}", implied)
        numpyro.sample(f"calib_{i}", dist.Normal(implied, c["se"]), obs=c["mean"])
