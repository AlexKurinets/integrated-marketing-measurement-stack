"""Media response transforms: geometric adstock and Hill saturation.

Both come in a NumPy version (used by the seed generator and the allocator) and a JAX version (used inside
the NumPyro model). The tests check that the two agree.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def adstock_weights(decay: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized geometric weights, shape (max_lag, C). Weights sum to one so steady-state adstock equals spend."""
    lags = np.arange(max_lag)[:, None]
    w = np.asarray(decay)[None, :] ** lags
    return w / w.sum(axis=0, keepdims=True)


def geometric_adstock(x: np.ndarray, decay: np.ndarray, max_lag: int) -> np.ndarray:
    """Carry-over of media over time. x has shape (M, T, C); decay has shape (C,)."""
    x = np.asarray(x, dtype=float)
    _, T, _ = x.shape
    w = adstock_weights(decay, max_lag)
    out = np.zeros_like(x)
    for lag in range(max_lag):
        shifted = np.zeros_like(x)
        shifted[:, lag:, :] = x[:, : T - lag, :]
        out += shifted * w[lag][None, None, :]
    return out


def hill_saturation(x: np.ndarray, half_sat: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Diminishing returns. Returns a value in [0, 1) that equals 0.5 when x == half_sat."""
    x = np.asarray(x, dtype=float)
    xs = np.power(np.clip(x, 0.0, None), slope)
    hs = np.power(half_sat, slope)
    return xs / (xs + hs)


def geometric_adstock_jax(x, decay, max_lag: int):
    T = x.shape[1]
    lags = jnp.arange(max_lag)[:, None]
    w = decay[None, :] ** lags
    w = w / w.sum(axis=0, keepdims=True)
    stacked = jnp.stack(
        [jnp.pad(x, ((0, 0), (lag, 0), (0, 0)))[:, :T, :] for lag in range(max_lag)],
        axis=0,
    )
    return (stacked * w[:, None, None, :]).sum(axis=0)


def hill_saturation_jax(x, half_sat, slope):
    xs = jnp.power(jnp.clip(x, 0.0, None), slope)
    hs = jnp.power(half_sat, slope)
    return xs / (xs + hs)


def fourier_terms(T: int, period: float, order: int) -> np.ndarray:
    """Seasonal basis, shape (T, 2 * order)."""
    t = np.arange(T)
    cols = []
    for k in range(1, order + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.stack(cols, axis=1)
