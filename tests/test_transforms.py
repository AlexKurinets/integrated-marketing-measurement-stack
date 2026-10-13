import jax.numpy as jnp
import numpy as np

from measurement.mmm.transforms import (
    adstock_weights,
    fourier_terms,
    geometric_adstock,
    geometric_adstock_jax,
    hill_saturation,
    hill_saturation_jax,
)


def test_adstock_weights_sum_to_one_so_steady_state_equals_spend():
    w = adstock_weights(np.array([0.3, 0.8]), max_lag=8)
    assert w.shape == (8, 2)
    assert np.allclose(w.sum(axis=0), 1.0)
    x = np.ones((1, 40, 2)) * 5.0
    out = geometric_adstock(x, np.array([0.3, 0.8]), 8)
    assert np.allclose(out[0, -1, :], 5.0)


def test_zero_decay_is_identity():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(3, 20, 2))
    out = geometric_adstock(x, np.array([0.0, 0.0]), 6)
    assert np.allclose(out, x)


def test_adstock_carries_spend_forward():
    x = np.zeros((1, 10, 1))
    x[0, 0, 0] = 1.0
    out = geometric_adstock(x, np.array([0.5]), 4)
    assert out[0, 0, 0] > out[0, 1, 0] > out[0, 2, 0] > out[0, 3, 0] > 0
    assert out[0, 4, 0] == 0.0


def test_hill_is_half_at_half_saturation_and_monotone():
    x = np.linspace(0, 5, 50)
    y = hill_saturation(x, np.array(1.5), np.array(2.0))
    assert np.isclose(hill_saturation(np.array([1.5]), 1.5, 2.0)[0], 0.5)
    assert np.all(np.diff(y) >= 0)
    assert y.max() < 1.0


def test_jax_and_numpy_transforms_agree():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(4, 30, 3))
    decay = np.array([0.2, 0.5, 0.7])
    a_np = geometric_adstock(x, decay, 5)
    a_jx = np.asarray(geometric_adstock_jax(jnp.array(x), jnp.array(decay), 5))
    assert np.allclose(a_np, a_jx, atol=1e-5)
    h_np = hill_saturation(a_np, np.array([1.0, 0.8, 1.2]), np.array([1.0, 1.5, 0.8]))
    h_jx = np.asarray(hill_saturation_jax(jnp.array(a_jx), jnp.array([1.0, 0.8, 1.2]), jnp.array([1.0, 1.5, 0.8])))
    assert np.allclose(h_np, h_jx, atol=1e-5)


def test_fourier_terms_shape_and_range():
    f = fourier_terms(104, 52, 3)
    assert f.shape == (104, 6)
    assert f.min() >= -1.0 and f.max() <= 1.0
    assert np.isclose(f[0, 1], 1.0)  # first cosine term at t=0
