"""Tests for cache-aware Nested Slice within Gibbs."""

import jax
import jax.numpy as jnp
import numpy as np

from jimgw.samplers.blackjax.swig import BlackJAXSwiGSampler
from jimgw.samplers.config import BlackJAXSwiGConfig


def _log_prior(position):
    return jnp.where(jnp.all((position >= 0.0) & (position <= 1.0)), 0.0, -jnp.inf)


def _build_cache(position):
    return position[0] ** 2


def _log_likelihood_from_cache(position, cache):
    return -40.0 * ((cache - 0.25) ** 2 + (position[1] - 0.5) ** 2)


def _log_likelihood(position):
    return _log_likelihood_from_cache(position, _build_cache(position))


def _make_sampler() -> BlackJAXSwiGSampler:
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        n_live=24,
        n_delete_frac=0.25,
        termination_dlogz=1.5,
        max_steps=4,
        max_shrinkage=30,
    )
    return BlackJAXSwiGSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
        block_indices=((0,), (1,)),
        refresh_cache=(True, False),
        build_cache_fn=_build_cache,
        log_likelihood_from_cache_fn=_log_likelihood_from_cache,
    )


def test_swig_cached_likelihood_remains_consistent():
    sampler = _make_sampler()
    initial = jax.random.uniform(jax.random.key(1), (24, 2))
    sampler.sample(jax.random.key(2), initial)
    result = sampler.get_samples()
    expected = jax.vmap(_log_likelihood)(jnp.asarray(result["samples"]))
    np.testing.assert_allclose(result["log_likelihood"], expected, rtol=1e-10)


def test_swig_does_not_store_cache_on_live_particles():
    sampler = _make_sampler()
    initial = jax.random.uniform(jax.random.key(3), (24, 2))
    sampler.sample(jax.random.key(4), initial)
    assert not hasattr(sampler._final_state.particles, "cache")
