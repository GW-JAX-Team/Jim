"""Tests for the BlackJAXNSAWSampler unit-cube prior validator (B3)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.samplers.blackjax.ns_aw import BlackJAXNSAWSampler  # noqa: E402
from jimgw.samplers.config import BlackJAXNSAWConfig  # noqa: E402

_CONFIG = BlackJAXNSAWConfig(n_live=50, termination_dlogz=0.5)


def _make_sampler(log_prior_fn, log_likelihood_fn=None, n_dims=2):
    """Construct a minimal BlackJAXNSAWSampler with given callables."""
    if log_likelihood_fn is None:

        def log_likelihood_fn(arr):
            return 0.0

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return BlackJAXNSAWSampler(
        n_dims=n_dims,
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=_CONFIG,
    )


def _flat_uniform_prior(n_dims):
    """Log-prior that is flat on [0,1]^n_dims."""
    log_volume = 0.0  # log(1^n_dims)

    def _lp(arr):
        inside = jnp.all((arr >= 0.0) & (arr <= 1.0))
        return jnp.where(inside, jnp.array(log_volume), -jnp.inf)

    return _lp


def test_valid_unit_cube_prior_passes():
    """A proper flat prior on [0,1]^2 should construct without error."""
    lp = _flat_uniform_prior(2)
    sampler = _make_sampler(lp)
    assert sampler.n_dims == 2


def test_non_flat_prior_raises():
    """A Gaussian prior (not flat) should raise ValueError."""

    def gaussian_prior(arr):
        return -0.5 * jnp.sum(arr**2)

    with pytest.raises(ValueError, match=r"must return 0\.0 for all points"):
        _make_sampler(gaussian_prior)


def test_prior_with_support_outside_unit_cube_raises():
    """A prior whose support extends outside [0,1]^n_dims should raise."""

    def wide_prior(arr):
        inside = jnp.all((arr >= -1.0) & (arr <= 2.0))
        return jnp.where(inside, 0.0, -jnp.inf)

    with pytest.raises(ValueError, match="must return -inf for all points outside"):
        _make_sampler(wide_prior)


def test_prior_with_zero_mass_at_origin_raises():
    """A prior that excludes the origin (e.g. a hole) should raise."""

    def holed_prior(arr):
        norm = jnp.sum(arr**2)
        inside = (norm > 0.01) & jnp.all((arr >= 0.0) & (arr <= 1.0))
        return jnp.where(inside, 0.0, -jnp.inf)

    with pytest.raises(ValueError, match=r"must return 0\.0 for all points"):
        _make_sampler(holed_prior)
