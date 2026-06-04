"""Unit tests for the Sampler ABC."""

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.samplers.base import Sampler
from jimgw.samplers.config import BaseSamplerConfig


def _make_callables(n_dims: int = 1):
    """Return minimal flat-array callables for a unit-uniform prior in [0,1]^n."""

    def log_prior_fn(arr):
        # Uniform[0,1]^n: log_prob = 0 inside, -inf outside.
        return jnp.where(jnp.all((arr >= 0.0) & (arr <= 1.0)), 0.0, -jnp.inf)

    def log_likelihood_fn(arr):
        return jnp.sum(arr)

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return log_prior_fn, log_likelihood_fn, log_posterior_fn


class _TrivialSampler(Sampler):
    """Minimal concrete Sampler for ABC contract tests."""

    def _sample(self, rng_key, initial_position) -> None:  # noqa: ARG002
        self._ran = True

    def get_samples(self) -> dict[str, np.ndarray]:
        return {
            "samples": np.zeros((3, self.n_dims)),
            "log_likelihood": np.zeros(3),
        }

    def _get_diagnostics(self) -> dict[str, Any]:
        return {"n_likelihood_evaluations": 0}


# --- ABC instantiation ---


def test_sampler_is_abstract():
    lp, ll, lpost = _make_callables()
    with pytest.raises(TypeError):
        Sampler(  # type: ignore[abstract]
            n_dims=1,
            log_prior_fn=lp,
            log_likelihood_fn=ll,
            log_posterior_fn=lpost,
            config=BaseSamplerConfig(),
        )


def test_trivial_sampler_works():
    lp, ll, lpost = _make_callables(n_dims=2)
    s = _TrivialSampler(
        n_dims=2,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    assert s._prev_elapsed == 0.0
    s.sample(jax.random.key(0), jnp.zeros((3, 2)))
    result = s.get_samples()
    assert result["samples"].shape == (3, 2)
    assert result["log_likelihood"].shape == (3,)


def test_trivial_sampler_get_diagnostics_returns_dict():
    lp, ll, lpost = _make_callables(n_dims=1)
    s = _TrivialSampler(
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    s.sample(jax.random.key(0), jnp.zeros((3, 1)))
    diag = s.get_diagnostics()
    assert isinstance(diag, dict)
    assert "n_likelihood_evaluations" in diag
    assert "sampling_time" in diag
    assert diag["sampling_time"] >= 0.0


def test_get_diagnostics_before_sample_raises():
    lp, ll, lpost = _make_callables(n_dims=1)
    s = _TrivialSampler(
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    with pytest.raises(RuntimeError, match="before sample"):
        s.get_diagnostics()


def test_sampling_time_is_non_negative():
    lp, ll, lpost = _make_callables(n_dims=2)
    s = _TrivialSampler(
        n_dims=2,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    s.sample(jax.random.key(0), jnp.zeros((3, 2)))
    diag = s.get_diagnostics()
    assert diag["sampling_time"] >= 0.0

    # _prev_elapsed is accumulated into sampling_time on resume.
    s2 = _TrivialSampler(
        n_dims=2,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    s2._prev_elapsed = 5.0
    s2.sample(jax.random.key(0), jnp.zeros((3, 2)))
    assert s2.get_diagnostics()["sampling_time"] >= 5.0


# --- Callable injection ---


def test_log_prior_fn_is_called():
    lp, ll, lpost = _make_callables(n_dims=1)
    s = _TrivialSampler(
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    # Inside [0,1]: log_prior = 0
    assert float(s._log_prior_fn(jnp.array([0.5]))) == pytest.approx(0.0)
    # Outside [0,1]: log_prior = -inf
    assert not jnp.isfinite(s._log_prior_fn(jnp.array([1.5])))


def test_sampler_does_not_own_initial_position_callable():
    """Samplers must not store a sample_initial_positions_fn; Jim owns that."""
    lp, ll, lpost = _make_callables(n_dims=2)
    s = _TrivialSampler(
        n_dims=2,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    assert not hasattr(s, "_sample_initial_positions_fn")


def test_initial_position_is_required():
    """sample() must require initial_position; calling without it raises TypeError."""
    lp, ll, lpost = _make_callables(n_dims=2)
    s = _TrivialSampler(
        n_dims=2,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    sig = inspect.signature(s.sample)
    param = sig.parameters["initial_position"]
    assert param.default is inspect.Parameter.empty, (
        "initial_position should be required (no default)"
    )


def test_n_dims_stored():
    lp, ll, lpost = _make_callables(n_dims=4)
    s = _TrivialSampler(
        n_dims=4,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    assert s.n_dims == 4
