"""Smoke test: BlackJAXNSSSampler on a 2-D Gaussian."""

from __future__ import annotations

import jax
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior  # noqa: E402
from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler  # noqa: E402
from jimgw.samplers.config import BlackJAXNSSConfig  # noqa: E402

_SIGMA = 0.05
_MU = 0.5


class _GaussianLikelihood:
    def evaluate(self, params: dict, data: dict) -> float:  # type: ignore[override]
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - _MU) ** 2 + (y - _MU) ** 2) / _SIGMA**2


def _make_sampler(n_live: int = 100) -> BlackJAXNSSSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXNSSConfig(
        n_live=n_live,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=5,
        termination_dlogz=0.5,
    )
    parameter_names = prior.parameter_names  # ("x", "y")

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return likelihood.evaluate(named, {})

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return BlackJAXNSSSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )


def test_nss_construction():
    sampler = _make_sampler()
    assert sampler.n_dims == 2


def test_nss_get_samples_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_samples()


def _init_pos(n_live: int, seed: int = 99) -> "jax.Array":
    return jax.random.uniform(jax.random.key(seed), (n_live, 2))


def test_nss_sample_and_get_samples():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(0), _init_pos(100))
    result = sampler.get_samples()
    assert isinstance(result, dict)
    assert "samples" in result
    assert "log_likelihood" in result


def test_nss_samples_fields():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(1), _init_pos(100))
    result = sampler.get_samples()

    assert isinstance(result["samples"], np.ndarray)
    assert result["samples"].ndim == 2
    assert result["samples"].shape[1] == 2
    n = result["samples"].shape[0]
    assert n > 0
    assert result["log_likelihood"].shape == (n,)


def test_nss_samples_in_prior_support():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(2), _init_pos(100))
    result = sampler.get_samples()

    assert np.all(result["samples"][:, 0] >= 0.0) and np.all(
        result["samples"][:, 0] <= 1.0
    )
    assert np.all(result["samples"][:, 1] >= 0.0) and np.all(
        result["samples"][:, 1] <= 1.0
    )


def test_nss_diagnostics_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_diagnostics()


def test_nss_diagnostics():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(4), _init_pos(100))
    diag = sampler.get_diagnostics()

    assert isinstance(diag, dict)
    assert diag["n_iterations"] > 0
    assert diag["n_stepping_out_history"] is not None
    assert diag["n_shrinking_history"] is not None
    assert diag["n_likelihood_evaluations_stepping_out"] is not None
    assert diag["n_likelihood_evaluations_shrinking"] is not None
    assert diag["n_likelihood_evaluations"] == (
        diag["n_likelihood_evaluations_stepping_out"]
        + diag["n_likelihood_evaluations_shrinking"]
    )
    assert "log_Z" in diag
    assert "log_Z_error" in diag
    assert np.isfinite(diag["log_Z"])
    assert "sampling_time" in diag
    assert diag["sampling_time"] >= 0.0
