"""Smoke test: BlackJAXNSAWSampler on a 2-D Gaussian."""

from __future__ import annotations

import jax
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior  # noqa: E402
from jimgw.samplers.blackjax.ns_aw import BlackJAXNSAWSampler  # noqa: E402
from jimgw.samplers.config import BlackJAXNSAWConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Toy problem: 2-D unit-cube Gaussian centred at (0.5, 0.5), sigma = 0.05.
# ---------------------------------------------------------------------------

_SIGMA = 0.05
_MU = 0.5


class _GaussianLikelihood:
    """Tight 2-D Gaussian, analytic log Z ≈ log(2π σ²) over unit square."""

    def evaluate(self, params: dict) -> float:  # type: ignore[override]
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - _MU) ** 2 + (y - _MU) ** 2) / _SIGMA**2


def _make_sampler(n_live: int = 200) -> BlackJAXNSAWSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXNSAWConfig(
        n_live=n_live,
        n_delete_frac=0.5,
        n_target=10,
        max_mcmc=500,
        max_proposals=200,
        termination_dlogz=0.5,
    )
    parameter_names = prior.parameter_names  # ("x", "y")

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return likelihood.evaluate(named)

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return BlackJAXNSAWSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ns_aw_construction():
    sampler = _make_sampler()
    assert sampler.n_dims == 2


def test_ns_aw_get_samples_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_samples()


def _init_pos(n_live: int, seed: int = 99) -> "jax.Array":
    return jax.random.uniform(jax.random.key(seed), (n_live, 2))


def test_ns_aw_sample_and_get_samples():
    sampler = _make_sampler(n_live=100)
    rng_key = jax.random.key(0)
    sampler.sample(rng_key, _init_pos(100))

    result = sampler.get_samples()
    assert isinstance(result, dict)
    assert "samples" in result
    assert "log_likelihood" in result


def test_ns_aw_samples_fields():
    sampler = _make_sampler(n_live=100)
    sampler.sample(jax.random.key(1), _init_pos(100))
    result = sampler.get_samples()

    # Compressed samples are (n, n_dims) in sampling space
    assert isinstance(result["samples"], np.ndarray)
    assert result["samples"].ndim == 2
    assert result["samples"].shape[1] == 2
    n = result["samples"].shape[0]
    assert n > 0
    assert result["log_likelihood"].shape == (n,)


def test_ns_aw_samples_in_prior_support():
    sampler = _make_sampler(n_live=100)
    sampler.sample(jax.random.key(2), _init_pos(100))
    result = sampler.get_samples()

    # Sampling space = prior space (no sample_transforms), so [0,1]^2.
    assert np.all(result["samples"][:, 0] >= 0.0) and np.all(
        result["samples"][:, 0] <= 1.0
    )
    assert np.all(result["samples"][:, 1] >= 0.0) and np.all(
        result["samples"][:, 1] <= 1.0
    )


def test_ns_aw_diagnostics_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_diagnostics()


def test_ns_aw_diagnostics():
    sampler = _make_sampler(n_live=100)
    sampler.sample(jax.random.key(4), _init_pos(100))
    diag = sampler.get_diagnostics()

    assert isinstance(diag, dict)
    assert diag["n_likelihood_evaluations"] > 0
    assert diag["n_iterations"] > 0
    assert diag["n_accept_history"] is not None
    assert diag["n_walks_history"] is not None
    assert diag["n_proposals_history"] is not None
    assert "log_Z" in diag
    assert "log_Z_error" in diag
    assert np.isfinite(diag["log_Z"])
    assert "sampling_time" in diag
    assert diag["sampling_time"] >= 0.0


def test_ns_aw_checkpoint_file_created(tmp_path):
    """Checkpoint .pkl must be created when checkpoint_path is configured."""
    ckpt = tmp_path / "checkpoint.pkl"
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names
    config = BlackJAXNSAWConfig(
        n_live=100,
        n_delete_frac=0.5,
        n_target=10,
        max_mcmc=500,
        max_proposals=200,
        termination_dlogz=0.5,
        checkpoint_path=ckpt,
        checkpoint_interval=0.0,
    )

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = BlackJAXNSAWSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    sampler.sample(jax.random.key(42), _init_pos(100))
    assert ckpt.exists(), "Checkpoint file was not created"
