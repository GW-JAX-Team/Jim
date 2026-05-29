"""Short end-to-end smoke test for FlowMCSampler.

Uses a tiny 2D Gaussian toy problem with very few steps to verify the
sampler runs and returns well-formed dicts from get_samples() and get_diagnostics().
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import CombinePrior, UniformPrior  # type: ignore[attr-defined]
from jimgw.samplers.config import FlowMCConfig
from jimgw.samplers.flowmc import FlowMCSampler


class _GaussianLikelihood(LikelihoodBase):
    """Isotropic 2D Gaussian centred at (0.5, 0.5) within [0,1]^2."""

    _model = None
    _data = None

    def evaluate(self, params: dict) -> float:  # noqa: ARG002
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.1**2


def _make_tiny_config() -> FlowMCConfig:
    return FlowMCConfig(
        n_chains=10,
        n_local_steps=5,
        n_global_steps=5,
        global_thinning=1,
        n_training_loops=2,
        n_production_loops=2,
        n_epochs=2,
        parallel_tempering=None,
    )


def _make_sampler() -> FlowMCSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names  # ("x", "y")
    n_dims = len(parameter_names)

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return likelihood.evaluate(named)

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return FlowMCSampler(
        n_dims=n_dims,
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=_make_tiny_config(),
    )


def test_flowmc_sampler_construction():
    s = _make_sampler()
    assert s.n_dims == 2


def test_flowmc_sampler_no_tempering_strategy_order():
    s = _make_sampler()
    assert "parallel_tempering" not in s.strategy_order


@pytest.mark.slow
def test_flowmc_sampler_sample_and_get_samples():
    s = _make_sampler()
    rng_key = jax.random.key(42)
    s.sample(rng_key, jnp.ones((10, 2)) * 0.5)
    result = s.get_samples()

    assert isinstance(result, dict)
    assert "samples" in result
    assert "log_likelihood" in result
    assert isinstance(result["samples"], np.ndarray)
    assert result["samples"].ndim == 2
    assert result["samples"].shape[1] == 2
    n = result["samples"].shape[0]
    assert n > 0
    assert result["log_likelihood"].shape == (n,)


@pytest.mark.slow
def test_flowmc_sampler_samples_in_prior_range():
    s = _make_sampler()
    s.sample(jax.random.key(1), jnp.ones((10, 2)) * 0.5)
    result = s.get_samples()
    # Samples are in sampling space = prior space (no transforms for this problem).
    x = result["samples"][:, 0]
    y = result["samples"][:, 1]
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_flowmc_sampler_get_samples_before_sample_raises():
    s = _make_sampler()
    with pytest.raises(RuntimeError):
        s.get_samples()


@pytest.mark.slow
def test_flowmc_diagnostics():
    s = _make_sampler()
    s.sample(jax.random.key(2), jnp.ones((10, 2)) * 0.5)
    diag = s.get_diagnostics()

    assert isinstance(diag, dict)
    assert diag["n_likelihood_evaluations"] > 0
    assert diag["n_training_loops_actual"] is not None
    assert diag["training_loss_history"] is not None
    assert diag["acceptance_training_local"] is not None
    assert diag["acceptance_training_global"] is not None
    assert diag["acceptance_production_local"] is not None
    assert diag["acceptance_production_global"] is not None
    assert "sampling_time" in diag
    assert diag["sampling_time"] >= 0.0


@pytest.mark.slow
def test_flowmc_checkpoint_file_created(tmp_path):
    """A checkpoint .pkl file must be created when checkpoint_dir is configured."""
    config = FlowMCConfig(
        n_chains=10,
        n_local_steps=5,
        n_global_steps=5,
        global_thinning=1,
        n_training_loops=2,
        n_production_loops=1,
        n_epochs=2,
        parallel_tempering=None,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1e-9,
    )
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names
    n_dims = len(parameter_names)

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    s = FlowMCSampler(
        n_dims=n_dims,
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    s.sample(jax.random.key(42), jnp.ones((10, 2)) * 0.5)
    assert (tmp_path / "checkpoint.pkl").exists(), "Checkpoint file was not created"


@pytest.mark.slow
def test_flowmc_resume_gives_same_result(tmp_path):
    """Resumed flowMC run gives identical production samples to an uninterrupted run.

    The checkpoint stores the RNG state at the end of training; resuming skips
    training and runs production from the same RNG state → identical samples.
    """
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names
    n_dims = len(parameter_names)
    init_pos = jnp.ones((10, 2)) * 0.5

    def _make_sampler(checkpoint_dir, checkpoint_interval=1e-9):
        config = FlowMCConfig(
            n_chains=10,
            n_local_steps=5,
            n_global_steps=5,
            global_thinning=1,
            n_training_loops=2,
            n_production_loops=1,
            n_epochs=2,
            parallel_tempering=None,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
        )

        def log_prior_fn(arr):
            return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

        def log_likelihood_fn(arr):
            return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

        def log_posterior_fn(arr):
            return log_prior_fn(arr) + log_likelihood_fn(arr)

        return FlowMCSampler(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )

    # Run A: no checkpointing (interval=0 disables it regardless of checkpoint_dir)
    s_a = _make_sampler(checkpoint_dir=None, checkpoint_interval=0.0)
    s_a.sample(jax.random.key(42), init_pos)
    result_a = s_a.get_samples()

    # Run B: write checkpoint at end of each training loop
    s_b = _make_sampler(checkpoint_dir=tmp_path)
    s_b.sample(jax.random.key(42), init_pos)
    assert (tmp_path / "checkpoint.pkl").exists()

    # Run C: resume from B's checkpoint → production uses same RNG state
    s_c = _make_sampler(checkpoint_dir=tmp_path)
    s_c.sample(jax.random.key(42), init_pos)
    result_c = s_c.get_samples()

    np.testing.assert_array_equal(result_a["samples"], result_c["samples"])
