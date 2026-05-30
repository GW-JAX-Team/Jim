"""Smoke test: BlackJAXSMCSampler on a 2-D Gaussian."""

from __future__ import annotations

import jax
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior  # noqa: E402
from jimgw.samplers.blackjax.smc import BlackJAXSMCSampler  # noqa: E402
from jimgw.samplers.config import BlackJAXSMCConfig  # noqa: E402

_SIGMA = 0.1
_MU = 0.5


class _GaussianLikelihood:
    def evaluate(self, params: dict) -> float:  # type: ignore[override]
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - _MU) ** 2 + (y - _MU) ** 2) / _SIGMA**2


def _make_sampler(n_particles: int = 200) -> BlackJAXSMCSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXSMCConfig(
        n_particles=n_particles,
        n_mcmc_steps_per_dim=5,
        target_ess=50,
        initial_cov_scale=0.5,
        target_acceptance_rate=0.234,
        scale_adaptation_gain=3.0,
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

    return BlackJAXSMCSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )


def test_smc_construction():
    sampler = _make_sampler()
    assert sampler.n_dims == 2


def test_smc_get_samples_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_samples()


def _init_pos(n: int, seed: int = 99) -> "jax.Array":
    return jax.random.uniform(jax.random.key(seed), (n, 2))


def test_smc_sample_and_get_samples():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(0), _init_pos(200))
    result = sampler.get_samples()
    assert isinstance(result, dict)
    assert "samples" in result
    assert "log_likelihood" in result


def test_smc_samples_fields():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(1), _init_pos(200))
    result = sampler.get_samples()

    assert isinstance(result["samples"], np.ndarray)
    assert result["samples"].ndim == 2
    assert result["samples"].shape[1] == 2
    n = result["samples"].shape[0]
    assert n > 0
    assert result["log_likelihood"].shape == (n,)


def test_smc_samples_in_prior_support():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(2), _init_pos(200))
    result = sampler.get_samples()

    assert np.all(result["samples"][:, 0] >= 0.0) and np.all(
        result["samples"][:, 0] <= 1.0
    )
    assert np.all(result["samples"][:, 1] >= 0.0) and np.all(
        result["samples"][:, 1] <= 1.0
    )


def test_smc_diagnostics_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_diagnostics()


def test_smc_ap_diagnostics():
    """AP mode: adaptive diagnostics are populated; persistent log-Z trajectory returned."""
    sampler = _make_sampler(n_particles=200)
    sampler.sample(jax.random.key(4), _init_pos(200))
    diag = sampler.get_diagnostics()

    assert isinstance(diag, dict)
    assert diag["n_likelihood_evaluations"] > 0

    # Adaptive mode fields
    assert diag["n_iterations"] > 0
    assert diag["acceptance_history"] is not None
    assert len(diag["acceptance_history"]) == diag["n_iterations"]
    assert diag["cov_scale_history"] is not None
    assert len(diag["cov_scale_history"]) == diag["n_iterations"]

    # Persistent mode fields
    assert diag["tempering_schedule"] is not None
    assert diag["persistent_log_Z"] is not None
    assert len(diag["tempering_schedule"]) == diag["n_iterations"]
    assert len(diag["persistent_log_Z"]) == diag["n_iterations"]
    assert float(diag["tempering_schedule"][-1]) == pytest.approx(1.0, abs=1e-6)
    assert "log_Z" in diag
    assert np.isfinite(diag["log_Z"])
    assert "sampling_time" in diag
    assert diag["sampling_time"] >= 0.0

    # ESS history (persistent ESS, one value per temperature step)
    assert "ess_history" in diag
    assert len(diag["ess_history"]) == diag["n_iterations"]
    assert np.all(diag["ess_history"] > 0)
    assert np.all(np.isfinite(diag["ess_history"]))

    # log_Z_error: delta-method IS weight variance estimate
    assert "log_Z_error" in diag
    assert np.isfinite(diag["log_Z_error"])
    assert diag["log_Z_error"] >= 0.0


def test_smc_n_evals_formula():
    """n_likelihood_evaluations == n_mcmc * n_iter * n_particles."""
    n_particles = 200
    n_mcmc_per_dim = 5
    n_dims = 2
    sampler = _make_sampler(n_particles=n_particles)
    sampler.sample(jax.random.key(5), _init_pos(n_particles))
    diag = sampler.get_diagnostics()

    expected = n_mcmc_per_dim * n_dims * diag["n_iterations"] * n_particles
    assert diag["n_likelihood_evaluations"] == expected


def _make_sampler_at(n_particles: int = 200) -> BlackJAXSMCSampler:
    """Non-persistent (adaptive tempered) mode."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXSMCConfig(
        n_particles=n_particles,
        n_mcmc_steps_per_dim=5,
        target_ess=50,
        persistent_sampling=False,
    )
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return likelihood.evaluate(named)

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return BlackJAXSMCSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )


def test_smc_at_diagnostics():
    """AT mode: Kish ESS history returned alongside acceptance and tempering schedule."""
    n_particles = 200
    sampler = _make_sampler_at(n_particles=n_particles)
    sampler.sample(jax.random.key(6), _init_pos(n_particles))
    diag = sampler.get_diagnostics()

    assert diag["n_iterations"] > 0
    assert "ess_history" in diag
    assert len(diag["ess_history"]) == diag["n_iterations"]
    assert np.all(diag["ess_history"] > 0)
    assert np.all(diag["ess_history"] <= n_particles)
    assert np.all(np.isfinite(diag["ess_history"]))

    assert "log_Z_error" in diag
    assert np.isfinite(diag["log_Z_error"])
    assert diag["log_Z_error"] >= 0.0


def test_smc_fp_diagnostics():
    """FP mode: persistent ESS history returned for a fixed temperature ladder."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    ladder = [0.0, 0.1, 0.3, 0.6, 1.0]
    config = BlackJAXSMCConfig(
        n_particles=200,
        n_mcmc_steps_per_dim=5,
        temperature_ladder=ladder,
        persistent_sampling=True,
    )
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return likelihood.evaluate(named)

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = BlackJAXSMCSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    sampler.sample(jax.random.key(7), _init_pos(200))
    diag = sampler.get_diagnostics()

    assert "ess_history" in diag
    assert len(diag["ess_history"]) == len(ladder) - 1
    assert np.all(diag["ess_history"] > 0)
    assert np.all(np.isfinite(diag["ess_history"]))

    assert "log_Z_error" in diag
    assert np.isfinite(diag["log_Z_error"])
    assert diag["log_Z_error"] >= 0.0


def test_smc_checkpoint_file_created(tmp_path):
    """Checkpoint .pkl must be created when checkpoint_dir is configured."""

    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names
    config = BlackJAXSMCConfig(
        n_particles=200,
        n_mcmc_steps_per_dim=5,
        target_ess=50,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1e-9,
    )

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = BlackJAXSMCSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    sampler.sample(jax.random.key(42), _init_pos(200))
    assert (tmp_path / "checkpoint.pkl").exists(), "Checkpoint file was not created"


def test_smc_resume_gives_same_result(tmp_path):
    """Resumed SMC run gives identical log_Z to an uninterrupted run."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names

    def _make(checkpoint_dir=None):
        config = BlackJAXSMCConfig(
            n_particles=200,
            n_mcmc_steps_per_dim=5,
            target_ess=50,
            initial_cov_scale=0.5,
            target_acceptance_rate=0.234,
            scale_adaptation_gain=3.0,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
        )

        def log_prior_fn(arr):
            return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

        def log_likelihood_fn(arr):
            return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

        def log_posterior_fn(arr):
            return log_prior_fn(arr) + log_likelihood_fn(arr)

        return BlackJAXSMCSampler(
            n_dims=len(parameter_names),
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )

    s_a = _make(checkpoint_dir=None)
    s_a.sample(jax.random.key(0), _init_pos(200))
    log_z_a = s_a.get_diagnostics()["log_Z"]

    s_b = _make(checkpoint_dir=tmp_path)
    s_b.sample(jax.random.key(0), _init_pos(200))

    s_c = _make(checkpoint_dir=tmp_path)
    s_c.sample(jax.random.key(0), _init_pos(200))
    log_z_c = s_c.get_diagnostics()["log_Z"]

    assert log_z_a == pytest.approx(log_z_c, rel=1e-6)


def _make_sampler_batched(
    n_particles: int = 200, batch_size: int = 20
) -> BlackJAXSMCSampler:
    """AP mode sampler with particle_batch_size > 0."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXSMCConfig(
        n_particles=n_particles,
        n_mcmc_steps_per_dim=5,
        target_ess=50,
        initial_cov_scale=0.5,
        target_acceptance_rate=0.234,
        scale_adaptation_gain=3.0,
        batch_size=batch_size,
    )
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return likelihood.evaluate(named)

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return BlackJAXSMCSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )


def test_smc_particle_batch_size_runs():
    """particle_batch_size > 0 (AP mode) should run and produce valid samples."""
    sampler = _make_sampler_batched(n_particles=200, batch_size=20)
    sampler.sample(jax.random.key(10), _init_pos(200))
    result = sampler.get_samples()

    assert isinstance(result, dict)
    assert "samples" in result
    assert result["samples"].ndim == 2
    assert result["samples"].shape[1] == 2
    assert result["samples"].shape[0] > 0
    # Samples must lie within the prior support [0, 1]^2
    assert np.all(result["samples"] >= 0.0) and np.all(result["samples"] <= 1.0)


def test_smc_particle_batch_size_at_mode():
    """particle_batch_size > 0 in non-persistent (AT) mode should run correctly."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXSMCConfig(
        n_particles=200,
        n_mcmc_steps_per_dim=5,
        target_ess=50,
        persistent_sampling=False,
        batch_size=20,
    )
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return likelihood.evaluate(named)

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = BlackJAXSMCSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    sampler.sample(jax.random.key(11), _init_pos(200))
    result = sampler.get_samples()

    assert isinstance(result, dict)
    assert "samples" in result
    assert result["samples"].shape[0] > 0
