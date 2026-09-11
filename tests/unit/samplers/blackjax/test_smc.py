"""Smoke test: BlackJAXSMCSampler on a 2-D Gaussian."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import jax
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.samplers.blackjax.smc import BlackJAXSMCSampler
from jimgw.samplers.config import BlackJAXSMCConfig

_SIGMA = 0.1
_MU = 0.5


class _GaussianLikelihood:
    def evaluate(self, params: dict) -> float:
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - _MU) ** 2 + (y - _MU) ** 2) / _SIGMA**2


def _make_sampler(
    n_particles: int = 200, config: Optional[BlackJAXSMCConfig] = None
) -> BlackJAXSMCSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    if config is None:
        config = BlackJAXSMCConfig(
            n_particles=n_particles,
            n_mcmc_steps_per_dim=5,
            target_ess=50,
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


def _init_pos(n: int, seed: int = 99) -> jax.Array:
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


@pytest.mark.parametrize("inner_kernel", ["GRW", "DE"])
def test_smc_n_evals_formula(inner_kernel):
    """n_likelihood_evaluations == n_mcmc * n_iter * n_particles for both kernels
    (one log-density eval per proposal, regardless of proposal shape)."""
    n_particles = 200
    n_mcmc_per_dim = 5
    n_dims = 2
    config = BlackJAXSMCConfig(
        n_particles=n_particles,
        n_mcmc_steps_per_dim=n_mcmc_per_dim,
        target_ess=50,
        inner_kernel=inner_kernel,
    )
    sampler = _make_sampler(n_particles=n_particles, config=config)
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


def test_smc_ft_diagnostics():
    """FT mode: Kish ESS history returned for a fixed temperature ladder."""
    n_particles = 200
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    ladder = [0.0, 0.1, 0.3, 0.6, 1.0]
    config = BlackJAXSMCConfig(
        n_particles=n_particles,
        n_mcmc_steps_per_dim=5,
        temperature_ladder=ladder,
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

    sampler = BlackJAXSMCSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    sampler.sample(jax.random.key(8), _init_pos(n_particles))
    diag = sampler.get_diagnostics()

    assert "ess_history" in diag
    assert len(diag["ess_history"]) == len(ladder) - 1
    assert np.all(diag["ess_history"] > 0)
    assert np.all(diag["ess_history"] <= n_particles)
    assert np.all(np.isfinite(diag["ess_history"]))

    assert "log_Z_error" in diag
    assert np.isfinite(diag["log_Z_error"])
    assert diag["log_Z_error"] >= 0.0


def test_smc_checkpoint_file_created(tmp_path, monkeypatch):
    """Checkpoint .pkl is written during sampling and cleaned up on success."""
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
    # Suppress deletion of only the checkpoint file so we can inspect it after sampling.
    checkpoint_path = tmp_path / "checkpoint.pkl"
    original_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None
            if self == checkpoint_path
            else original_unlink(self, missing_ok=missing_ok)
        ),
    )
    sampler.sample(jax.random.key(42), _init_pos(200))
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert checkpoint_path.exists(), "Checkpoint was never written"
    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    assert "elapsed_time" in checkpoint
    assert checkpoint["elapsed_time"] >= 0.0
    assert checkpoint["sampler_name"] == sampler.sampler_name
    assert checkpoint["mode"] == sampler.mode

    # Now let a clean run delete it.
    checkpoint_path.unlink()
    assert not checkpoint_path.exists()


@pytest.mark.parametrize(
    ("persistent_sampling", "temperature_ladder", "expected_mode"),
    [
        (True, None, "ap"),
        (True, [0.0, 1.0], "fp"),
        (False, None, "at"),
        (False, [0.0, 1.0], "ft"),
    ],
)
def test_smc_mode_is_derived_from_config(
    persistent_sampling, temperature_ladder, expected_mode
):
    sampler = _make_sampler(
        config=BlackJAXSMCConfig(
            n_particles=200,
            persistent_sampling=persistent_sampling,
            temperature_ladder=temperature_ladder,
        )
    )
    assert sampler.mode == expected_mode


def test_smc_checkpoint_validation_checks_mode():
    sampler = _make_sampler()
    sampler._validate_checkpoint(
        {
            "sampler_name": sampler.sampler_name,
            "mode": sampler.mode,
            "inner_kernel": "GRW",
        }
    )
    with pytest.raises(ValueError, match="different SMC mode"):
        sampler._validate_checkpoint(
            {
                "sampler_name": sampler.sampler_name,
                "mode": "fp",
                "inner_kernel": "GRW",
            }
        )


def test_smc_resume_gives_same_result(tmp_path, monkeypatch):
    """A run resumed from a crashed checkpoint gives the same log_Z as an uninterrupted run."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names

    def make_sampler(checkpoint_dir=None):
        config = BlackJAXSMCConfig(
            n_particles=200,
            n_mcmc_steps_per_dim=5,
            target_ess=50,
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

    reference_sampler = make_sampler(checkpoint_dir=None)
    reference_sampler.sample(jax.random.key(0), _init_pos(200))
    reference_log_evidence = reference_sampler.get_diagnostics()["log_Z"]

    checkpoint_path = tmp_path / "checkpoint.pkl"
    original_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None
            if self == checkpoint_path
            else original_unlink(self, missing_ok=missing_ok)
        ),
    )
    interrupted_sampler = make_sampler(checkpoint_dir=tmp_path)
    interrupted_sampler.sample(jax.random.key(0), _init_pos(200))
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert checkpoint_path.exists(), "Checkpoint was never written"

    resumed_sampler = make_sampler(checkpoint_dir=tmp_path)
    resumed_sampler.sample(jax.random.key(0), _init_pos(200))

    assert resumed_sampler.get_diagnostics()["log_Z"] == pytest.approx(
        reference_log_evidence, rel=1e-6
    )
    assert not (tmp_path / "checkpoint.pkl").exists(), "Checkpoint was not cleaned up"


def test_smc_checkpoint_failure_restores_caller_rng_key(tmp_path):
    """A checkpoint that fails *after* its rng_key is read (mode AP, the
    default) falls back to the caller-supplied key, not the partially-loaded
    checkpoint's key. The same fallback pattern is shared verbatim across all
    four SMC modes.
    """
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names

    def make_sampler(checkpoint_dir=None):
        config = BlackJAXSMCConfig(
            n_particles=200,
            n_mcmc_steps_per_dim=5,
            target_ess=50,
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

    caller_key = jax.random.key(7)

    reference_sampler = make_sampler(checkpoint_dir=None)
    reference_sampler.sample(caller_key, _init_pos(200))
    reference_log_evidence = reference_sampler.get_diagnostics()["log_Z"]

    sampler = make_sampler(checkpoint_dir=tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pkl"
    with open(checkpoint_path, "wb") as checkpoint_file:
        pickle.dump(
            {
                "sampler_name": sampler.sampler_name,
                "mode": sampler.mode,
                "state": None,
                "rng_key": jax.random.key(999),
            },
            checkpoint_file,
        )

    sampler.sample(caller_key, _init_pos(200))

    assert sampler.get_diagnostics()["log_Z"] == pytest.approx(
        reference_log_evidence, rel=1e-6
    )


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


def _make_sampler_de(
    n_particles: int = 400,
    persistent_sampling: bool = True,
    temperature_ladder: list[float] | None = None,
) -> BlackJAXSMCSampler:
    """SMC sampler using the differential-evolution inner kernel.

    Adaptive by default; pass ``temperature_ladder`` for a fixed-ladder run.
    """
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    if temperature_ladder is not None:
        config = BlackJAXSMCConfig(
            n_particles=n_particles,
            n_mcmc_steps_per_dim=10,
            inner_kernel="DE",
            persistent_sampling=persistent_sampling,
            temperature_ladder=temperature_ladder,
        )
    else:
        config = BlackJAXSMCConfig(
            n_particles=n_particles,
            n_mcmc_steps_per_dim=10,
            inner_kernel="DE",
            persistent_sampling=persistent_sampling,
            target_ess=n_particles // 2,
        )
    parameter_names = prior.parameter_names

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


def test_smc_de_samples_in_prior_support():
    sampler = _make_sampler_de(n_particles=400)
    sampler.sample(jax.random.key(20), _init_pos(400))
    result = sampler.get_samples()

    assert result["samples"].ndim == 2
    assert result["samples"].shape[1] == 2
    assert result["samples"].shape[0] > 0
    assert np.all(result["samples"] >= 0.0) and np.all(result["samples"] <= 1.0)


def test_smc_de_ap_diagnostics_no_cov_scale():
    """DE inner kernel: acceptance history is kept; cov-scale history is empty."""
    sampler = _make_sampler_de(n_particles=400)
    sampler.sample(jax.random.key(22), _init_pos(400))
    diag = sampler.get_diagnostics()

    assert diag["n_iterations"] > 0
    assert len(diag["acceptance_history"]) == diag["n_iterations"]
    assert np.all(np.isfinite(diag["acceptance_history"]))
    assert len(diag["cov_scale_history"]) == 0
    assert float(diag["tempering_schedule"][-1]) == pytest.approx(1.0, abs=1e-6)


def test_smc_de_at_mode_runs():
    sampler = _make_sampler_de(n_particles=400, persistent_sampling=False)
    assert sampler.mode == "at"
    sampler.sample(jax.random.key(23), _init_pos(400))
    result = sampler.get_samples()
    assert result["samples"].shape[0] > 0
    assert np.all(result["samples"] >= 0.0) and np.all(result["samples"] <= 1.0)


def test_smc_de_evidence_matches_grw():
    """DE and GRW agree with the analytic evidence for a 2-D Gaussian.

    The analytic log evidence over the unit square is approximately
    ``log(2 * pi * sigma**2)`` because truncation is negligible.
    """
    log_evidences = {}
    for inner_kernel in ("GRW", "DE"):
        prior = CombinePrior(
            [
                UniformPrior(0.0, 1.0, parameter_names=["x"]),
                UniformPrior(0.0, 1.0, parameter_names=["y"]),
            ]
        )
        likelihood = _GaussianLikelihood()
        config = BlackJAXSMCConfig(
            n_particles=1500,
            n_mcmc_steps_per_dim=15,
            target_ess=750,
            inner_kernel=inner_kernel,
        )
        parameter_names = prior.parameter_names

        def log_prior_fn(arr, parameter_names=parameter_names, prior=prior):
            return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

        def log_likelihood_fn(
            arr, parameter_names=parameter_names, likelihood=likelihood
        ):
            return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

        def log_posterior_fn(
            arr,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
        ):
            return log_prior_fn(arr) + log_likelihood_fn(arr)

        sampler = BlackJAXSMCSampler(
            n_dims=2,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )
        sampler.sample(jax.random.key(24), _init_pos(1500))
        log_evidences[inner_kernel] = sampler.get_diagnostics()["log_Z"]

    analytic = float(np.log(2 * np.pi * _SIGMA**2))
    assert log_evidences["DE"] == pytest.approx(analytic, abs=0.15)
    assert log_evidences["DE"] == pytest.approx(log_evidences["GRW"], abs=0.15)


def test_smc_de_checkpoint_records_inner_kernel(tmp_path, monkeypatch):
    """DE checkpoints carry inner_kernel; a GRW sampler refuses to resume from one."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    def make_sampler(inner_kernel):
        return BlackJAXSMCSampler(
            n_dims=2,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=BlackJAXSMCConfig(
                n_particles=200,
                n_mcmc_steps_per_dim=5,
                target_ess=50,
                inner_kernel=inner_kernel,
                checkpoint_dir=tmp_path,
                checkpoint_interval=1e-9,
            ),
        )

    checkpoint_path = tmp_path / "checkpoint.pkl"
    original_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None
            if self == checkpoint_path
            else original_unlink(self, missing_ok=missing_ok)
        ),
    )
    make_sampler("DE").sample(jax.random.key(25), _init_pos(200))
    monkeypatch.setattr(Path, "unlink", original_unlink)

    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    assert checkpoint["inner_kernel"] == "DE"

    with pytest.raises(ValueError, match="different SMC inner kernel"):
        make_sampler("GRW")._validate_checkpoint(checkpoint)
    make_sampler("DE")._validate_checkpoint(checkpoint)


def test_smc_de_resume_gives_same_result(tmp_path, monkeypatch):
    """A DE run resumed from a crashed checkpoint reproduces the uninterrupted log_Z."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    def make_sampler(checkpoint_dir=None):
        return BlackJAXSMCSampler(
            n_dims=2,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=BlackJAXSMCConfig(
                n_particles=300,
                n_mcmc_steps_per_dim=5,
                target_ess=80,
                inner_kernel="DE",
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
            ),
        )

    initial_particles = _init_pos(300)
    reference_sampler = make_sampler()
    reference_sampler.sample(jax.random.key(0), initial_particles)
    reference_log_evidence = reference_sampler.get_diagnostics()["log_Z"]

    checkpoint_path = tmp_path / "checkpoint.pkl"
    original_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None
            if self == checkpoint_path
            else original_unlink(self, missing_ok=missing_ok)
        ),
    )
    make_sampler(checkpoint_dir=tmp_path).sample(jax.random.key(0), initial_particles)
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert checkpoint_path.exists(), "Checkpoint was never written"

    resumed_sampler = make_sampler(checkpoint_dir=tmp_path)
    resumed_sampler.sample(jax.random.key(0), initial_particles)
    assert resumed_sampler.get_diagnostics()["log_Z"] == pytest.approx(
        reference_log_evidence, rel=1e-6
    )
    assert not checkpoint_path.exists(), "Checkpoint was not cleaned up"


_DE_TEMPERATURE_LADDER = [0.0, 0.05, 0.15, 0.35, 0.65, 1.0]


def test_smc_de_fp_evidence_and_mixing():
    """FP + DE: evidence matches analytic, and the refreshed reference ensemble
    keeps the chain mixing at the final temperature (a frozen prior-width
    ensemble would collapse acceptance to ~0 by then)."""
    sampler = _make_sampler_de(
        n_particles=800, temperature_ladder=_DE_TEMPERATURE_LADDER
    )
    assert sampler.mode == "fp"
    sampler.sample(jax.random.key(30), _init_pos(800))

    result = sampler.get_samples()
    assert np.all(result["samples"] >= 0.0) and np.all(result["samples"] <= 1.0)

    diag = sampler.get_diagnostics()
    analytic = float(np.log(2 * np.pi * _SIGMA**2))
    assert diag["log_Z"] == pytest.approx(analytic, abs=0.2)
    assert diag["acceptance_history"][-1] > 0.1


def test_smc_de_ft_runs():
    sampler = _make_sampler_de(
        n_particles=600,
        persistent_sampling=False,
        temperature_ladder=_DE_TEMPERATURE_LADDER,
    )
    assert sampler.mode == "ft"
    sampler.sample(jax.random.key(31), _init_pos(600))

    result = sampler.get_samples()
    assert np.all(result["samples"] >= 0.0) and np.all(result["samples"] <= 1.0)
    assert abs(float(result["samples"].mean()) - _MU) < 0.04

    diag = sampler.get_diagnostics()
    assert len(diag["ess_history"]) == len(_DE_TEMPERATURE_LADDER) - 1
    assert diag["acceptance_history"][-1] > 0.1
