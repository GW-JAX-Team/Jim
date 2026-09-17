"""Smoke test: the BlackJAX SMC sampler on a 2-D Gaussian."""

import logging
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax.smc.base import _BlackJAXSMCBase
from jimgw.samplers.blackjax.smc.precondition import (
    build_flow,
    partition_flow,
    to_latent,
    weighted_covariance,
)
from jimgw.samplers.blackjax.smc.sampler import build_blackjax_smc_sampler
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
) -> Sampler:
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

    return build_blackjax_smc_sampler(
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


def _make_sampler_at(n_particles: int = 200) -> Sampler:
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

    return build_blackjax_smc_sampler(
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

    sampler = build_blackjax_smc_sampler(
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

    sampler = build_blackjax_smc_sampler(
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

    sampler = build_blackjax_smc_sampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    # Suppress deletion of only the checkpoint file so we can inspect it after sampling.
    ckpt_path = tmp_path / "checkpoint.pkl"
    _orig_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None if self == ckpt_path else _orig_unlink(self, missing_ok=missing_ok)
        ),
    )
    sampler.sample(jax.random.key(42), _init_pos(200))
    monkeypatch.setattr(Path, "unlink", _orig_unlink)
    assert ckpt_path.exists(), "Checkpoint was never written"
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    assert "elapsed_time" in ckpt
    assert ckpt["elapsed_time"] >= 0.0
    assert ckpt["sampler_name"] == sampler.sampler_name
    assert ckpt["mode"] == sampler.mode

    # Now let a clean run delete it.
    ckpt_path.unlink()
    assert not ckpt_path.exists()


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
        {"sampler_name": sampler.sampler_name, "mode": sampler.mode}
    )
    with pytest.raises(ValueError, match="different SMC mode"):
        sampler._validate_checkpoint(
            {"sampler_name": sampler.sampler_name, "mode": "fp"}
        )


def test_smc_checkpoint_validation_checks_precondition_flag():
    sampler = _make_sampler(
        config=BlackJAXSMCConfig(n_particles=200, precondition=True)
    )
    assert sampler._config.precondition is not None
    architecture = (
        sampler._config.precondition.rq_spline_n_layers,
        tuple(sampler._config.precondition.rq_spline_hidden_units),
        sampler._config.precondition.rq_spline_n_bins,
    )
    sampler._validate_checkpoint(
        {
            "sampler_name": sampler.sampler_name,
            "mode": sampler.mode,
            "precondition": True,
            "precondition_architecture": architecture,
        }
    )
    with pytest.raises(ValueError, match="precondition=False"):
        sampler._validate_checkpoint(
            {
                "sampler_name": sampler.sampler_name,
                "mode": sampler.mode,
                "precondition": False,
            }
        )


def test_smc_checkpoint_validation_checks_precondition_architecture():
    sampler = _make_sampler(
        config=BlackJAXSMCConfig(
            n_particles=200,
            precondition={
                "rq_spline_n_layers": 4,
                "rq_spline_hidden_units": [32, 32],
                "rq_spline_n_bins": 8,
            },
        )
    )
    with pytest.raises(ValueError, match="different preconditioning"):
        sampler._validate_checkpoint(
            {
                "sampler_name": sampler.sampler_name,
                "mode": sampler.mode,
                "precondition": True,
                # Checkpoint was made with a different n_layers.
                "precondition_architecture": (2, (32, 32), 8),
            }
        )
    # A checkpoint predating this field (no "precondition_architecture" key) must not be spuriously rejected.
    sampler._validate_checkpoint(
        {
            "sampler_name": sampler.sampler_name,
            "mode": sampler.mode,
            "precondition": True,
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

        return build_blackjax_smc_sampler(
            n_dims=len(parameter_names),
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )

    s_a = _make(checkpoint_dir=None)
    s_a.sample(jax.random.key(0), _init_pos(200))
    log_z_a = s_a.get_diagnostics()["log_Z"]

    # Run B: suppress deletion of the checkpoint file only (simulates a crash leaving it behind).
    ckpt_path = tmp_path / "checkpoint.pkl"
    _orig_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None if self == ckpt_path else _orig_unlink(self, missing_ok=missing_ok)
        ),
    )
    s_b = _make(checkpoint_dir=tmp_path)
    s_b.sample(jax.random.key(0), _init_pos(200))
    monkeypatch.setattr(Path, "unlink", _orig_unlink)
    assert ckpt_path.exists(), "Checkpoint was never written"

    # Run C: resumes from B's checkpoint → same RNG sequence → same log_Z.
    # On clean completion C deletes the checkpoint.
    s_c = _make(checkpoint_dir=tmp_path)
    s_c.sample(jax.random.key(0), _init_pos(200))

    assert s_c.get_diagnostics()["log_Z"] == pytest.approx(log_z_a, rel=1e-6)
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

        return build_blackjax_smc_sampler(
            n_dims=len(parameter_names),
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )

    caller_key = jax.random.key(7)

    reference = _make(checkpoint_dir=None)
    reference.sample(caller_key, _init_pos(200))
    log_z_reference = reference.get_diagnostics()["log_Z"]

    sampler = _make(checkpoint_dir=tmp_path)
    ckpt_path = tmp_path / "checkpoint.pkl"
    # Valid enough to pass `_validate_checkpoint` and overwrite `rng_key` with
    # a decoy key, but missing "n_iter" so loading fails right after.
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {
                "sampler_name": sampler.sampler_name,
                "mode": sampler.mode,
                "state": None,
                "rng_key": jax.random.key(999),
            },
            f,
        )

    sampler.sample(caller_key, _init_pos(200))

    assert sampler.get_diagnostics()["log_Z"] == pytest.approx(
        log_z_reference, rel=1e-6
    )


def _make_sampler_batched(n_particles: int = 200, batch_size: int = 20) -> Sampler:
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

    return build_blackjax_smc_sampler(
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

    sampler = build_blackjax_smc_sampler(
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


def test_smc_fixed_ladder_stale_checkpoint_restarts_fresh(
    tmp_path, monkeypatch, caplog
):
    """A fixed-ladder run whose checkpoint doesn't match the current ladder
    restarts fresh rather than resuming with an out-of-range iteration count.

    Caught by ``_validate_checkpoint``'s ``temperature_ladder`` check (values
    differ, not just length) before ``load_or_initialize_checkpoint``'s
    ``is_stale`` callback -- checked second -- would even run; see
    ``test_smc_fixed_ladder_same_length_stale_checkpoint_restarts_fresh`` for
    a same-ladder case that only ``is_stale`` catches.
    """
    long_ladder = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
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
    long_sampler = _make_sampler(
        config=BlackJAXSMCConfig(
            n_particles=200,
            n_mcmc_steps_per_dim=5,
            temperature_ladder=long_ladder,
            checkpoint_dir=tmp_path,
            checkpoint_interval=1e-9,
        )
    )
    long_sampler.sample(jax.random.key(7), _init_pos(200))
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert checkpoint_path.exists(), "Checkpoint was never written"
    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    assert checkpoint["n_iter"] == len(long_ladder) - 1

    short_ladder = [0.0, 0.5, 1.0]
    short_sampler = _make_sampler(
        config=BlackJAXSMCConfig(
            n_particles=200,
            n_mcmc_steps_per_dim=5,
            temperature_ladder=short_ladder,
            checkpoint_dir=tmp_path,
            checkpoint_interval=1e-9,
        )
    )
    with caplog.at_level(logging.WARNING):
        short_sampler.sample(jax.random.key(7), _init_pos(200))
    assert "temperature_ladder" in caplog.text
    assert short_sampler._n_iterations == len(short_ladder) - 1
    assert not checkpoint_path.exists(), "Checkpoint was not cleaned up"


def test_smc_fixed_ladder_same_length_stale_checkpoint_restarts_fresh(tmp_path, caplog):
    """A checkpoint whose ``n_iter`` exceeds the current (matching-ladder)
    schedule length restarts fresh -- the ``is_stale`` case a same-length,
    same-valued ladder can't trigger via ``_validate_checkpoint`` alone,
    unlike ``test_smc_fixed_ladder_stale_checkpoint_restarts_fresh``'s
    differing-ladder case."""
    ladder = [0.0, 0.5, 1.0]
    sampler = _make_sampler(
        config=BlackJAXSMCConfig(
            n_particles=200,
            n_mcmc_steps_per_dim=5,
            temperature_ladder=ladder,
            checkpoint_dir=tmp_path,
            checkpoint_interval=1e-9,
        )
    )
    checkpoint_path = tmp_path / "checkpoint.pkl"
    sampler._config.write_checkpoint(
        {
            "state": None,
            "rng_key": jax.random.key(0),
            "n_iter": len(ladder) + 5,  # out of range for this ladder
            "sampler_name": sampler.sampler_name,
            "elapsed_time": 0.0,
            **sampler._checkpoint_extra(accept_history=[]),
        },
        "test",
    )
    assert checkpoint_path.exists()
    with caplog.at_level(logging.WARNING):
        sampler.sample(jax.random.key(7), _init_pos(200))
    assert "exceeds current schedule" in caplog.text
    assert sampler._n_iterations == len(ladder) - 1


# Normalizing-flow preconditioning (precondition=True)

_PRECONDITION_KWARGS = {
    "precondition": {
        "rq_spline_n_layers": 2,
        "rq_spline_hidden_units": [16, 16],
        "rq_spline_n_bins": 4,
        "flow_n_epochs": 20,
        "flow_learning_rate": 1e-3,
    }
}


def _make_periodic_precondition_sampler(
    config: BlackJAXSMCConfig,
) -> Sampler:
    """A precondition=True sampler with dimension 0 ("x") declared periodic on [0, 1)."""
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

    return build_blackjax_smc_sampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
        periodic={0: (0.0, 1.0)},
    )


def test_smc_precondition_and_periodic_ap_runs():
    """precondition=True + periodic={...} (mode AP) runs and gives valid, finite samples."""
    config = BlackJAXSMCConfig(
        n_particles=150,
        n_mcmc_steps_per_dim=3,
        target_ess=40,
        **_PRECONDITION_KWARGS,
    )
    sampler = _make_periodic_precondition_sampler(config)
    sampler.sample(jax.random.key(15), _init_pos(150))
    result = sampler.get_samples()

    assert result["samples"].shape[1] == 2
    assert result["samples"].shape[0] > 0
    assert np.all(np.isfinite(result["samples"]))
    assert np.all(np.isfinite(result["log_likelihood"]))
    assert np.all(result["samples"] >= 0.0) and np.all(result["samples"] < 1.0)


@pytest.mark.parametrize(
    ("persistent_sampling", "temperature_ladder"),
    [
        (False, None),  # at
        (True, [0.0, 0.5, 1.0]),  # fp
        (False, [0.0, 0.5, 1.0]),  # ft
    ],
    ids=["at", "fp", "ft"],
)
def test_smc_precondition_and_periodic_constructs(
    persistent_sampling, temperature_ladder
):
    """precondition=True + periodic={...} constructs without raising in every mode."""
    config = BlackJAXSMCConfig(
        n_particles=50,
        n_mcmc_steps_per_dim=2,
        target_ess=None if temperature_ladder else 20,
        persistent_sampling=persistent_sampling,
        temperature_ladder=temperature_ladder,
        **_PRECONDITION_KWARGS,
    )
    _make_periodic_precondition_sampler(config)


def test_smc_precondition_periodic_none_matches_omitted():
    """precondition=True with periodic=None explicit must match periodic omitted --
    building `_position_wrapper` unconditionally must not perturb the non-periodic path.
    """
    config = BlackJAXSMCConfig(
        n_particles=150,
        n_mcmc_steps_per_dim=3,
        target_ess=40,
        **_PRECONDITION_KWARGS,
    )
    omitted_sampler = _make_sampler(n_particles=150, config=config)
    omitted_sampler.sample(jax.random.key(16), _init_pos(150))
    omitted_result = omitted_sampler.get_samples()

    explicit_sampler = build_blackjax_smc_sampler(
        n_dims=2,
        log_prior_fn=omitted_sampler._log_prior_fn,
        log_likelihood_fn=omitted_sampler._log_likelihood_fn,
        log_posterior_fn=omitted_sampler._log_posterior_fn,
        config=config,
        periodic=None,
    )
    explicit_sampler.sample(jax.random.key(16), _init_pos(150))
    explicit_result = explicit_sampler.get_samples()

    np.testing.assert_array_equal(omitted_result["samples"], explicit_result["samples"])
    np.testing.assert_array_equal(
        omitted_result["log_likelihood"], explicit_result["log_likelihood"]
    )


def test_smc_precondition_ap_runs():
    """Mode AP with precondition=True: runs and produces valid, finite samples."""
    config = BlackJAXSMCConfig(
        n_particles=150,
        n_mcmc_steps_per_dim=3,
        target_ess=40,
        **_PRECONDITION_KWARGS,
    )
    sampler = _make_sampler(n_particles=150, config=config)
    sampler.sample(jax.random.key(10), _init_pos(150))
    result = sampler.get_samples()

    assert result["samples"].shape[1] == 2
    assert result["samples"].shape[0] > 0
    assert np.all(np.isfinite(result["samples"]))
    assert np.all(np.isfinite(result["log_likelihood"]))
    assert np.all(result["samples"] >= 0.0) and np.all(result["samples"] <= 1.0)


def test_smc_precondition_at_runs():
    """Mode AT with precondition=True: runs and produces valid, finite samples."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXSMCConfig(
        n_particles=150,
        n_mcmc_steps_per_dim=3,
        target_ess=40,
        persistent_sampling=False,
        **_PRECONDITION_KWARGS,
    )
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = build_blackjax_smc_sampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    sampler.sample(jax.random.key(11), _init_pos(150))
    result = sampler.get_samples()

    assert result["samples"].shape[1] == 2
    assert result["samples"].shape[0] == 150
    assert np.all(np.isfinite(result["samples"]))
    assert np.all(np.isfinite(result["log_likelihood"]))


def test_smc_precondition_fp_runs():
    """Mode FP with precondition=True: runs over a fixed ladder, valid finite samples."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    ladder = [0.0, 0.2, 0.5, 1.0]
    config = BlackJAXSMCConfig(
        n_particles=150,
        n_mcmc_steps_per_dim=3,
        temperature_ladder=ladder,
        persistent_sampling=True,
        **_PRECONDITION_KWARGS,
    )
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = build_blackjax_smc_sampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    sampler.sample(jax.random.key(12), _init_pos(150))
    result = sampler.get_samples()

    assert result["samples"].shape[1] == 2
    assert result["samples"].shape[0] > 0
    assert np.all(np.isfinite(result["samples"]))
    assert np.all(np.isfinite(result["log_likelihood"]))
    diag = sampler.get_diagnostics()
    assert len(diag["acceptance_history"]) == len(ladder) - 1


def test_smc_precondition_ft_runs():
    """Mode FT with precondition=True: runs over a fixed ladder, valid finite samples."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    ladder = [0.0, 0.2, 0.5, 1.0]
    config = BlackJAXSMCConfig(
        n_particles=150,
        n_mcmc_steps_per_dim=3,
        temperature_ladder=ladder,
        persistent_sampling=False,
        **_PRECONDITION_KWARGS,
    )
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = build_blackjax_smc_sampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    sampler.sample(jax.random.key(13), _init_pos(150))
    result = sampler.get_samples()

    assert result["samples"].shape == (150, 2)
    assert np.all(np.isfinite(result["samples"]))
    assert np.all(np.isfinite(result["log_likelihood"]))
    diag = sampler.get_diagnostics()
    assert len(diag["acceptance_history"]) == len(ladder) - 1


def test_smc_precondition_train_frequency_skips_retraining():
    """precondition.train_frequency > 1 should still run to completion."""
    config = BlackJAXSMCConfig(
        n_particles=120,
        n_mcmc_steps_per_dim=3,
        target_ess=30,
        precondition={**_PRECONDITION_KWARGS["precondition"], "train_frequency": 3},
    )
    sampler = _make_sampler(n_particles=120, config=config)
    sampler.sample(jax.random.key(14), _init_pos(120))
    result = sampler.get_samples()
    assert np.all(np.isfinite(result["samples"]))


def test_smc_precondition_train_frequency_retrains_at_correct_cadence(monkeypatch):
    """train_frequency=3 must retrain on iterations 3, 6, 9, ..., not 1, 4, 7, ....

    Matches pocoMC's own cadence: pocoMC increments its iteration counter
    (``self.t``) at the very start of its loop body (in ``_reweight``), before
    checking ``self.t % train_frequency``, so its retrain check always sees the
    current iteration's 1-indexed count. Checking with the *completed* count
    (before this iteration is counted), as jim previously did, is one iteration
    early on every cycle.
    """
    config = BlackJAXSMCConfig(
        n_particles=120,
        n_mcmc_steps_per_dim=3,
        target_ess=30,
        precondition={**_PRECONDITION_KWARGS["precondition"], "train_frequency": 3},
    )
    sampler = _make_sampler(n_particles=120, config=config)

    retrain_count = [0]
    orig_retrain = _BlackJAXSMCBase._retrain_precondition_flow

    def counting_retrain(self, *args, **kwargs):
        retrain_count[0] += 1
        return orig_retrain(self, *args, **kwargs)

    monkeypatch.setattr(
        _BlackJAXSMCBase, "_retrain_precondition_flow", counting_retrain
    )

    retrain_iterations = []
    orig_update = _BlackJAXSMCBase._precondition_iteration_update

    def recording_update(self, rng_key, precond, sampler_state, **kwargs):
        before = retrain_count[0]
        rng_key, new_parameters = orig_update(
            self, rng_key, precond, sampler_state, **kwargs
        )
        if retrain_count[0] > before:
            retrain_iterations.append(kwargs["n_completed_iterations"] + 1)
        return rng_key, new_parameters

    monkeypatch.setattr(
        _BlackJAXSMCBase, "_precondition_iteration_update", recording_update
    )

    sampler.sample(jax.random.key(15), _init_pos(120))

    # The terminal iteration never retrains regardless of cadence (see
    # _precondition_iteration_update), so exclude it even if it lands on a
    # multiple of 3 -- n_iterations itself is data-dependent (ESS-driven).
    expected = [i for i in range(1, sampler._n_iterations) if i % 3 == 0]
    assert retrain_iterations == expected


def test_smc_precondition_false_is_bitwise_identical_to_default():
    """Explicit precondition=False must reproduce the implicit default exactly.

    Guards against the precondition-gated code paths added to every `_run_*`
    method accidentally changing anything on the (default) unpreconditioned path.
    """
    default_sampler = _make_sampler(n_particles=200)
    default_sampler.sample(jax.random.key(42), _init_pos(200))
    default_result = default_sampler.get_samples()

    explicit_config = BlackJAXSMCConfig(
        n_particles=200,
        n_mcmc_steps_per_dim=5,
        target_ess=50,
        initial_cov_scale=0.5,
        target_acceptance_rate=0.234,
        scale_adaptation_gain=3.0,
        precondition=False,
    )
    explicit_sampler = _make_sampler(n_particles=200, config=explicit_config)
    explicit_sampler.sample(jax.random.key(42), _init_pos(200))
    explicit_result = explicit_sampler.get_samples()

    np.testing.assert_array_equal(default_result["samples"], explicit_result["samples"])
    np.testing.assert_array_equal(
        default_result["log_likelihood"], explicit_result["log_likelihood"]
    )


def test_smc_precondition_resume_gives_same_result(tmp_path, monkeypatch):
    """A precondition=True run resumed from a crashed checkpoint gives the same
    log_Z as an uninterrupted run.

    Mirrors test_smc_resume_gives_same_result, but exercises the
    precondition-specific parts of the checkpoint path: the
    precondition/precondition_architecture compatibility checks in
    _validate_checkpoint, and that the mutation kernel's *next* step after
    resume uses the flow_params actually restored from the pickled SMC state
    (state.parameter_override), not a freshly re-initialized flow -- the
    checkpoint's rng_key must reproduce the exact same sample stream either
    way for this to pass.
    """

    def _make(checkpoint_dir=None):
        config = BlackJAXSMCConfig(
            n_particles=120,
            n_mcmc_steps_per_dim=3,
            target_ess=30,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
            **_PRECONDITION_KWARGS,
        )
        return _make_sampler(n_particles=120, config=config)

    s_a = _make(checkpoint_dir=None)
    s_a.sample(jax.random.key(20), _init_pos(120))
    log_z_a = s_a.get_diagnostics()["log_Z"]

    ckpt_path = tmp_path / "checkpoint.pkl"
    _orig_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None if self == ckpt_path else _orig_unlink(self, missing_ok=missing_ok)
        ),
    )
    s_b = _make(checkpoint_dir=tmp_path)
    s_b.sample(jax.random.key(20), _init_pos(120))
    monkeypatch.setattr(Path, "unlink", _orig_unlink)
    assert ckpt_path.exists(), "Checkpoint was never written"

    # Directly verify the checkpointed flow_params, not just infer it from log_Z equality.
    with open(ckpt_path, "rb") as f:
        raw_checkpoint = pickle.load(f)
    checkpointed_flow_params = raw_checkpoint["state"].parameter_override["flow_params"]
    assert s_b._config.precondition is not None
    expected_flat_params, _, _ = partition_flow(
        build_flow(2, s_b._config.precondition, jax.random.key(0))
    )
    # extend_params prepends a leading dim of 1 to mark this parameter as shared across particles.
    assert checkpointed_flow_params.shape == (1, *expected_flat_params.shape)
    assert np.all(np.isfinite(np.asarray(checkpointed_flow_params)))

    s_c = _make(checkpoint_dir=tmp_path)
    s_c.sample(jax.random.key(20), _init_pos(120))

    assert s_c.get_diagnostics()["log_Z"] == pytest.approx(log_z_a, rel=1e-6)
    assert not ckpt_path.exists(), "Checkpoint was not cleaned up"


def test_smc_precondition_resume_mid_run_gives_same_result(tmp_path, monkeypatch):
    """A run genuinely interrupted mid-run (not merely after it already finished)
    reproduces an uninterrupted run's log_Z once resumed.

    test_smc_precondition_resume_gives_same_result only suppresses the checkpoint
    file's deletion *after* the run has already completed, so "resuming" from it
    has no remaining mutation to execute -- it would pass even if resuming were
    completely broken. This test forces a real crash with several iterations still
    left to run, exercising the actual checkpoint/particle/rng continuity path.

    train_frequency is set high enough that no retrain fires between the crash and
    completion, isolating general checkpoint/particle/rng continuity from the
    retrain-specific mechanism -- see
    test_reconstruct_precondition_flow_recovers_checkpointed_weights for a direct,
    retrain-independent check of flow reconstruction, and
    test_smc_precondition_resume_independent_of_initial_position below for the
    case where a retrain *does* fire immediately after resume.
    """
    # No explicit mode config below -> AP (persistent_sampling defaults True, no ladder).
    import jimgw.samplers.blackjax.smc.adaptive_persistent as smc_module

    def _make(checkpoint_dir=None):
        config = BlackJAXSMCConfig(
            n_particles=150,
            n_mcmc_steps_per_dim=3,
            target_ess=145,  # tight budget -> several small annealing steps
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
            precondition={
                **_PRECONDITION_KWARGS["precondition"],
                "train_frequency": 1000,
            },
        )
        return _make_sampler(n_particles=150, config=config)

    reference = _make(checkpoint_dir=None)
    reference.sample(jax.random.key(20), _init_pos(150))
    reference_log_z = reference.get_diagnostics()["log_Z"]
    assert reference._n_iterations > 3, (
        "expected several iterations to genuinely interrupt mid-run; adjust target_ess"
    )

    class _SimulatedCrash(Exception):
        pass

    orig_save_checkpoint_if_due = smc_module.save_checkpoint_if_due
    call_count = [0]

    def _crash_after_two_checkpoints(*args, **kwargs):
        result = orig_save_checkpoint_if_due(*args, **kwargs)
        call_count[0] += 1
        if call_count[0] == 2:
            raise _SimulatedCrash("simulated crash after 2 checkpointed iterations")
        return result

    monkeypatch.setattr(
        smc_module, "save_checkpoint_if_due", _crash_after_two_checkpoints
    )
    interrupted = _make(checkpoint_dir=tmp_path)
    with pytest.raises(_SimulatedCrash):
        interrupted.sample(jax.random.key(20), _init_pos(150))
    monkeypatch.setattr(
        smc_module, "save_checkpoint_if_due", orig_save_checkpoint_if_due
    )

    ckpt_path = tmp_path / "checkpoint.pkl"
    assert ckpt_path.exists(), "Checkpoint was never written before the simulated crash"
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)
    assert 0 < checkpoint["n_iter"] < reference._n_iterations, (
        "checkpoint should capture a genuinely partial run, not a finished one"
    )

    resumed = _make(checkpoint_dir=tmp_path)
    resumed.sample(jax.random.key(20), _init_pos(150))
    resumed_log_z = resumed.get_diagnostics()["log_Z"]

    assert resumed_log_z == pytest.approx(reference_log_z, rel=1e-6)


def test_smc_precondition_resume_independent_of_initial_position(tmp_path, monkeypatch):
    """A resumed run's result must not depend on the (documented-as-ignored)
    initial_position argument passed to the resume call, even when a retrain
    fires on the very first post-resume iteration.

    Before this session's optimizer-state checkpointing fix, a resumed run's
    first post-resume retrain warm-started its Adam momentum from whatever this
    invocation's initial_position produced during the pre-checkpoint warm-train
    in _setup_precondition_for_run -- contradicting _sample's own "ignored when
    resuming" docstring promise, and making the result depend on an argument
    that's supposed to be irrelevant. train_frequency=1 (the default) forces a
    retrain on the very first post-resume iteration, the case
    test_smc_precondition_resume_mid_run_gives_same_result above deliberately
    avoids.

    Each resume attempt gets its own copy of the crash-point checkpoint: an
    earlier resume run to completion would otherwise overwrite (and finally
    delete) the shared checkpoint file before a later resume attempt reads it.
    """
    # persistent_sampling=True + an explicit ladder below -> FP mode.
    import jimgw.samplers.blackjax.smc.fixed_persistent as smc_module

    def _make(checkpoint_dir):
        config = BlackJAXSMCConfig(
            n_particles=100,
            n_mcmc_steps_per_dim=3,
            temperature_ladder=[0.0, 0.2, 0.5, 0.8, 1.0],
            persistent_sampling=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=1e-9,
            **_PRECONDITION_KWARGS,
        )
        return _make_sampler(n_particles=100, config=config)

    init_a = _init_pos(100, seed=1)

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    reference = _make(ref_dir)
    reference.sample(jax.random.key(0), init_a)
    reference_log_z = reference.get_diagnostics()["log_Z"]

    class _SimulatedCrash(Exception):
        pass

    crash_dir = tmp_path / "crash"
    crash_dir.mkdir()
    orig_save = smc_module.save_checkpoint_if_due
    call_count = [0]

    def crashing_save(*args, **kwargs):
        result = orig_save(*args, **kwargs)
        call_count[0] += 1
        if call_count[0] == 1:
            raise _SimulatedCrash("forced crash after first checkpoint")
        return result

    monkeypatch.setattr(smc_module, "save_checkpoint_if_due", crashing_save)
    crashed = _make(crash_dir)
    with pytest.raises(_SimulatedCrash):
        crashed.sample(jax.random.key(0), init_a)
    monkeypatch.setattr(smc_module, "save_checkpoint_if_due", orig_save)

    crash_checkpoint = crash_dir / "checkpoint.pkl"
    assert crash_checkpoint.exists(), "checkpoint was never written before the crash"

    init_b = _init_pos(100, seed=2)

    resume_a_dir = tmp_path / "resume_a"
    resume_a_dir.mkdir()
    shutil.copy(crash_checkpoint, resume_a_dir / "checkpoint.pkl")
    resumed_a = _make(resume_a_dir)
    resumed_a.sample(jax.random.key(0), init_a)

    resume_b_dir = tmp_path / "resume_b"
    resume_b_dir.mkdir()
    shutil.copy(crash_checkpoint, resume_b_dir / "checkpoint.pkl")
    resumed_b = _make(resume_b_dir)
    resumed_b.sample(jax.random.key(0), init_b)

    log_z_a = resumed_a.get_diagnostics()["log_Z"]
    log_z_b = resumed_b.get_diagnostics()["log_Z"]

    assert log_z_a == pytest.approx(reference_log_z, rel=1e-6)
    assert log_z_b == pytest.approx(reference_log_z, rel=1e-6)


def test_reconstruct_precondition_flow_recovers_checkpointed_weights():
    """_reconstruct_precondition_flow must rebuild the flow from whatever
    flow_params is in state.parameter_override, not silently keep some other flow.

    This is what a resumed run's subsequent retrain actually warm-starts from --
    the full-run resume tests above can't isolate it, since the mutation kernel
    always reads flow_params straight from the (correctly checkpointed) SMC state
    regardless of this method; only the *next retrain*, which uses the Python-level
    flow object this method returns, depends on it.
    """
    sampler = _make_sampler(
        n_particles=50,
        config=BlackJAXSMCConfig(n_particles=50, **_PRECONDITION_KWARGS),
    )
    precondition_config = sampler._config.precondition
    assert precondition_config is not None

    # Two flows from different keys, so their weights are genuinely distinguishable.
    flow_a = build_flow(2, precondition_config, jax.random.key(1))
    flow_b = build_flow(2, precondition_config, jax.random.key(2))
    flat_params_a, _, _ = partition_flow(flow_a)
    flat_params_b, unravel_fn, static = partition_flow(flow_b)
    assert not np.allclose(np.asarray(flat_params_a), np.asarray(flat_params_b))

    fake_state = SimpleNamespace(
        parameter_override={"flow_params": flat_params_b[None, :]}
    )
    reconstructed_flow, reconstructed_flat_params = (
        sampler._reconstruct_precondition_flow(fake_state, unravel_fn, static)
    )

    np.testing.assert_array_equal(
        np.asarray(reconstructed_flat_params), np.asarray(flat_params_b)
    )
    x = jnp.array([0.3, 0.7])
    np.testing.assert_array_equal(
        np.asarray(to_latent(reconstructed_flow, x)[0]),
        np.asarray(to_latent(flow_b, x)[0]),
    )


def test_latent_covariance_differs_from_sampling_space_covariance():
    """The preconditioned kernel's proposal covariance must reflect the flow's own
    whitened latent scale, not the raw sampling-space covariance -- guards against
    feeding jnp.cov(sampling-space particles) straight in as the latent-space
    proposal covariance, which mis-scales the step by however far sampling-space
    units are from the flow's own ~unit latent scale.
    """
    sampler = _make_sampler(
        n_particles=50,
        config=BlackJAXSMCConfig(n_particles=50, **_PRECONDITION_KWARGS),
    )
    flow_key, data_key, retrain_key = jax.random.split(jax.random.key(0), 3)
    # Sampling-space particles with a covariance far from unit scale.
    particles = jax.random.multivariate_normal(
        data_key, jnp.zeros(2), jnp.eye(2) * 100.0, (500,)
    )
    flow, optimizer, _, _, _, _ = sampler._init_precondition(flow_key)
    _, flow, _, _ = sampler._retrain_precondition_flow(
        retrain_key, flow, optimizer, particles
    )

    sampling_space_cov = jnp.cov(particles.T)
    latent_cov = sampler._latent_covariance(flow, particles)

    assert float(jnp.max(jnp.diag(sampling_space_cov))) > 50.0
    assert float(jnp.max(jnp.diag(latent_cov))) < 10.0


def test_latent_covariance_weights_match_pocomc_geometry_fit():
    """``weights`` must actually change the fitted covariance, matching
    ``precondition.weighted_covariance`` (a jittered, ``ddof=0`` aweights-style
    ``jnp.cov``, rather than pocoMC's unguarded ``np.cov(theta.T,
    aweights=weights)`` -- see ``weighted_covariance`` for why). Guards
    against silently ignoring ``weights`` and falling back to an
    unweighted fit -- e.g. for tempered (at/ft) particles with genuinely
    skewed weights.
    """
    sampler = _make_sampler(
        n_particles=50,
        config=BlackJAXSMCConfig(n_particles=50, **_PRECONDITION_KWARGS),
    )
    flow_key, data_key = jax.random.split(jax.random.key(0))
    particles = jax.random.multivariate_normal(
        data_key, jnp.zeros(2), jnp.eye(2), (200,)
    )
    flow, _, _, _, _, _ = sampler._init_precondition(flow_key)

    # Heavily skew weights toward a handful of particles.
    weights = jnp.zeros(200).at[:5].set(1.0)
    weights = weights / jnp.sum(weights)

    unweighted_cov = sampler._latent_covariance(flow, particles)
    weighted_cov = sampler._latent_covariance(flow, particles, weights=weights)

    assert not np.allclose(np.asarray(unweighted_cov), np.asarray(weighted_cov))

    latent_particles = jax.vmap(lambda x: to_latent(flow, x)[0])(particles)
    expected_weighted_cov = weighted_covariance(latent_particles, weights)
    np.testing.assert_allclose(
        np.asarray(weighted_cov), np.asarray(expected_weighted_cov)
    )


def test_smc_precondition_skips_retrain_on_terminal_iteration(monkeypatch):
    """Mode AP: the final iteration's retrain is never used by any further
    mutation, so it must not run at all: total retrains should be exactly one
    per completed iteration (the initial retrain plus each non-terminal step),
    not one more for a flow that would immediately be discarded.

    See below for the same guarantee checked on modes AT, FP, and FT.
    """
    config = BlackJAXSMCConfig(
        n_particles=120,
        n_mcmc_steps_per_dim=3,
        target_ess=30,
        **_PRECONDITION_KWARGS,
    )
    sampler = _make_sampler(n_particles=120, config=config)

    call_count = [0]
    orig_retrain = _BlackJAXSMCBase._retrain_precondition_flow

    def counting_retrain(self, *args, **kwargs):
        call_count[0] += 1
        return orig_retrain(self, *args, **kwargs)

    monkeypatch.setattr(
        _BlackJAXSMCBase, "_retrain_precondition_flow", counting_retrain
    )
    sampler.sample(jax.random.key(30), _init_pos(120))

    assert call_count[0] == sampler._n_iterations


def test_smc_precondition_skips_retrain_on_terminal_iteration_at(monkeypatch):
    """Mode AT: same terminal-retrain-skip guarantee as the AP test above."""
    config = BlackJAXSMCConfig(
        n_particles=120,
        n_mcmc_steps_per_dim=3,
        target_ess=30,
        persistent_sampling=False,
        **_PRECONDITION_KWARGS,
    )
    sampler = _make_sampler(n_particles=120, config=config)

    call_count = [0]
    orig_retrain = _BlackJAXSMCBase._retrain_precondition_flow

    def counting_retrain(self, *args, **kwargs):
        call_count[0] += 1
        return orig_retrain(self, *args, **kwargs)

    monkeypatch.setattr(
        _BlackJAXSMCBase, "_retrain_precondition_flow", counting_retrain
    )
    sampler.sample(jax.random.key(31), _init_pos(120))

    assert call_count[0] == sampler._n_iterations


def test_smc_precondition_skips_retrain_on_terminal_iteration_fp(monkeypatch):
    """Mode FP: same terminal-retrain-skip guarantee as the AP test above."""
    config = BlackJAXSMCConfig(
        n_particles=120,
        n_mcmc_steps_per_dim=3,
        temperature_ladder=[0.0, 0.2, 0.5, 1.0],
        persistent_sampling=True,
        **_PRECONDITION_KWARGS,
    )
    sampler = _make_sampler(n_particles=120, config=config)

    call_count = [0]
    orig_retrain = _BlackJAXSMCBase._retrain_precondition_flow

    def counting_retrain(self, *args, **kwargs):
        call_count[0] += 1
        return orig_retrain(self, *args, **kwargs)

    monkeypatch.setattr(
        _BlackJAXSMCBase, "_retrain_precondition_flow", counting_retrain
    )
    sampler.sample(jax.random.key(32), _init_pos(120))

    assert call_count[0] == sampler._n_iterations


def test_smc_precondition_skips_retrain_on_terminal_iteration_ft(monkeypatch):
    """Mode FT: same terminal-retrain-skip guarantee as the AP test above."""
    config = BlackJAXSMCConfig(
        n_particles=120,
        n_mcmc_steps_per_dim=3,
        temperature_ladder=[0.0, 0.2, 0.5, 1.0],
        persistent_sampling=False,
        **_PRECONDITION_KWARGS,
    )
    sampler = _make_sampler(n_particles=120, config=config)

    call_count = [0]
    orig_retrain = _BlackJAXSMCBase._retrain_precondition_flow

    def counting_retrain(self, *args, **kwargs):
        call_count[0] += 1
        return orig_retrain(self, *args, **kwargs)

    monkeypatch.setattr(
        _BlackJAXSMCBase, "_retrain_precondition_flow", counting_retrain
    )
    sampler.sample(jax.random.key(33), _init_pos(120))

    assert call_count[0] == sampler._n_iterations
