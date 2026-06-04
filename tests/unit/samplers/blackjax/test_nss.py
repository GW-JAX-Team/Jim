"""Smoke test: BlackJAXNSSSampler on a 2-D Gaussian."""

from __future__ import annotations

import jax
import numpy as np
import pickle
import pytest
from pathlib import Path

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior  # noqa: E402
from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler  # noqa: E402
from jimgw.samplers.config import BlackJAXNSSConfig  # noqa: E402

_SIGMA = 0.05
_MU = 0.5


class _GaussianLikelihood:
    def evaluate(self, params: dict) -> float:  # type: ignore[override]
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
        return likelihood.evaluate(named)

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


def test_nss_checkpoint_file_created(tmp_path, monkeypatch):
    """Checkpoint .pkl is written during sampling and cleaned up on success."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names
    config = BlackJAXNSSConfig(
        n_live=100,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=5,
        termination_dlogz=0.5,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1e-9,
    )

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = BlackJAXNSSSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    ckpt_path = tmp_path / "checkpoint.pkl"
    _orig_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None if self == ckpt_path else _orig_unlink(self, missing_ok=missing_ok)
        ),
    )
    sampler.sample(jax.random.key(42), _init_pos(100))
    monkeypatch.setattr(Path, "unlink", _orig_unlink)
    assert ckpt_path.exists(), "Checkpoint was never written"
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    assert "elapsed_time" in ckpt
    assert ckpt["elapsed_time"] >= 0.0
    ckpt_path.unlink()


def test_nss_resume_gives_same_result(tmp_path, monkeypatch):
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
        config = BlackJAXNSSConfig(
            n_live=100,
            n_delete_frac=0.5,
            num_inner_steps_per_dim=5,
            termination_dlogz=0.5,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
        )

        def log_prior_fn(arr):
            return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

        def log_likelihood_fn(arr):
            return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

        def log_posterior_fn(arr):
            return log_prior_fn(arr) + log_likelihood_fn(arr)

        return BlackJAXNSSSampler(
            n_dims=len(parameter_names),
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )

    s_a = _make(checkpoint_dir=None)
    s_a.sample(jax.random.key(0), _init_pos(100))
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
    s_b.sample(jax.random.key(0), _init_pos(100))
    monkeypatch.setattr(Path, "unlink", _orig_unlink)
    assert ckpt_path.exists(), "Checkpoint was never written"

    # Run C: resumes from B's checkpoint → same log_Z. Deletes checkpoint on success.
    s_c = _make(checkpoint_dir=tmp_path)
    s_c.sample(jax.random.key(0), _init_pos(100))

    assert s_c.get_diagnostics()["log_Z"] == pytest.approx(log_z_a, rel=1e-6)
    assert not (tmp_path / "checkpoint.pkl").exists(), "Checkpoint was not cleaned up"
