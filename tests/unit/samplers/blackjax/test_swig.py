"""Tests for cache-aware Nested Slice within Gibbs."""

import pickle
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _make_sampler(
    checkpoint_dir: Optional[Path] = None,
) -> BlackJAXSwiGSampler:
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        n_live=24,
        n_delete_frac=0.25,
        termination_dlogz=1.5,
        max_steps=4,
        max_shrinkage=30,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
    )
    return BlackJAXSwiGSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
        rebuild_required_by_block={(0,): True, (1,): False},
        build_cache=_build_cache,
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


def test_swig_has_a_distinct_sampler_name():
    assert _make_sampler().sampler_name == "BlackJAX SwiG"


def test_swig_forwards_n_devices_to_internal_nss_config():
    """`n_devices` must reach the internally-built `BlackJAXNSSConfig` that
    `_sample` actually reads — a value forwarded by hand across two Pydantic
    models is easy to silently drop.
    """
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        n_live=24,
        n_delete_frac=0.25,
        termination_dlogz=1.5,
        n_devices=2,
    )
    sampler = BlackJAXSwiGSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
        rebuild_required_by_block={(0,): True, (1,): False},
        build_cache=_build_cache,
        log_likelihood_from_cache_fn=_log_likelihood_from_cache,
    )
    assert sampler._config.n_devices == 2


def test_swig_diagnostics():
    sampler = _make_sampler()
    initial = jax.random.uniform(jax.random.key(8), (24, 2))
    sampler.sample(jax.random.key(7), initial)
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


def test_swig_checkpoint_records_sampler_name(tmp_path, monkeypatch):
    sampler = _make_sampler(checkpoint_dir=tmp_path)
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
    sampler.sample(
        jax.random.key(5),
        jax.random.uniform(jax.random.key(6), (24, 2)),
    )
    monkeypatch.setattr(Path, "unlink", original_unlink)

    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    assert checkpoint["sampler_name"] == sampler.sampler_name
    checkpoint_path.unlink()


def test_swig_falls_back_to_fresh_run_on_foreign_checkpoint(tmp_path):
    """A checkpoint written by a different sampler is treated like a corrupt
    one: SwiG logs a warning and starts fresh rather than raising.

    Unlike flowMC (which validates the checkpoint before entering its resume
    try/except and so raises), SwiG (like NSS/NS AW/SMC) validates *inside* the
    same try/except that already catches corrupt-checkpoint errors, so a foreign
    ``sampler_name`` is swallowed the same way.
    """
    sampler = _make_sampler(checkpoint_dir=tmp_path)
    ckpt_path = tmp_path / "checkpoint.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({"sampler_name": "BlackJAX NSS"}, f)

    sampler.sample(
        jax.random.key(0),
        jax.random.uniform(jax.random.key(1), (24, 2)),
    )
    result = sampler.get_samples()
    assert "samples" in result


def test_swig_resume_gives_same_result(tmp_path, monkeypatch):
    """A run resumed from a crashed checkpoint gives the same log_Z as an uninterrupted run."""
    initial = jax.random.uniform(jax.random.key(0), (24, 2))

    s_a = _make_sampler(checkpoint_dir=None)
    s_a.sample(jax.random.key(1), initial)
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
    s_b = _make_sampler(checkpoint_dir=tmp_path)
    s_b.sample(jax.random.key(1), initial)
    monkeypatch.setattr(Path, "unlink", _orig_unlink)
    assert ckpt_path.exists(), "Checkpoint was never written"

    # Run C: resumes from B's checkpoint → same log_Z. Deletes checkpoint on success.
    s_c = _make_sampler(checkpoint_dir=tmp_path)
    s_c.sample(jax.random.key(1), initial)

    assert s_c.get_diagnostics()["log_Z"] == pytest.approx(log_z_a, rel=1e-6)
    assert not (tmp_path / "checkpoint.pkl").exists(), "Checkpoint was not cleaned up"


def test_swig_checkpoint_failure_restores_caller_rng_key(tmp_path):
    """A checkpoint that fails *after* its rng_key is read falls back to the
    caller-supplied key, not the partially-loaded checkpoint's key.
    """
    caller_key = jax.random.key(7)
    initial = jax.random.uniform(jax.random.key(8), (24, 2))

    reference = _make_sampler(checkpoint_dir=None)
    reference.sample(caller_key, initial)
    log_z_reference = reference.get_diagnostics()["log_Z"]

    sampler = _make_sampler(checkpoint_dir=tmp_path)
    ckpt_path = tmp_path / "checkpoint.pkl"
    # Valid enough to pass `_validate_checkpoint` and overwrite `rng_key` with
    # a decoy key, but missing "n_iter" so loading fails right after.
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {
                "sampler_name": sampler.sampler_name,
                "state": None,
                "dead": None,
                "rng_key": jax.random.key(999),
            },
            f,
        )

    sampler.sample(caller_key, initial)

    assert sampler.get_diagnostics()["log_Z"] == pytest.approx(
        log_z_reference, rel=1e-6
    )
