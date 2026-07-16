"""Multi-device tests for BlackJAX NSS and SwiG."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler
from jimgw.samplers.blackjax.sharding import make_live_mesh
from jimgw.samplers.blackjax.swig import BlackJAXSwiGSampler
from jimgw.samplers.config import BlackJAXNSSConfig, BlackJAXSwiGConfig

_HAS_FOUR_DEVICES = jax.local_device_count() >= 4


def _log_prior(position):
    return jnp.where(jnp.all((position >= 0.0) & (position <= 1.0)), 0.0, -jnp.inf)


def _build_cache(position):
    return position[0] ** 2


def _log_likelihood_from_cache(position, cache):
    return -20.0 * ((cache - 0.25) ** 2 + (position[1] - 0.5) ** 2)


def _log_likelihood(position):
    return _log_likelihood_from_cache(position, _build_cache(position))


def test_make_live_mesh_rejects_unavailable_devices():
    with pytest.raises(ValueError, match="only .* local JAX devices"):
        make_live_mesh(jax.local_device_count() + 1, 16, 4)


@pytest.mark.skipif(
    not _HAS_FOUR_DEVICES,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
def test_nss_runs_sharded_and_preserves_diagnostics():
    config = BlackJAXNSSConfig(
        n_live=16,
        n_delete_frac=0.25,
        num_inner_steps_per_dim=1,
        termination_dlogz=2.0,
        n_devices=4,
    )
    sampler = BlackJAXNSSSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
    )

    initial = jax.random.uniform(jax.random.key(0), (16, 2))
    sampler.sample(jax.random.key(1), initial)
    diagnostics = sampler.get_diagnostics()

    assert diagnostics["n_iterations"] > 0
    assert diagnostics["n_likelihood_evaluations"] > 0
    stepping_out = diagnostics["n_stepping_out_history"]
    assert stepping_out.shape[-1] == 2
    assert stepping_out.shape[0] % config.n_devices == 0


@pytest.mark.skipif(
    not _HAS_FOUR_DEVICES,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
def test_swig_runs_sharded_with_consistent_cache():
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        n_live=16,
        n_delete_frac=0.25,
        num_gibbs_sweeps=1,
        max_steps=3,
        max_shrinkage=20,
        termination_dlogz=2.0,
        n_devices=4,
    )
    sampler = BlackJAXSwiGSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
        block_indices=((0,), (1,)),
        refresh_cache=(True, False),
        build_cache_fn=_build_cache,
        log_likelihood_from_cache_fn=_log_likelihood_from_cache,
    )

    initial = jax.random.uniform(jax.random.key(2), (16, 2))
    sampler.sample(jax.random.key(3), initial)
    result = sampler.get_samples()
    expected = jax.vmap(_log_likelihood)(jnp.asarray(result["samples"]))

    np.testing.assert_allclose(result["log_likelihood"], expected, rtol=2e-6)
    assert sampler.get_diagnostics()["n_likelihood_evaluations"] > 0
    assert not hasattr(sampler._final_state.particles, "cache")


@pytest.mark.skipif(
    not _HAS_FOUR_DEVICES,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
def test_sharded_checkpoint_is_host_backed_and_resumable(tmp_path, monkeypatch):
    config = BlackJAXNSSConfig(
        n_live=16,
        n_delete_frac=0.25,
        num_inner_steps_per_dim=1,
        termination_dlogz=2.0,
        n_devices=4,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1e-9,
    )

    def make_sampler():
        return BlackJAXNSSSampler(
            n_dims=2,
            log_prior_fn=_log_prior,
            log_likelihood_fn=_log_likelihood,
            log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
            config=config,
        )

    checkpoint = tmp_path / "checkpoint.pkl"
    original_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None if self == checkpoint else original_unlink(self, missing_ok=missing_ok)
        ),
    )
    initial = jax.random.uniform(jax.random.key(4), (16, 2))
    make_sampler().sample(jax.random.key(5), initial)
    monkeypatch.setattr(Path, "unlink", original_unlink)

    with checkpoint.open("rb") as stream:
        saved = pickle.load(stream)
    assert isinstance(saved["state"].particles.position, np.ndarray)

    resumed = make_sampler()
    resumed.sample(jax.random.key(999), initial)
    assert resumed.get_diagnostics()["n_iterations"] > 0
    assert not checkpoint.exists()
