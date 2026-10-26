"""The BlackJAX SMC sampler modes on a 2-D Gaussian in the unit square."""

import logging
import pickle
import threading
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax.smc.sampler import build_blackjax_smc_sampler
from jimgw.samplers.config import (
    BlackJAXSMCConfig,
    SMCFlowConfig,
    SMCGRWConfig,
    SMCTPCNConfig,
)

_SIGMA = 0.1
_MU = 0.5
# Small kernels and flow, so that each run compiles and finishes in seconds on CPU.
_TPCN = SMCTPCNConfig(n_patience_per_dim=2, n_max_per_dim=10)
_GRW = SMCGRWConfig(n_mcmc_steps_per_dim=5)
_FLOW = SMCFlowConfig(n_layers=2, hidden_size=[16], n_epochs=5, batch_size=100)
_LADDER = [0.0, 0.1, 0.3, 0.6, 1.0]


def _config(**fields) -> BlackJAXSMCConfig:
    """A small configuration: short t-pCN without a flow unless overridden."""
    return BlackJAXSMCConfig(
        **{"n_particles": 200, "mcmc_kernel": _TPCN, "flow": False, **fields}
    )


def _make_sampler(
    config: BlackJAXSMCConfig, counter: Optional[list[int]] = None
) -> Sampler:
    """The sampler for the Gaussian; ``counter[0]`` counts likelihood calls if given."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    parameter_names = prior.parameter_names
    lock = threading.Lock()

    def count(marker):
        with lock:
            counter[0] += int(np.size(marker))

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        if counter is not None:
            jax.debug.callback(count, arr[0] * 0 + 1)
        return -0.5 * jnp.sum((arr - _MU) ** 2) / _SIGMA**2

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return build_blackjax_smc_sampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )


def _init_pos(n: int, seed: int = 99) -> jax.Array:
    return jax.random.uniform(jax.random.key(seed), (n, 2))


def _keep_checkpoint(monkeypatch, checkpoint_path: Path) -> None:
    """Stop a finished run from deleting ``checkpoint_path`` (as after a crash)."""
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


@pytest.fixture(scope="module")
def ap_sampler():
    """A finished AP run with t-pCN and a flow, the default kernel and flow."""
    sampler = _make_sampler(_config(target_ess=50, flow=_FLOW))
    sampler.sample(jax.random.key(4), _init_pos(200))
    return sampler


# ----------------------------------------------------------------------
# Construction and results
# ----------------------------------------------------------------------


def test_smc_construction():
    sampler = _make_sampler(_config())
    assert sampler.n_dims == 2


def test_smc_results_before_sample_raise():
    sampler = _make_sampler(_config())
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_samples()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_diagnostics()


@pytest.mark.slow
def test_smc_samples(ap_sampler):
    result = ap_sampler.get_samples()
    samples = result["samples"]
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2 and samples.shape[1] == 2 and samples.shape[0] > 0
    assert result["log_likelihood"].shape == (samples.shape[0],)
    assert np.all(np.isfinite(result["log_likelihood"]))
    assert np.all((samples >= 0.0) & (samples <= 1.0))
    np.testing.assert_allclose(samples.mean(0), _MU, atol=0.03)


@pytest.mark.slow
def test_smc_ap_diagnostics(ap_sampler):
    """AP mode: mutation history, the persistent log-Z trajectory and ESS."""
    diag = ap_sampler.get_diagnostics()
    n_iterations = diag["n_iterations"]

    assert n_iterations > 0
    for key in ("acceptance_history", "n_steps_history", "scale_history"):
        assert diag[key].shape == (n_iterations,)
    assert diag["training_loss_history"].shape == (n_iterations, _FLOW.n_epochs)
    assert np.all(diag["n_steps_history"] >= 1)
    assert np.all(diag["n_steps_history"] <= _TPCN.n_max_per_dim * 2)

    assert len(diag["tempering_schedule"]) == n_iterations
    assert len(diag["persistent_log_Z"]) == n_iterations
    assert float(diag["tempering_schedule"][-1]) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(diag["log_Z"])
    assert diag["sampling_time"] >= 0.0

    assert len(diag["ess_history"]) == n_iterations
    assert np.all(diag["ess_history"] > 0)
    assert np.all(np.isfinite(diag["ess_history"]))
    assert np.isfinite(diag["log_Z_error"])
    assert diag["log_Z_error"] >= 0.0


# ----------------------------------------------------------------------
# Measured likelihood count
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"target_ess": 50}, id="ap"),
        pytest.param(
            {"target_ess": 50, "mcmc_kernel": _GRW, "flow": _FLOW}, id="ap-grw-flow"
        ),
        pytest.param({"target_ess": 50, "persistent_sampling": False}, id="at"),
        pytest.param({"temperature_ladder": _LADDER}, id="fp"),
        pytest.param(
            {"temperature_ladder": _LADDER, "persistent_sampling": False}, id="ft"
        ),
    ],
)
def test_smc_likelihood_count_matches_the_calls_made(fields):
    counter = [0]
    sampler = _make_sampler(_config(n_particles=50, **fields), counter=counter)
    sampler.sample(jax.random.key(5), _init_pos(50))
    assert sampler.get_diagnostics()["n_likelihood_evaluations"] == counter[0]


# ----------------------------------------------------------------------
# Other modes
# ----------------------------------------------------------------------


def test_smc_at_diagnostics():
    """AT mode: Kish ESS history returned alongside the mutation history."""
    n_particles = 200
    sampler = _make_sampler(_config(target_ess=50, persistent_sampling=False))
    sampler.sample(jax.random.key(6), _init_pos(n_particles))
    diag = sampler.get_diagnostics()

    assert diag["n_iterations"] > 0
    assert len(diag["tempering_schedule"]) == diag["n_iterations"]
    assert len(diag["n_steps_history"]) == diag["n_iterations"]
    assert len(diag["ess_history"]) == diag["n_iterations"]
    assert np.all(diag["ess_history"] > 0)
    assert np.all(diag["ess_history"] <= n_particles)
    assert np.all(np.isfinite(diag["ess_history"]))
    assert np.isfinite(diag["log_Z_error"])
    assert diag["log_Z_error"] >= 0.0


@pytest.mark.parametrize("persistent_sampling", [True, False], ids=["fp", "ft"])
def test_smc_fixed_ladder_diagnostics(persistent_sampling):
    """FP/FT mode: one ESS value and one history entry per ladder step."""
    n_particles = 200
    sampler = _make_sampler(
        _config(temperature_ladder=_LADDER, persistent_sampling=persistent_sampling)
    )
    sampler.sample(jax.random.key(7), _init_pos(n_particles))
    diag = sampler.get_diagnostics()

    n_steps = len(_LADDER) - 1
    assert len(diag["ess_history"]) == n_steps
    assert len(diag["acceptance_history"]) == n_steps
    assert len(diag["n_steps_history"]) == n_steps
    assert np.all(diag["ess_history"] > 0)
    assert np.all(diag["ess_history"] <= n_particles * n_steps)
    assert np.all(np.isfinite(diag["ess_history"]))
    assert np.isfinite(diag["log_Z_error"])
    assert diag["log_Z_error"] >= 0.0
    if persistent_sampling:
        assert np.isfinite(diag["log_Z"])


@pytest.mark.parametrize("persistent_sampling", [True, False], ids=["ap", "at"])
def test_smc_particle_batch_size_runs(persistent_sampling):
    """batch_size > 0 evaluates the particles in batches and still samples the target."""
    sampler = _make_sampler(
        _config(target_ess=50, batch_size=20, persistent_sampling=persistent_sampling)
    )
    sampler.sample(jax.random.key(10), _init_pos(200))
    samples = sampler.get_samples()["samples"]
    assert samples.shape[0] > 0 and samples.shape[1] == 2
    assert np.all((samples >= 0.0) & (samples <= 1.0))


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
        _config(
            persistent_sampling=persistent_sampling,
            temperature_ladder=temperature_ladder,
        )
    )
    assert sampler.mode == expected_mode


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def test_smc_checkpoint_file_created(tmp_path, monkeypatch):
    """Checkpoint .pkl is written during sampling and cleaned up on success."""
    sampler = _make_sampler(
        _config(target_ess=50, checkpoint_dir=tmp_path, checkpoint_interval=1e-9)
    )
    ckpt_path = tmp_path / "checkpoint.pkl"
    _keep_checkpoint(monkeypatch, ckpt_path)
    sampler.sample(jax.random.key(42), _init_pos(200))
    monkeypatch.undo()
    assert ckpt_path.exists(), "Checkpoint was never written"
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    assert ckpt["elapsed_time"] >= 0.0
    assert ckpt["sampler_name"] == sampler.sampler_name
    assert ckpt["mode"] == sampler.mode
    assert len(ckpt["mutation_history"]["n_steps_history"]) == ckpt["n_iter"]


def test_smc_checkpoint_validation_checks_mode_and_mutation():
    sampler = _make_sampler(_config())
    checkpoint = {"sampler_name": sampler.sampler_name, **sampler._checkpoint_extra()}
    sampler._validate_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="different SMC mode"):
        sampler._validate_checkpoint({**checkpoint, "mode": "fp"})

    for other in (_config(mcmc_kernel=_GRW), _config(flow=_FLOW)):
        other_sampler = _make_sampler(other)
        with pytest.raises(ValueError, match="mcmc_kernel or flow"):
            other_sampler._validate_checkpoint(checkpoint)


@pytest.mark.slow
def test_smc_resume_gives_same_result(tmp_path, monkeypatch):
    """A run resumed from a crashed checkpoint, flow included, matches an uninterrupted run."""

    def make(checkpoint_dir=None):
        return _make_sampler(
            _config(
                target_ess=50,
                flow=_FLOW,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
            )
        )

    uninterrupted = make()
    uninterrupted.sample(jax.random.key(0), _init_pos(200))
    log_z = uninterrupted.get_diagnostics()["log_Z"]

    # Run B leaves its last checkpoint behind, as a crash would.
    ckpt_path = tmp_path / "checkpoint.pkl"
    _keep_checkpoint(monkeypatch, ckpt_path)
    make(checkpoint_dir=tmp_path).sample(jax.random.key(0), _init_pos(200))
    monkeypatch.undo()
    assert ckpt_path.exists(), "Checkpoint was never written"

    # Run C resumes from it: same RNG sequence and carried step size, same log_Z.
    resumed = make(checkpoint_dir=tmp_path)
    resumed.sample(jax.random.key(0), _init_pos(200))

    assert resumed.get_diagnostics()["log_Z"] == pytest.approx(log_z, rel=1e-6)
    assert not ckpt_path.exists(), "Checkpoint was not cleaned up"


@pytest.mark.slow
def test_smc_checkpoint_failure_restores_caller_rng_key(tmp_path):
    """A checkpoint that fails *after* its rng_key is read (mode AP, the
    default) falls back to the caller-supplied key, not the partially-loaded
    checkpoint's key. The same fallback pattern is shared verbatim across all
    four SMC modes.
    """
    caller_key = jax.random.key(7)
    reference = _make_sampler(_config(target_ess=50))
    reference.sample(caller_key, _init_pos(200))
    log_z_reference = reference.get_diagnostics()["log_Z"]

    sampler = _make_sampler(
        _config(target_ess=50, checkpoint_dir=tmp_path, checkpoint_interval=1e-9)
    )
    # Valid enough to pass `_validate_checkpoint` and overwrite `rng_key` with
    # a decoy key, but missing "n_iter" so loading fails right after.
    with open(tmp_path / "checkpoint.pkl", "wb") as f:
        pickle.dump(
            {
                "sampler_name": sampler.sampler_name,
                **sampler._checkpoint_extra(),
                "state": None,
                "rng_key": jax.random.key(999),
            },
            f,
        )

    sampler.sample(caller_key, _init_pos(200))

    assert sampler.get_diagnostics()["log_Z"] == pytest.approx(
        log_z_reference, rel=1e-6
    )


@pytest.mark.slow
def test_smc_fixed_ladder_stale_checkpoint_restarts_fresh(
    tmp_path, monkeypatch, caplog
):
    """A fixed-ladder run whose checkpoint n_iter exceeds the current ladder length
    restarts fresh rather than resuming with an out-of-range iteration count."""
    long_ladder = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    checkpoint_path = tmp_path / "checkpoint.pkl"
    _keep_checkpoint(monkeypatch, checkpoint_path)
    long_sampler = _make_sampler(
        _config(
            temperature_ladder=long_ladder,
            checkpoint_dir=tmp_path,
            checkpoint_interval=1e-9,
        )
    )
    long_sampler.sample(jax.random.key(7), _init_pos(200))
    monkeypatch.undo()
    assert checkpoint_path.exists(), "Checkpoint was never written"
    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    assert checkpoint["n_iter"] == len(long_ladder) - 1

    short_ladder = [0.0, 0.5, 1.0]
    short_sampler = _make_sampler(
        _config(
            temperature_ladder=short_ladder,
            checkpoint_dir=tmp_path,
            checkpoint_interval=1e-9,
        )
    )
    with caplog.at_level(logging.WARNING):
        short_sampler.sample(jax.random.key(7), _init_pos(200))
    assert "exceeds current schedule" in caplog.text
    assert short_sampler._n_iterations == len(short_ladder) - 1
    assert not checkpoint_path.exists(), "Checkpoint was not cleaned up"


def test_smc_fixed_ladder_same_length_stale_checkpoint_restarts_fresh(tmp_path, caplog):
    """A checkpoint whose ``n_iter`` exceeds the current schedule length
    restarts fresh, exercised directly against ``is_stale`` via a hand-built
    checkpoint (rather than two full sampler runs, as in
    ``test_smc_fixed_ladder_stale_checkpoint_restarts_fresh``)."""
    ladder = [0.0, 0.5, 1.0]
    sampler = _make_sampler(
        _config(
            temperature_ladder=ladder, checkpoint_dir=tmp_path, checkpoint_interval=1e-9
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
            **sampler._checkpoint_extra(
                mutation_history=sampler._new_mutation_history()
            ),
        },
        "test",
    )
    assert checkpoint_path.exists()
    with caplog.at_level(logging.WARNING):
        sampler.sample(jax.random.key(7), _init_pos(200))
    assert "exceeds current schedule" in caplog.text
    assert sampler._n_iterations == len(ladder) - 1
