"""Unit tests for the SMC ensemble mutation: kernels, stopping rule and flow coordinates."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.samplers.blackjax.smc.mutation import EnsembleMutation, _TPCNKernel
from jimgw.samplers.config import SMCFlowConfig, SMCGRWConfig, SMCTPCNConfig

_FLOW = SMCFlowConfig(n_layers=2, hidden_size=[16], n_epochs=5, batch_size=100)
_TWO_PI = 2 * float(jnp.pi)

# Target: a correlated Gaussian in (x0, x1) and a wrapped Gaussian angle x2 whose
# mode straddles the ends of [0, 2 pi).
_MEAN = jnp.array([1.0, -2.0])
_COV = jnp.array([[0.04, 0.03], [0.03, 0.09]])
_ANGLE_MEAN = 0.2
_ANGLE_SIGMA = 0.4
_PERIODIC = {2: (0.0, _TWO_PI)}


def _log_target(x):
    centred = x[:2] - _MEAN
    gaussian = -0.5 * centred @ jnp.linalg.solve(_COV, centred)
    offset = jnp.mod(x[2] - _ANGLE_MEAN + jnp.pi, _TWO_PI) - jnp.pi
    return gaussian - 0.5 * (offset / _ANGLE_SIGMA) ** 2


def _target_draws(key, n):
    key_gaussian, key_angle = jax.random.split(key)
    gaussian = jax.random.multivariate_normal(key_gaussian, _MEAN, _COV, (n,))
    angle = _ANGLE_MEAN + _ANGLE_SIGMA * jax.random.normal(key_angle, (n,))
    return jnp.column_stack([gaussian, jnp.mod(angle, _TWO_PI)])


def _mutate(mutation, x0, key, parameters=None):
    """One mutation through BlackJAX's update_strategy contract."""
    if parameters is None:
        parameters = mutation.initial_parameter_value()
    shared_step = partial(mutation.read_parameters, **parameters)
    update_fn, _ = mutation.update_strategy(
        mutation.mcmc_init_fn, _log_target, shared_step, mutation.max_steps, len(x0)
    )
    keys = jax.random.split(key, len(x0))
    return jax.jit(update_fn)(keys, x0, {})


def _circular_mean(angle):
    return jnp.arctan2(jnp.mean(jnp.sin(angle)), jnp.mean(jnp.cos(angle)))


# ----------------------------------------------------------------------
# Student-t fit
# ----------------------------------------------------------------------


def test_student_t_fit_recovers_its_parameters():
    mean = jnp.array([0.5, -1.0, 2.0])
    chol = jnp.array([[1.0, 0.0, 0.0], [0.5, 0.8, 0.0], [-0.3, 0.2, 0.5]])
    nu = 5.0
    key_normal, key_chi2 = jax.random.split(jax.random.key(0))
    normal = jax.random.normal(key_normal, (20000, 3)) @ chol.T
    chi2 = 2.0 * jax.random.gamma(key_chi2, 0.5 * nu, (20000,))
    z = mean + normal * jnp.sqrt(nu / chi2)[:, None]

    fit = _TPCNKernel(SMCTPCNConfig(), 3).fit(z)

    np.testing.assert_allclose(fit["mean"], mean, atol=0.03)
    np.testing.assert_allclose(fit["chol"] @ fit["chol"].T, chol @ chol.T, atol=0.05)
    assert 3.5 < float(fit["nu"]) < 7.5
    np.testing.assert_allclose(
        fit["inverse_chol"] @ fit["chol"], jnp.eye(3), atol=1e-10
    )


# ----------------------------------------------------------------------
# Invariance and stopping
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel", [SMCTPCNConfig(), SMCGRWConfig(n_mcmc_steps_per_dim=10)]
)
@pytest.mark.parametrize("flow", [False, _FLOW], ids=["no_flow", "flow"])
def test_mutation_keeps_its_target_invariant(kernel, flow):
    # Particles already drawn from the target must still follow it afterwards.
    n = 2000
    mutation = EnsembleMutation(
        n_dims=3, periodic=_PERIODIC, kernel_config=kernel, flow_config=flow
    )
    x0 = _target_draws(jax.random.key(1), n)
    x, info = _mutate(mutation, x0, jax.random.key(2))

    assert float(info["acceptance_rate"]) > 0.05
    standard_error = jnp.sqrt(jnp.diag(_COV) / n)
    np.testing.assert_array_less(jnp.abs(x[:, :2].mean(0) - _MEAN), 5 * standard_error)
    np.testing.assert_allclose(jnp.cov(x[:, :2].T), _COV, rtol=0.15)
    angle_error = (
        jnp.mod(_circular_mean(x[:, 2]) - _ANGLE_MEAN + jnp.pi, _TWO_PI) - jnp.pi
    )
    assert abs(float(angle_error)) < 5 * _ANGLE_SIGMA / np.sqrt(n)
    assert bool(jnp.all((x[:, 2] >= 0.0) & (x[:, 2] < _TWO_PI)))


def test_tpcn_stops_at_a_plateau_or_at_the_cap():
    x0 = _target_draws(jax.random.key(3), 500)
    patient = EnsembleMutation(
        n_dims=3,
        periodic=_PERIODIC,
        kernel_config=SMCTPCNConfig(n_patience_per_dim=1, n_max_per_dim=100),
        flow_config=False,
    )
    _, info = _mutate(patient, x0, jax.random.key(4))
    summary = patient.summary(info)
    assert summary["stop_reason"] == "plateau"
    assert summary["n_steps"] < patient.max_steps

    capped = EnsembleMutation(
        n_dims=3,
        periodic=_PERIODIC,
        kernel_config=SMCTPCNConfig(n_patience_per_dim=100, n_max_per_dim=1),
        flow_config=False,
    )
    _, info = _mutate(capped, x0, jax.random.key(4))
    summary = capped.summary(info)
    assert summary["stop_reason"] == "n_max"
    assert summary["n_steps"] == capped.max_steps == 3


def test_grw_scale_follows_the_acceptance_rate():
    config = SMCGRWConfig(n_mcmc_steps_per_dim=2)
    mutation = EnsembleMutation(
        n_dims=3, periodic=_PERIODIC, kernel_config=config, flow_config=False
    )
    _, info = _mutate(
        mutation, _target_draws(jax.random.key(5), 500), jax.random.key(6)
    )
    acceptance = float(info["acceptance_rate"])
    expected = config.initial_cov_scale * np.exp(
        config.scale_adaptation_gain * (acceptance - config.target_acceptance_rate)
    )
    assert float(info["next_parameters"]["scale"]) == pytest.approx(expected)
    assert int(info["n_steps"]) == mutation.max_steps == 6


# ----------------------------------------------------------------------
# Flow coordinates
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_flow_coordinates_round_trip_and_jacobian():
    mutation = EnsembleMutation(
        n_dims=3, periodic=_PERIODIC, kernel_config=SMCTPCNConfig(), flow_config=_FLOW
    )
    x0 = _target_draws(jax.random.key(7), 400)
    cuts = mutation._cuts_fn(x0)
    u0 = jax.vmap(mutation._to_unbounded, in_axes=(0, None))(x0, cuts)
    to_latent, from_latent, training_loss = mutation._coordinates(
        jax.random.key(8), u0, cuts
    )
    z0 = jax.vmap(to_latent)(u0)
    x, log_det = jax.vmap(from_latent)(z0)

    assert float(jnp.max(jnp.abs(x - x0))) < 1e-8
    jacobian = jax.vmap(jax.jacfwd(lambda z: from_latent(z)[0]))(z0)
    np.testing.assert_allclose(
        log_det, jnp.linalg.slogdet(jacobian)[1], rtol=1e-8, atol=1e-8
    )
    assert training_loss.shape == (_FLOW.n_epochs,)


@pytest.mark.slow
def test_every_mutation_trains_a_fresh_flow():
    mutation = EnsembleMutation(
        n_dims=3, periodic=_PERIODIC, kernel_config=SMCTPCNConfig(), flow_config=_FLOW
    )
    x0 = _target_draws(jax.random.key(9), 400)
    _, first = _mutate(mutation, x0, jax.random.key(10))
    carried = first["next_parameters"]
    _, second = _mutate(mutation, x0, jax.random.key(10), carried)

    # Only the step size is carried; the flow's training does not depend on the
    # previous mutation.
    assert set(carried) == {"scale"}
    np.testing.assert_allclose(second["training_loss"], first["training_loss"])
