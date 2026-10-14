"""Unit tests for a single step of the NS AW acceptance-walk kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("blackjax")

from blackjax.ns.base import StateWithLogLikelihood

from jimgw.samplers.blackjax.ns._acceptance_walk_kernel import (
    DEKernelParams,
    _de_one_step,
)
from jimgw.samplers.periodic import to_unit_cube_stepper

_N_LIVE = 400
_NUM_SURVIVORS = _N_LIVE - 1
_LOGLIKELIHOOD_0 = jnp.array(-1.0)


def _log_prior(x):
    return jnp.where(jnp.all((x >= 0.0) & (x <= 1.0)), 0.0, -jnp.inf)


def _flat_params(key):
    """Uniform live points in the unit interval, all with log-likelihood 0."""
    live_points = jax.random.uniform(key, (_N_LIVE, 1))
    loglikelihoods = jnp.zeros(_N_LIVE)
    params = DEKernelParams(
        live_points=live_points,
        loglikelihoods=loglikelihoods,
        mix=0.5,
        scale=2.38 / jnp.sqrt(2.0),
        num_walks=jnp.array(1),
        walks_float=jnp.array(1.0),
        n_accept_total=jnp.array(0),
        n_likelihood_evals_total=jnp.array(0),
    )
    _, top_indices = jax.lax.top_k(loglikelihoods, _NUM_SURVIVORS)
    return params, top_indices


def _step(key, position, params, top_indices, loglikelihood_fn, stepper_fn):
    state = StateWithLogLikelihood(
        jnp.atleast_1d(jnp.asarray(position)),
        jnp.array(0.0),
        jnp.array(0.0),
        _LOGLIKELIHOOD_0,
    )
    return _de_one_step(
        key,
        state,
        _LOGLIKELIHOOD_0,
        _log_prior,
        loglikelihood_fn,
        params,
        stepper_fn,
        _NUM_SURVIVORS,
        top_indices,
    )


def test_out_of_cube_proposal_is_a_spent_step():
    """The point stays put and the likelihood is not evaluated outside the cube."""
    params, top_indices = _flat_params(jax.random.key(0))

    def nan_outside_cube(x):
        return jnp.where(jnp.any(x > 1.0), jnp.nan, 0.0)

    with jax.debug_nans(True):
        new_state, info = _step(
            jax.random.key(1),
            0.75,
            params,
            top_indices,
            nan_outside_cube,
            lambda position, delta, gamma: position + 10.0,
        )

    assert not bool(info.is_accepted)
    assert float(new_state.position[0]) == pytest.approx(0.75)
    assert int(info.likelihood_evals) == 0


def test_in_cube_proposal_is_accepted():
    params, top_indices = _flat_params(jax.random.key(0))

    new_state, info = _step(
        jax.random.key(1),
        0.75,
        params,
        top_indices,
        lambda x: 0.0 * x[0],
        lambda position, delta, gamma: jnp.full_like(position, 0.25),
    )

    assert bool(info.is_accepted)
    assert float(new_state.position[0]) == pytest.approx(0.25)
    assert int(info.likelihood_evals) == 1


def test_step_keeps_a_uniform_distribution_uniform():
    """With a flat likelihood a step must leave uniform points uniform.

    This needs an out-of-cube proposal to be a spent step: redrawing it inside
    the step would push points towards the centre of the cube.
    """
    n_points = 20000
    live_key, start_key, step_key = jax.random.split(jax.random.key(0), 3)
    params, top_indices = _flat_params(live_key)
    stepper_fn = to_unit_cube_stepper(None, 1)
    start = jax.random.uniform(start_key, (n_points, 1))

    def one_step(key, position):
        new_state, _ = _step(
            key, position, params, top_indices, lambda x: 0.0 * x[0], stepper_fn
        )
        return new_state.position[0]

    stepped = jax.vmap(one_step)(jax.random.split(step_key, n_points), start)

    central_fraction = float(jnp.mean((stepped > 0.25) & (stepped < 0.75)))
    standard_error = (0.25 / n_points) ** 0.5
    assert abs(central_fraction - 0.5) < 4 * standard_error
