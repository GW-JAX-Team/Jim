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
    _de_walk,
    bilby_adaptive_de_sampler,
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
    assert int(info.steps) == 1


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


def test_out_of_cube_proposals_are_redrawn_as_steps():
    """Out-of-cube proposals are redrawn within a call, each counted as a step."""
    params, top_indices = _flat_params(jax.random.key(0))
    state = StateWithLogLikelihood(
        jnp.array([0.75]), jnp.array(0.0), jnp.array(0.0), _LOGLIKELIHOOD_0
    )

    new_state, info = _de_one_step(
        jax.random.key(1),
        state,
        _LOGLIKELIHOOD_0,
        _log_prior,
        lambda x: 0.0 * x[0],
        params,
        lambda position, delta, gamma: position + 10.0,
        _NUM_SURVIVORS,
        top_indices,
        max_steps_remaining=5,
    )

    assert int(info.steps) == 5
    assert int(info.likelihood_evals) == 0
    assert not bool(info.is_accepted)
    assert float(new_state.position[0]) == pytest.approx(0.75)


def _walk_params(live_points, num_walks):
    n_live = live_points.shape[0]
    loglikelihoods = jnp.zeros(n_live)
    params = DEKernelParams(
        live_points=live_points,
        loglikelihoods=loglikelihoods,
        mix=0.5,
        scale=2.38 / jnp.sqrt(2.0 * live_points.shape[1]),
        num_walks=jnp.array(num_walks),
        walks_float=jnp.array(float(num_walks)),
    )
    return params


def _run_walks(keys, starts, params, loglikelihood_fn, num_survivors):
    stepper_fn = to_unit_cube_stepper(None, starts.shape[1])

    def walk(key, position):
        state = StateWithLogLikelihood(
            position, jnp.array(0.0), loglikelihood_fn(position), _LOGLIKELIHOOD_0
        )
        return _de_walk(
            key,
            state,
            _LOGLIKELIHOOD_0,
            _log_prior,
            loglikelihood_fn,
            params,
            stepper_fn,
            num_survivors,
        )

    return jax.vmap(walk)(keys, starts)


def test_walk_keeps_a_uniform_distribution_uniform():
    """A whole walk with a flat likelihood leaves uniform points uniform.

    Every out-of-cube proposal must count as a walk step; redraws that did not
    count would push points towards the centre of the cube.
    """
    n_points, num_walks = 20000, 20
    live_key, start_key, walk_key = jax.random.split(jax.random.key(0), 3)
    live_points = jax.random.uniform(live_key, (_N_LIVE, 1))
    params = _walk_params(live_points, num_walks)
    starts = jax.random.uniform(start_key, (n_points, 1))

    final, info = _run_walks(
        jax.random.split(walk_key, n_points),
        starts,
        params,
        lambda x: 0.0 * x[0],
        _NUM_SURVIVORS,
    )

    assert bool(jnp.all(info.walks_completed == num_walks))
    assert bool(jnp.all(info.n_likelihood_evals <= num_walks))
    assert float(jnp.mean(info.n_likelihood_evals)) < num_walks
    central_fraction = float(
        jnp.mean((final.position[:, 0] > 0.25) & (final.position[:, 0] < 0.75))
    )
    standard_error = (0.25 / n_points) ** 0.5
    assert abs(central_fraction - 0.5) < 4 * standard_error


def test_walk_matches_one_proposal_per_step():
    """A walk gives the same final distribution as one proposal per step.

    The reference takes ``num_walks`` single-proposal steps, each spent when its
    proposal leaves the cube. The live points sit near a corner so that most
    proposals leave the cube, and a likelihood constraint is active.
    """
    n_points, num_walks, n_dim = 20000, 30, 2
    live_key, start_key, walk_key, ref_key = jax.random.split(jax.random.key(3), 4)
    live_points = 0.2 * jax.random.uniform(live_key, (_N_LIVE, n_dim))
    params = _walk_params(live_points, num_walks)
    starts = 0.2 * jax.random.uniform(start_key, (n_points, n_dim))
    _, top_indices = jax.lax.top_k(params.loglikelihoods, _NUM_SURVIVORS)

    def loglikelihood_fn(x):
        return -jnp.sum(x)

    loglikelihood_0 = jnp.array(-0.3)
    stepper_fn = to_unit_cube_stepper(None, n_dim)
    starts = jnp.where(jnp.sum(starts, axis=1, keepdims=True) < 0.3, starts, 0.05)

    def walk(key, position):
        state = StateWithLogLikelihood(
            position, jnp.array(0.0), loglikelihood_fn(position), loglikelihood_0
        )
        return _de_walk(
            key,
            state,
            loglikelihood_0,
            _log_prior,
            loglikelihood_fn,
            params,
            stepper_fn,
            _NUM_SURVIVORS,
        )

    def reference(key, position):
        state = StateWithLogLikelihood(
            position, jnp.array(0.0), loglikelihood_fn(position), loglikelihood_0
        )

        def body(state, step_key):
            new_state, info = _de_one_step(
                step_key,
                state,
                loglikelihood_0,
                _log_prior,
                loglikelihood_fn,
                params,
                stepper_fn,
                _NUM_SURVIVORS,
                top_indices,
                max_steps_remaining=1,
            )
            return new_state, info.likelihood_evals

        final, evals = jax.lax.scan(body, state, jax.random.split(key, num_walks))
        return final, jnp.sum(evals)

    final, info = jax.vmap(walk)(jax.random.split(walk_key, n_points), starts)
    ref_final, ref_evals = jax.vmap(reference)(
        jax.random.split(ref_key, n_points), starts
    )

    assert bool(jnp.all(info.walks_completed == num_walks))
    # Most proposals leave the cube here, so most steps need no likelihood.
    assert float(jnp.mean(info.n_likelihood_evals)) < 0.75 * num_walks
    assert float(jnp.mean(info.n_likelihood_evals)) == pytest.approx(
        float(jnp.mean(ref_evals)), rel=0.05
    )
    # Two-sample Kolmogorov-Smirnov distance per dimension (critical value at 0.1%).
    critical = 1.95 * (2.0 / n_points) ** 0.5
    for d in range(n_dim):
        a = jnp.sort(final.position[:, d])
        b = jnp.sort(ref_final.position[:, d])
        grid = jnp.concatenate([a, b])
        ecdf_a = jnp.searchsorted(a, grid, side="right") / n_points
        ecdf_b = jnp.searchsorted(b, grid, side="right") / n_points
        assert float(jnp.max(jnp.abs(ecdf_a - ecdf_b))) < critical


def test_walks_adapt_to_the_accepts_of_the_same_step():
    """Each update divides the accepts of a step by the walk length of that step."""
    nlive, num_delete, n_target = 200, 10, 60
    sampler = bilby_adaptive_de_sampler(
        _log_prior,
        lambda x: -0.5 * jnp.sum(((x - 0.5) / 0.1) ** 2),
        nlive=nlive,
        n_target=n_target,
        num_delete=num_delete,
        stepper_fn=to_unit_cube_stepper(None, 2),
    )
    init_key, run_key = jax.random.split(jax.random.key(0))
    state = sampler.init(jax.random.uniform(init_key, (nlive, 2)))
    step = jax.jit(sampler.step)
    delay = max((nlive // 10 - 1) // num_delete, 1)

    for step_key in jax.random.split(run_key, 4):
        used = state.inner_kernel_params["params"]
        state, info = step(step_key, state)
        accept = float(jnp.sum(info.update_info.n_accept)) / num_delete
        assert accept > 0
        accept_prob = max(0.5, accept) / int(used.num_walks)
        expected = (float(used.walks_float) * delay + n_target / accept_prob) / (
            delay + 1
        )
        new_params = state.inner_kernel_params["params"]
        assert float(new_params.walks_float) == pytest.approx(expected, rel=1e-5)


def test_walks_grow_after_a_step_without_accepts():
    """A step without accepts lengthens the walks, counting it as half an accept."""
    nlive, num_delete, n_target = 200, 10, 60
    sampler = bilby_adaptive_de_sampler(
        _log_prior,
        lambda x: 0.0 * x[0],
        nlive=nlive,
        n_target=n_target,
        num_delete=num_delete,
        stepper_fn=to_unit_cube_stepper(None, 1),
    )
    init_key, step_key = jax.random.split(jax.random.key(0))
    state = sampler.init(jax.random.uniform(init_key, (nlive, 1)))
    used = state.inner_kernel_params["params"]

    state, info = jax.jit(sampler.step)(step_key, state)

    assert int(jnp.sum(info.update_info.n_accept)) == 0
    delay = max((nlive // 10 - 1) // num_delete, 1)
    accept_prob = 0.5 / int(used.num_walks)
    expected = (float(used.walks_float) * delay + n_target / accept_prob) / (delay + 1)
    new_params = state.inner_kernel_params["params"]
    assert float(new_params.walks_float) == pytest.approx(expected, rel=1e-5)


def test_step_keeps_the_state_types():
    """A step returns a state of the same types, so the step compiles only once."""
    nlive, num_delete = 40, 4
    sampler = bilby_adaptive_de_sampler(
        _log_prior,
        lambda x: -0.5 * jnp.sum(((x - 0.5) / 0.1) ** 2),
        nlive=nlive,
        num_delete=num_delete,
        stepper_fn=to_unit_cube_stepper(None, 2),
    )
    init_key, step_key = jax.random.split(jax.random.key(0))
    state = sampler.init(jax.random.uniform(init_key, (nlive, 2)))

    new_state, _ = jax.jit(sampler.step)(step_key, state)

    assert jax.tree.map(jax.typeof, new_state) == jax.tree.map(jax.typeof, state)
