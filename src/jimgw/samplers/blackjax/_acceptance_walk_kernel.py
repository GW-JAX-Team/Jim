"""bilby/dynesty-style adaptive DE acceptance-walk kernel for unit-hypercube nested sampling.

Ported and modified from:
  https://github.com/mrosep/blackjax_ns_gw/blob/main/src/custom_kernels/acceptance_walk.py

Reference
---------
Prathaban, M., Yallup, D., Alvey, J., Yang, M., Templeton, W., Handley, W.,
"Gravitational-wave inference at GPU speed: A bilby-like nested sampling
kernel within blackjax-ns", arXiv:2509.04336 (Sep 2025).
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple, cast

import jax
import jax.flatten_util
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.ns.base import (
    NSState,
    StateWithLogLikelihood,
    delete_fn as default_delete_fn,
    init_state_strategy,
)
from blackjax.ns.adaptive import (
    AdaptiveNSState,
    build_kernel as build_adaptive_kernel,
    init as adaptive_init,
)


class DEInfo(NamedTuple):
    is_accepted: Any
    evals: Any
    likelihood_evals: Any


class DEWalkInfo(NamedTuple):
    n_accept: Any
    walks_completed: Any
    n_likelihood_evals: Any
    total_proposals: Any


class DEKernelParams(NamedTuple):
    live_points: Any
    loglikelihoods: Any
    mix: Any
    scale: Any
    num_walks: Any
    walks_float: Any
    n_accept_total: Any
    n_likelihood_evals_total: Any


def _de_one_step(
    rng_key: jax.Array,
    state: StateWithLogLikelihood,
    loglikelihood_0: jax.Array,
    logprior_fn,
    loglikelihood_fn,
    params: DEKernelParams,
    stepper_fn,
    num_survivors: int,
    max_proposals: int = 1000,
):
    def body_fun(carry):
        is_valid, key, pos, logp, count = carry
        key_a, key_b, key_mix, key_gamma, new_key = jax.random.split(key, 5)

        _, top_indices = jax.lax.top_k(params.loglikelihoods, num_survivors)
        pos_a = jax.random.randint(key_a, (), 0, num_survivors)
        pos_b_raw = jax.random.randint(key_b, (), 0, num_survivors - 1)
        pos_b = jnp.where(pos_b_raw >= pos_a, pos_b_raw + 1, pos_b_raw)

        point_a = jax.tree_util.tree_map(
            lambda x: x[top_indices[pos_a]], params.live_points
        )
        point_b = jax.tree_util.tree_map(
            lambda x: x[top_indices[pos_b]], params.live_points
        )
        delta = jax.tree_util.tree_map(lambda a, b: a - b, point_a, point_b)

        is_small_step = jax.random.uniform(key_mix) < params.mix
        gamma = jnp.where(
            is_small_step,
            params.scale * jax.random.gamma(key_gamma, 4.0) * 0.25,
            1.0,
        )

        new_pos = stepper_fn(state.position, delta, gamma)
        new_logp = logprior_fn(new_pos)
        new_is_valid = jnp.isfinite(new_logp)

        return (new_is_valid, new_key, new_pos, new_logp, count + 1)

    def cond_fun(carry):
        is_valid, _, _, _, count = carry
        return jnp.logical_and(jnp.logical_not(is_valid), count < max_proposals)

    init = (jnp.array(False), rng_key, state.position, state.logdensity, jnp.array(0))
    is_valid, _, pos_prop, logp_prop, n_proposals = jax.lax.while_loop(
        cond_fun, body_fun, init
    )

    logl_prop = loglikelihood_fn(pos_prop)
    is_accepted = jnp.logical_and(is_valid, logl_prop > loglikelihood_0)

    final_pos = jax.tree_util.tree_map(
        lambda p, c: jnp.where(is_accepted, p, c), pos_prop, state.position
    )
    final_logp = jnp.where(is_accepted, logp_prop, state.logdensity)
    final_logl = jnp.where(is_accepted, logl_prop, state.loglikelihood)

    new_state = StateWithLogLikelihood(
        final_pos, final_logp, final_logl, loglikelihood_0
    )
    likelihood_evals = is_valid.astype(jnp.int32)
    info = DEInfo(
        is_accepted=is_accepted, evals=n_proposals, likelihood_evals=likelihood_evals
    )

    return new_state, info


def _de_walk(
    rng_key: jax.Array,
    state: StateWithLogLikelihood,
    loglikelihood_0: jax.Array,
    logprior_fn,
    loglikelihood_fn,
    params: DEKernelParams,
    stepper_fn,
    num_survivors: int,
    max_proposals: int = 1000,
    max_mcmc: int = 5000,
):
    one_step = partial(
        _de_one_step,
        num_survivors=num_survivors,
        max_proposals=max_proposals,
    )

    def single_step(rng_key, state, loglikelihood_0):
        return one_step(
            rng_key=rng_key,
            state=state,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            loglikelihood_0=loglikelihood_0,
            params=params,
            stepper_fn=stepper_fn,
        )

    def cond_fun(carry):
        _, _, _, _, total_proposals, walks_completed = carry
        return jnp.logical_and(
            walks_completed < params.num_walks, total_proposals < max_mcmc
        )

    def body_fun(carry):
        (
            key,
            current_state,
            n_accept,
            n_likelihood_evals,
            total_proposals,
            walks_completed,
        ) = carry
        step_key, next_key = jax.random.split(key)
        new_state, info = single_step(step_key, current_state, loglikelihood_0)
        return (
            next_key,
            new_state,
            n_accept + info.is_accepted,
            n_likelihood_evals + info.likelihood_evals,
            total_proposals + info.evals,
            walks_completed + 1,
        )

    init_val = (
        rng_key,
        state,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )

    _, final_state, n_accept, n_likelihood_evals, total_proposals, walks_completed = (
        jax.lax.while_loop(cond_fun, body_fun, init_val)
    )

    info = DEWalkInfo(
        n_accept=n_accept,
        walks_completed=walks_completed,
        n_likelihood_evals=n_likelihood_evals,
        total_proposals=total_proposals,
    )
    return final_state, info


def _update_bilby_walks(
    ns_state: NSState,
    logprior_fn,
    loglikelihood_fn,
    n_target: int,
    max_mcmc: int,
    n_delete: int,
) -> DEKernelParams:
    prev_params = ns_state.inner_kernel_params["params"]  # type: ignore[attr-defined]  # blackjax fork stubs
    is_uninitialized = prev_params.n_accept_total < 0

    default_walks_float = jnp.array(100.0, dtype=jnp.float32)
    default_n_accept_total = jnp.array(0, dtype=jnp.int32)
    default_current_walks = jnp.array(100, dtype=jnp.int32)

    walks_float = cast(
        jax.Array,
        jnp.where(
            is_uninitialized,
            default_walks_float,
            prev_params.walks_float.astype(jnp.float32),
        ),
    )
    n_accept_total = cast(
        jax.Array,
        jnp.where(
            is_uninitialized,
            default_n_accept_total,
            prev_params.n_accept_total.astype(jnp.int32),
        ),
    )
    current_walks = cast(
        jax.Array,
        jnp.where(
            is_uninitialized,
            default_current_walks,
            prev_params.num_walks.astype(jnp.int32),
        ),
    )

    leaves = cast(list[jax.Array], jax.tree_util.tree_leaves(ns_state.particles))
    nlive = leaves[0].shape[0]
    og_delay = nlive // 10 - 1
    delay = jnp.maximum(og_delay // n_delete, 1)

    avg_accept_per_particle = n_accept_total / n_delete
    accept_prob = jnp.maximum(0.5, avg_accept_per_particle) / jnp.maximum(
        1.0, current_walks
    )
    new_walks_float = (walks_float * delay + n_target / accept_prob) / (delay + 1)
    new_walks_float = cast(
        jax.Array, jnp.where(n_accept_total == 0, walks_float, new_walks_float)
    )
    num_walks_int = jnp.minimum(jnp.ceil(new_walks_float).astype(jnp.int32), max_mcmc)

    example_particle = jax.tree_util.tree_map(
        lambda x: x[0], ns_state.particles.position
    )
    flat_particle, _ = jax.flatten_util.ravel_pytree(example_particle)
    n_dim = flat_particle.shape[0]

    # n_accept_total / n_likelihood_evals_total are immediately overwritten by
    # update_fn with the per-batch counts; the zeros below are placeholders.
    return DEKernelParams(
        live_points=ns_state.particles.position,
        loglikelihoods=ns_state.particles.loglikelihood,
        mix=0.5,
        scale=2.38 / jnp.sqrt(2 * n_dim),
        num_walks=jnp.array(num_walks_int, dtype=jnp.int32),
        walks_float=jnp.array(new_walks_float, dtype=jnp.float32),
        n_accept_total=jnp.array(0, dtype=jnp.int32),
        n_likelihood_evals_total=jnp.array(0, dtype=jnp.int32),
    )


def bilby_adaptive_de_sampler(
    logprior_fn,
    loglikelihood_fn,
    nlive: int,
    n_target: int = 60,
    max_mcmc: int = 5000,
    num_delete: int = 1,
    stepper_fn=None,
    max_proposals: int = 1000,
) -> SamplingAlgorithm:
    """bilby/dynesty-style adaptive DE sampler for unit-hypercube nested sampling."""
    if stepper_fn is None:
        raise ValueError("stepper_fn must be provided")

    num_survivors = nlive - num_delete
    delete_fn = partial(default_delete_fn, num_delete=num_delete)

    def update_fn(rng_key, ns_state, info, current_params):
        new_params = _update_bilby_walks(
            ns_state=ns_state,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            n_target=n_target,
            max_mcmc=max_mcmc,
            n_delete=num_delete,
        )
        batch_n_accept = jnp.sum(info.update_info.n_accept)
        batch_n_likelihood_evals = jnp.sum(info.update_info.n_likelihood_evals)
        new_params = new_params._replace(
            n_accept_total=batch_n_accept,
            n_likelihood_evals_total=batch_n_likelihood_evals,
        )
        return {"params": new_params}

    def inner_kernel(rng_key, state, loglikelihood_0, *, params):
        choice_key, sample_key = jax.random.split(rng_key)
        particles = state.particles
        loglikelihoods = particles.loglikelihood
        weights = (loglikelihoods > loglikelihood_0).astype(jnp.float32)
        weights = cast(
            jax.Array, jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
        )
        start_idx = jax.random.choice(
            choice_key,
            weights.shape[0],
            shape=(num_delete,),
            p=weights / weights.sum(),
            replace=True,
        )
        start_states = jax.tree_util.tree_map(lambda x: x[start_idx], particles)
        sub_keys = jax.random.split(sample_key, num_delete)

        def single(rng_key, state):
            return _de_walk(
                rng_key=rng_key,
                state=state,
                loglikelihood_0=loglikelihood_0,
                logprior_fn=logprior_fn,
                loglikelihood_fn=loglikelihood_fn,
                params=params,
                stepper_fn=stepper_fn,
                num_survivors=num_survivors,
                max_proposals=max_proposals,
                max_mcmc=max_mcmc,
            )

        return jax.vmap(single)(sub_keys, start_states)

    base_kernel_step = build_adaptive_kernel(delete_fn, inner_kernel, update_fn)  # type: ignore[arg-type]  # blackjax fork stubs

    def init_fn(particles):
        _init_state_fn = partial(
            init_state_strategy,
            logprior_fn=jax.vmap(logprior_fn),
            loglikelihood_fn=jax.vmap(loglikelihood_fn),
        )

        def init_params_fn(rng_key, ns_state, info, current_params):
            example_particle = jax.tree_util.tree_map(
                lambda x: x[0], ns_state.particles.position
            )
            flat_particle, _ = jax.flatten_util.ravel_pytree(example_particle)
            n_dim = flat_particle.shape[0]
            scale = 2.38 / jnp.sqrt(2 * n_dim)
            initial_de_params = DEKernelParams(
                live_points=ns_state.particles.position,
                loglikelihoods=ns_state.particles.loglikelihood,
                mix=0.5,
                scale=scale,
                num_walks=jnp.array(100, dtype=jnp.int32),
                walks_float=jnp.array(100.0, dtype=jnp.float32),
                n_accept_total=jnp.array(-1, dtype=jnp.int32),
                n_likelihood_evals_total=jnp.array(-1, dtype=jnp.int32),
            )
            return {"params": initial_de_params}

        return adaptive_init(
            particles, _init_state_fn, update_inner_kernel_params_fn=init_params_fn
        )

    def step_fn(rng_key, state: AdaptiveNSState):
        return base_kernel_step(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]  # blackjax fork stubs
