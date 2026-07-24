"""Multi-device strategies for BlackJAX nested sampling."""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

_LIVE_AXIS = "live"


def make_live_mesh(n_devices: int, n_live: int, n_delete: int) -> Optional[Mesh]:
    """Build a single-host device mesh for the live-particle axis."""
    if n_devices == 1:
        return None

    devices = jax.local_devices()
    if n_devices > len(devices):
        raise ValueError(
            f"n_devices={n_devices} requested, but only {len(devices)} local "
            "JAX devices are available."
        )
    if n_live % n_devices:
        raise ValueError(f"n_live={n_live} must be divisible by n_devices={n_devices}.")
    if n_delete % n_devices:
        raise ValueError(
            f"n_delete={n_delete} must be divisible by n_devices={n_devices}."
        )

    return Mesh(np.asarray(devices[:n_devices]), axis_names=(_LIVE_AXIS,))


def place_state(state, mesh: Mesh):
    """Shard live particles and replicate adaptive/integrator state."""
    live = NamedSharding(mesh, P(_LIVE_AXIS))
    replicated = NamedSharding(mesh, P())
    return state._replace(
        particles=jax.tree.map(lambda x: jax.device_put(x, live), state.particles),
        inner_kernel_params=jax.tree.map(
            lambda x: jax.device_put(x, replicated), state.inner_kernel_params
        ),
        integrator=jax.tree.map(
            lambda x: jax.device_put(x, replicated), state.integrator
        ),
    )


def place_key(rng_key, mesh: Mesh):
    """Replicate a sampler PRNG key over the mesh."""
    return jax.device_put(rng_key, NamedSharding(mesh, P()))


def _all_gather(mesh: Mesh, value, in_spec):
    def gather(x):
        return jax.lax.all_gather(x, _LIVE_AXIS, tiled=True)

    return jax.shard_map(
        gather,
        mesh=mesh,
        in_specs=in_spec,
        out_specs=P(),
        check_vma=False,
    )(value)


def delete_fn_sharded(mesh: Mesh, n_delete: int) -> Callable:
    """Select the globally lowest-likelihood live particles."""

    def delete_fn(state):
        global_log_likelihood = _all_gather(
            mesh, state.particles.loglikelihood, P(_LIVE_AXIS)
        )
        _, dead_idx = jax.lax.top_k(-global_log_likelihood, n_delete)
        return dead_idx, dead_idx

    return delete_fn


def update_with_mcmc_take_last_sharded(
    constrained_step_fn: Callable,
    n_inner_steps: int,
    n_delete: int,
    mesh: Mesh,
) -> Callable:
    """Run independent constrained replacement chains across the mesh."""
    live = NamedSharding(mesh, P(_LIVE_AXIS))

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        particles = state.particles
        particle_specs = jax.tree.map(
            lambda x: P(_LIVE_AXIS) if x.ndim > 0 else P(), particles
        )
        global_particles = jax.shard_map(
            lambda xs: jax.tree.map(
                lambda x: jax.lax.all_gather(x, _LIVE_AXIS, tiled=True), xs
            ),
            mesh=mesh,
            in_specs=(particle_specs,),
            out_specs=jax.tree.map(lambda _: P(), particles),
            check_vma=False,
        )(particles)
        global_log_likelihood = global_particles.loglikelihood

        survivor_weights = (global_log_likelihood > loglikelihood_0).astype(jnp.float32)
        survivor_weights = jnp.where(
            survivor_weights.sum() > 0,
            survivor_weights,
            jnp.ones_like(survivor_weights),
        )

        choice_key, sample_key = jax.random.split(rng_key)
        start_idx = jax.random.choice(
            choice_key,
            global_log_likelihood.shape[0],
            shape=(n_delete,),
            p=survivor_weights / survivor_weights.sum(),
            replace=True,
        )
        start_states = jax.tree.map(lambda x: x[start_idx], global_particles)
        start_states = jax.tree.map(lambda x: jax.device_put(x, live), start_states)
        sample_keys = jax.device_put(jax.random.split(sample_key, n_delete), live)

        state_specs = jax.tree.map(
            lambda x: P(_LIVE_AXIS) if x.ndim > 0 else P(), start_states
        )
        parameter_specs = jax.tree.map(lambda _: P(), step_parameters)

        def run_local_chains(keys, states, threshold, parameters):
            def run_chain(key, chain_state):
                keys = jax.random.split(key, n_inner_steps)

                def body_fn(current_state, step_key):
                    return constrained_step_fn(
                        step_key,
                        current_state,
                        threshold,
                        **parameters,
                    )

                return jax.lax.scan(body_fn, chain_state, keys)

            return jax.vmap(run_chain)(keys, states)

        return jax.shard_map(
            run_local_chains,
            mesh=mesh,
            in_specs=(P(_LIVE_AXIS), state_specs, P(), parameter_specs),
            out_specs=(state_specs, P(_LIVE_AXIS)),
            check_vma=False,
        )(sample_keys, start_states, loglikelihood_0, step_parameters)

    return update_function


def build_sharded_from_mcmc_kernel(
    constrained_step_fn: Callable,
    n_inner_steps: int,
    update_inner_kernel_params_fn: Callable,
    n_delete: int,
    mesh: Mesh,
) -> Callable:
    """Assemble BlackJAX's adaptive NS kernel with sharded strategies."""
    inner_kernel = update_with_mcmc_take_last_sharded(
        constrained_step_fn,
        n_inner_steps=n_inner_steps,
        n_delete=n_delete,
        mesh=mesh,
    )
    return build_adaptive_kernel(
        delete_fn_sharded(mesh, n_delete),
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
