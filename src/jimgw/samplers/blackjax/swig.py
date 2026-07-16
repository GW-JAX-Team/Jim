"""Cache-aware Nested Slice within Gibbs (SwiG) sampling."""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from blackjax import SamplingAlgorithm
from blackjax.mcmc.slice import SliceInfo
from blackjax.mcmc.slice import build_kernel as build_slice_kernel
from blackjax.mcmc.slice import stepping_out
from blackjax.ns.from_mcmc import build_kernel as build_from_mcmc_kernel
from blackjax.ns.nss import sample_direction_from_covariance
from blackjax.smc.tuning.from_particles import particles_covariance_matrix

from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler
from jimgw.samplers.config import BlackJAXSwiGConfig
from jimgw.samplers.periodic import _build_masks_arrays


class CachedSliceState(NamedTuple):
    """Ephemeral slice state; caches are never stored on the live particles."""

    position: jax.Array
    logdensity: jax.Array
    loglikelihood: jax.Array
    loglikelihood_birth: jax.Array
    cache: object


def _build_block_covariance_update(
    block_indices: tuple[tuple[int, ...], ...],
) -> Callable:
    def update(rng_key, state, info, params=None):
        del rng_key, info, params
        covariance = jnp.atleast_2d(
            particles_covariance_matrix(state.particles.position)
        )
        covariances = tuple(
            covariance[jnp.ix_(jnp.asarray(block), jnp.asarray(block))]
            for block in block_indices
        )
        return {"block_covariances": covariances}

    return update


def _build_swig_constrained_step(
    *,
    log_prior_fn: Callable,
    build_cache_fn: Callable,
    log_likelihood_from_cache_fn: Callable,
    block_indices: tuple[tuple[int, ...], ...],
    refresh_cache: tuple[bool, ...],
    num_gibbs_sweeps: int,
    num_inner_steps_per_dim: int,
    max_steps: int,
    max_shrinkage: int,
    periodic: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
) -> Callable:
    slice_kernel = build_slice_kernel(
        interval=stepping_out,
        max_expansions=max_steps,
        max_shrinkage=max_shrinkage,
    )
    periodic_mask, periodic_lower, periodic_period = _build_masks_arrays(
        periodic, n_dims
    )

    def wrap(position):
        return jnp.where(
            periodic_mask,
            periodic_lower + jnp.mod(position - periodic_lower, periodic_period),
            position,
        )

    def constrained_step(rng_key, state, loglikelihood_0, block_covariances):
        cache = build_cache_fn(state.position)
        cached_state = CachedSliceState(
            position=state.position,
            logdensity=state.logdensity,
            loglikelihood=state.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
            cache=cache,
        )
        accepted = jnp.asarray(True)
        num_expansions = jnp.asarray(0)
        num_shrink = jnp.asarray(0)

        for _ in range(num_gibbs_sweeps):
            for block, must_refresh, covariance in zip(
                block_indices, refresh_cache, block_covariances, strict=True
            ):
                block_array = jnp.asarray(block)
                n_steps = num_inner_steps_per_dim * len(block)

                def one_slice(carry, key):
                    current, all_accepted, expansions, shrink = carry

                    def proposal_generator(direction_key, position, logdensity_fn):
                        del logdensity_fn
                        block_position = position[block_array]
                        block_direction = sample_direction_from_covariance(
                            direction_key, block_position, covariance
                        )
                        direction = (
                            jnp.zeros_like(position)
                            .at[block_array]
                            .set(block_direction)
                        )

                        def slice_fn(t):
                            proposed = wrap(position + t * direction)
                            logprior = log_prior_fn(proposed)
                            if must_refresh:
                                proposed_cache = build_cache_fn(proposed)
                            else:
                                proposed_cache = current.cache
                            loglikelihood = log_likelihood_from_cache_fn(
                                proposed, proposed_cache
                            )
                            proposed_state = CachedSliceState(
                                position=proposed,
                                logdensity=logprior,
                                loglikelihood=loglikelihood,
                                loglikelihood_birth=jnp.asarray(loglikelihood_0),
                                cache=proposed_cache,
                            )
                            return proposed_state, loglikelihood > loglikelihood_0

                        return slice_fn

                    new_state, info = slice_kernel(
                        key, current, None, proposal_generator
                    )
                    return (
                        new_state,
                        all_accepted & info.is_accepted,
                        expansions + info.num_expansions,
                        shrink + info.num_shrink,
                    ), None

                keys = jax.random.split(rng_key, n_steps + 1)
                rng_key = keys[0]
                (cached_state, accepted, num_expansions, num_shrink), _ = jax.lax.scan(
                    one_slice,
                    (cached_state, accepted, num_expansions, num_shrink),
                    keys[1:],
                )

        final_state = state._replace(
            position=cached_state.position,
            logdensity=cached_state.logdensity,
            loglikelihood=cached_state.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
        )
        info = SliceInfo(
            is_accepted=accepted,
            num_expansions=num_expansions,
            num_shrink=num_shrink,
            bracket_left=jnp.zeros(n_dims),
            bracket_right=jnp.zeros(n_dims),
        )
        return final_state, info

    return constrained_step


class BlackJAXSwiGSampler(BlackJAXNSSSampler):
    """Nested Slice within Gibbs using cache-aware slices over named blocks."""

    _config: BlackJAXSwiGConfig

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: BlackJAXSwiGConfig,
        periodic: Optional[dict[int, tuple[float, float]]] = None,
        block_indices: Optional[tuple[tuple[int, ...], ...]] = None,
        refresh_cache: Optional[tuple[bool, ...]] = None,
        build_cache_fn: Optional[Callable] = None,
        log_likelihood_from_cache_fn: Optional[Callable] = None,
    ) -> None:
        if block_indices is None or refresh_cache is None:
            raise ValueError("resolved block indices and cache flags are required")
        if build_cache_fn is None or log_likelihood_from_cache_fn is None:
            raise ValueError("cache-aware likelihood callables are required")
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,  # type: ignore[arg-type]
            periodic=periodic,
        )
        self._block_indices = block_indices
        self._refresh_cache = refresh_cache
        self._build_cache_fn = build_cache_fn
        self._log_likelihood_from_cache_fn = log_likelihood_from_cache_fn
        self._periodic = periodic
        self._block_covariance_update = _build_block_covariance_update(block_indices)

    @property
    def _checkpoint_tag(self) -> str:
        return "SwiG"

    @property
    def _update_inner_kernel_params_fn(self) -> Callable:
        return self._block_covariance_update

    def _build_nested_sampler(self, n_delete: int) -> SamplingAlgorithm:
        constrained_step = _build_swig_constrained_step(
            log_prior_fn=self._log_prior_fn,
            build_cache_fn=self._build_cache_fn,
            log_likelihood_from_cache_fn=self._log_likelihood_from_cache_fn,
            block_indices=self._block_indices,
            refresh_cache=self._refresh_cache,
            num_gibbs_sweeps=self._config.num_gibbs_sweeps,
            num_inner_steps_per_dim=self._config.num_inner_steps_per_dim,
            max_steps=self._config.max_steps,
            max_shrinkage=self._config.max_shrinkage,
            periodic=self._periodic,
            n_dims=self.n_dims,
        )
        kernel = build_from_mcmc_kernel(
            constrained_step,
            num_inner_steps=1,
            update_inner_kernel_params_fn=self._block_covariance_update,
            num_delete=n_delete,
        )
        return SamplingAlgorithm(lambda position, rng_key=None: position, kernel)
