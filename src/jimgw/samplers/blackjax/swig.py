"""Cache-aware Nested Slice within Gibbs (SwiG) sampling."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from blackjax import SamplingAlgorithm
from blackjax.base import Position, State
from blackjax.mcmc.slice import SliceInfo
from blackjax.mcmc.slice import build_kernel as build_slice_kernel
from blackjax.mcmc.slice import stepping_out
from blackjax.ns.from_mcmc import build_kernel as build_from_mcmc_kernel
from blackjax.ns.nss import sample_direction_from_covariance
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.types import PRNGKey
from jaxtyping import Array, Float

from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler
from jimgw.samplers.config import BlackJAXNSSConfig, BlackJAXSwiGConfig
from jimgw.samplers.periodic import _build_masks_arrays
from jimgw.typing import FloatScalar


class CachedSliceState(NamedTuple):
    """Ephemeral slice state; caches are never stored on the live particles."""

    position: Float[Array, " n_dims"]
    logdensity: FloatScalar
    loglikelihood: FloatScalar
    loglikelihood_birth: FloatScalar
    cache: object


def _build_swig_constrained_step(
    *,
    log_prior_fn: Callable,
    build_cache: Callable,
    log_likelihood_from_cache_fn: Callable,
    rebuild_required_by_block: dict[tuple[int, ...], bool],
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

    def wrap_periodic_position(position):
        return jnp.where(
            periodic_mask,
            periodic_lower + jnp.mod(position - periodic_lower, periodic_period),
            position,
        )

    def constrained_step(rng_key, state, loglikelihood_0, block_covariances):
        cache = build_cache(state.position)
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
            for (parameter_indices, requires_rebuild), covariance in zip(
                rebuild_required_by_block.items(), block_covariances, strict=True
            ):
                parameter_index_array = jnp.asarray(parameter_indices)
                n_steps = num_inner_steps_per_dim * len(parameter_indices)

                def one_slice(
                    carry,
                    key,
                    parameter_index_array=parameter_index_array,
                    covariance=covariance,
                    requires_rebuild=requires_rebuild,
                ):
                    current, all_accepted, expansions, shrink = carry

                    def proposal_generator(direction_key, position, logdensity_fn):
                        del logdensity_fn
                        block_position = position[parameter_index_array]
                        block_direction = sample_direction_from_covariance(
                            direction_key, block_position, covariance
                        )
                        direction = (
                            jnp.zeros_like(position)
                            .at[parameter_index_array]
                            .set(block_direction)
                        )

                        def slice_fn(t):
                            proposed = wrap_periodic_position(position + t * direction)
                            logprior = log_prior_fn(proposed)
                            proposed_cache = (
                                build_cache(proposed)
                                if requires_rebuild
                                else current.cache
                            )
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

    _swig_config: BlackJAXSwiGConfig

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: BlackJAXSwiGConfig,
        periodic: Optional[dict[int, tuple[float, float]]] = None,
        rebuild_required_by_block: Optional[dict[tuple[int, ...], bool]] = None,
        build_cache: Optional[Callable] = None,
        log_likelihood_from_cache_fn: Optional[Callable] = None,
    ) -> None:
        if (
            rebuild_required_by_block is None
            or build_cache is None
            or log_likelihood_from_cache_fn is None
        ):
            raise ValueError("BlackJAXSwiGSampler requires cache callbacks.")
        if periodic is not None and not isinstance(periodic, dict):
            raise TypeError("Cache-aware sampling requires dict-form periodic bounds.")

        nss_config = BlackJAXNSSConfig(
            n_live=config.n_live,
            n_delete_frac=config.n_delete_frac,
            num_inner_steps_per_dim=config.num_inner_steps_per_dim,
            termination_dlogz=config.termination_dlogz,
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_interval=config.checkpoint_interval,
        )
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=nss_config,
            periodic=periodic,
        )
        self._swig_config = config
        self._rebuild_required_by_block = rebuild_required_by_block
        self._build_cache = build_cache
        self._log_likelihood_from_cache_fn = log_likelihood_from_cache_fn
        self._periodic = periodic

        def update_block_covariances(rng_key, state, info, params=None):
            del rng_key, info, params
            covariance = jnp.atleast_2d(
                particles_covariance_matrix(state.particles.position)
            )
            covariances = tuple(
                covariance[
                    jnp.ix_(
                        jnp.asarray(parameter_indices),
                        jnp.asarray(parameter_indices),
                    )
                ]
                for parameter_indices in self._rebuild_required_by_block
            )
            return {"block_covariances": covariances}

        self._update_block_covariances = update_block_covariances

    @property
    def sampler_name(self) -> str:
        return "BlackJAX SwiG"

    @property
    def _update_inner_kernel_params_fn(self) -> Callable:
        return self._update_block_covariances

    def _build_nested_sampler(self, n_delete: int) -> SamplingAlgorithm:
        constrained_step = _build_swig_constrained_step(
            log_prior_fn=self._log_prior_fn,
            build_cache=self._build_cache,
            log_likelihood_from_cache_fn=self._log_likelihood_from_cache_fn,
            rebuild_required_by_block=self._rebuild_required_by_block,
            num_gibbs_sweeps=self._swig_config.num_gibbs_sweeps,
            num_inner_steps_per_dim=self._swig_config.num_inner_steps_per_dim,
            max_steps=self._swig_config.max_steps,
            max_shrinkage=self._swig_config.max_shrinkage,
            periodic=self._periodic,
            n_dims=self.n_dims,
        )
        kernel = build_from_mcmc_kernel(
            constrained_step,
            num_inner_steps=1,
            update_inner_kernel_params_fn=self._update_block_covariances,
            num_delete=n_delete,
        )

        def initialize(position: Position, rng_key: Optional[PRNGKey]) -> State:
            del rng_key
            # ``build_from_mcmc_kernel`` initializes directly from a position,
            # although BlackJAX types every initializer as returning a state.
            return position  # type: ignore[return-value]

        return SamplingAlgorithm(initialize, kernel)
