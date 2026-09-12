"""BlackJAX Nested Slice Sampling (NSS)."""

import time
from collections.abc import Callable
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from anesthetic.samples import NestedSamples
from blackjax import SamplingAlgorithm, nss
from blackjax.mcmc.slice import build_kernel as build_slice_kernel
from blackjax.mcmc.slice import stepping_out
from blackjax.ns.adaptive import AdaptiveNSState
from blackjax.ns.adaptive import init as _ns_adaptive_init
from blackjax.ns.base import NSInfo
from blackjax.ns.base import init_state_strategy as _init_state_strategy
from blackjax.ns.nss import (
    live_covariance,
    sample_direction_from_covariance,
    slice_constrained_step,
)
from blackjax.ns.utils import finalise
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Key

from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax.sharding import (
    _LIVE_AXIS,
    build_sharded_from_mcmc_kernel,
    make_live_mesh,
    place_key,
    place_state,
)
from jimgw.samplers.blackjax.utils import (
    load_or_initialize_checkpoint,
    prepare_checkpointing,
    remove_checkpoint_and_jax_cache,
    save_checkpoint_if_due,
)
from jimgw.samplers.config import BlackJAXNSSConfig
from jimgw.samplers.periodic import to_prior_space_proposal


class BlackJAXNSSSampler(Sampler):
    """BlackJAX Nested Slice Sampler (NSS).

    NSS combines nested sampling with an adaptive slice-sampling inner kernel.
    It works directly in the sampling space defined by ``sample_transforms``
    (no unit-cube constraint required).  Operates on flat arrays of shape
    ``(n_dims,)``; the NSS kernel is pytree-generic.

    Configure via [`BlackJAXNSSConfig`][jimgw.samplers.config.BlackJAXNSSConfig].

    Args:
        n_dims: Dimension of the sampling space.
        log_prior_fn: Log-prior callable ``(arr,) -> float``.
        log_likelihood_fn: Log-likelihood callable ``(arr,) -> float``.
        log_posterior_fn: Log-posterior callable ``(arr,) -> float``.
        config: Optional ``BlackJAXNSSConfig``; defaults to all-default values.
        periodic: Optional periodic-parameter spec in index space,
            ``dict[int, (lo, hi)]`` where the key is the dimension index and
            the value is the ``(lower, upper)`` period bounds.  ``None`` means
            no periodic parameters.  Provided by Jim after resolving names.
    """

    _config: BlackJAXNSSConfig
    _proposal: Callable
    _final_state: NSInfo
    _nested_samples: NestedSamples
    _n_iterations: int

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: Optional[BlackJAXNSSConfig] = None,
        periodic: Optional[dict[int, tuple[float, float]]] = None,
    ) -> None:
        if config is None:
            config = BlackJAXNSSConfig()
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )
        self._proposal = to_prior_space_proposal(
            periodic, n_dims, sample_direction_from_covariance
        )

    @property
    def sampler_name(self) -> str:
        return "BlackJAX NSS"

    @property
    def _update_inner_kernel_params_fn(self) -> Callable:
        return live_covariance

    def _build_nested_sampler(self, n_delete: int, mesh: Optional[Mesh] = None):
        config = self._config
        num_inner_steps = config.num_inner_steps_per_dim * self.n_dims
        if mesh is not None:
            init_state_fn = partial(
                _init_state_strategy,
                logprior_fn=self._log_prior_fn,
                loglikelihood_fn=self._log_likelihood_fn,
            )
            slice_kernel = build_slice_kernel(
                interval=stepping_out,
                max_expansions=10,
                max_shrinkage=100,
            )
            constrained_step = slice_constrained_step(
                init_state_fn, slice_kernel, self._proposal
            )
            kernel = build_sharded_from_mcmc_kernel(
                constrained_step,
                n_inner_steps=num_inner_steps,
                update_inner_kernel_params_fn=self._update_inner_kernel_params_fn,
                n_delete=n_delete,
                mesh=mesh,
            )
            # `_sample` initializes state directly; BlackJAX still requires an init callable.
            return SamplingAlgorithm(
                lambda position, rng_key=None: position,  # type: ignore[return-value]
                kernel,
            )
        return nss(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            num_delete=n_delete,
            num_inner_steps=num_inner_steps,
            proposal=self._proposal,
        )

    def _sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_live n_dims"],
    ) -> None:
        """Run the BlackJAX NSS sampler.

        If ``config.checkpoint_dir`` is set, a ``checkpoint.pkl`` is written
        atomically after each nested-sampling iteration (subject to
        ``config.checkpoint_interval``) and the sampler resumes from the
        checkpoint if one already exists at that path.

        Args:
            rng_key: JAX PRNG key.
            initial_position: Starting live points in the sampling space,
                shape ``(n_live, n_dims)``.  Must match ``config.n_live``.
                Ignored when resuming from a checkpoint.

        Raises:
            ValueError: If ``initial_position`` shape does not match
                ``(n_live, n_dims)``.
        """
        config = self._config
        n_live = config.n_live
        n_delete = int(n_live * config.n_delete_frac)
        mesh = make_live_mesh(config.n_devices, n_live, n_delete)
        checkpoint_path, run_start_time = prepare_checkpointing(config)

        def validate_initial_particles(pos):
            arr = jnp.asarray(pos)
            if arr.ndim != 2 or arr.shape != (n_live, self.n_dims):
                raise ValueError(
                    f"initial_position must have shape ({n_live}, {self.n_dims}), "
                    f"got {arr.shape}."
                )
            return arr

        nested_sampler = self._build_nested_sampler(n_delete, mesh)

        initialize_single_particle = partial(
            _init_state_strategy,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
        )

        def initialize_nested_sampler_state(positions):
            """Initialize particle states in batches to bound peak GPU memory.

            A full ``vmap`` over all live particles materializes concurrent
            intermediate buffers. ``lax.map`` limits the batch to ``n_delete``
            particles without adding computation.
            """
            if mesh is not None:
                positions = jax.device_put(
                    positions, NamedSharding(mesh, P(_LIVE_AXIS))
                )

            def initialize_particle_batch(pos):
                return jax.lax.map(initialize_single_particle, pos, batch_size=n_delete)

            state = _ns_adaptive_init(
                positions,
                init_state_fn=initialize_particle_batch,
                update_inner_kernel_params_fn=self._update_inner_kernel_params_fn,
            )
            return place_state(state, mesh) if mesh is not None else state

        state, rng_key, n_completed_iterations, extra = load_or_initialize_checkpoint(
            self,
            checkpoint_path,
            config,
            rng_key,
            initial_position,
            lambda position: initialize_nested_sampler_state(
                validate_initial_particles(position)
            ),
            initial_extra={"dead": []},
            load_extra=lambda checkpoint: {"dead": checkpoint["dead"]},
            log_label=self.sampler_name,
            after_load=lambda state: (
                place_state(state, mesh) if mesh is not None else state
            ),
        )
        dead = extra["dead"]

        if mesh is not None:
            rng_key = place_key(rng_key, mesh)

        def should_terminate(state: AdaptiveNSState) -> bool:
            dlogz = jnp.logaddexp(0, state.integrator.logZ_live - state.integrator.logZ)
            return bool(jnp.isfinite(dlogz) and dlogz < config.termination_dlogz)

        step_fn = jax.jit(nested_sampler.step)
        last_checkpoint_write_time = time.perf_counter()

        while not should_terminate(state):
            rng_key, step_key = jax.random.split(rng_key)
            state, dead_info = step_fn(step_key, state)
            dead.append(dead_info)
            n_completed_iterations += 1
            last_checkpoint_write_time = save_checkpoint_if_due(
                self,
                config,
                checkpoint_path,
                last_checkpoint_write_time,
                run_start_time,
                state,
                rng_key,
                n_completed_iterations,
                {"dead": dead},
                log_label=self.sampler_name,
                before_save=lambda state, rng_key, extra: (
                    jax.device_get(state),
                    jax.device_get(rng_key),
                    jax.device_get(extra),
                ),
            )

        final_state = finalise(
            state, dead
        )  # AdaptiveNSState structurally satisfies NSState (.particles field)
        self._final_state = jax.device_get(final_state)
        self._n_iterations = n_completed_iterations

        # Build anesthetic NestedSamples for use in get_samples() and get_diagnostics().
        particles_sample = np.array(self._final_state.particles.position)
        log_likelihood = np.array(self._final_state.particles.loglikelihood)
        logL_birth = np.array(self._final_state.particles.loglikelihood_birth)
        logL_birth = np.where(np.isnan(logL_birth), -np.inf, logL_birth)
        self._nested_samples = NestedSamples(
            particles_sample,
            logL=log_likelihood,
            logL_birth=logL_birth,
            logzero=np.nan,
            dtype=np.float64,
        )
        remove_checkpoint_and_jax_cache(config, checkpoint_path)

    def get_samples(self) -> dict[str, np.ndarray]:
        """Return equally-weighted posterior samples via anesthetic's ``posterior_points``.

        Uses `NestedSamples.posterior_points` to
        resample the nested dead-point collection to a set of truly equal-weight
        samples (rows duplicated proportional to integer weights).

        Returns:
            Dict with keys ``"samples"`` (shape ``(n, n_dims)``) and
            ``"log_likelihood"`` (shape ``(n,)``).
        """
        if not self._sampled:
            raise RuntimeError("get_samples() called before sample()")
        posterior = self._nested_samples.posterior_points()
        samples = np.asarray(posterior.iloc[:, : self.n_dims])
        log_L = np.asarray(posterior["logL"])
        return {"samples": samples, "log_likelihood": log_L}

    def _get_diagnostics(self) -> dict[str, Any]:
        """Return NSS run diagnostics.

        Returns a dict with the following keys:

        * ``"n_likelihood_evaluations"`` — total likelihood calls.
        * ``"n_iterations"`` — total nested-sampling iterations.
        * ``"n_stepping_out_history"`` — stepping-out evaluations per iteration.
        * ``"n_shrinking_history"`` — shrinking evaluations per iteration.
        * ``"n_likelihood_evaluations_stepping_out"`` — total stepping-out evaluations.
        * ``"n_likelihood_evaluations_shrinking"`` — total shrinking evaluations.
        * ``"acceptance_history"`` — per-iteration acceptance flag.
        * ``"log_Z"`` — log Bayesian evidence (anesthetic mean estimate).
        * ``"log_Z_error"`` — standard deviation of log Z from 100 bootstrap samples.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")
        ui: Any = (
            self._final_state.update_info
        )  # SliceInfo — blackjax stubs type this as base NamedTuple
        total_steps = int(jnp.sum(ui.num_expansions))
        total_shrink = int(jnp.sum(ui.num_shrink))

        log_Z = np.asarray(self._nested_samples.logZ()).item()
        log_Z_error = np.std(np.asarray(self._nested_samples.logZ(nsamples=100))).item()

        return {
            "n_likelihood_evaluations": total_steps + total_shrink,
            "n_iterations": self._n_iterations,
            "n_stepping_out_history": np.asarray(ui.num_expansions),
            "n_shrinking_history": np.asarray(ui.num_shrink),
            "n_likelihood_evaluations_stepping_out": total_steps,
            "n_likelihood_evaluations_shrinking": total_shrink,
            "acceptance_history": np.asarray(ui.is_accepted),
            "log_Z": log_Z,
            "log_Z_error": log_Z_error,
        }
