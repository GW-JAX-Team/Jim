"""BlackJAX Nested Slice Sampling (NSS)."""

import logging
import pickle
import shutil
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
from jimgw.samplers.config import BlackJAXNSSConfig
from jimgw.samplers.periodic import to_prior_space_proposal

logger = logging.getLogger(__name__)


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
            # `nested_sampler.init` is never called (state init happens in
            # `_batched_nss_init`); BlackJAX still requires SamplingAlgorithm.init
            # to type as returning a State.
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
        ckpt_path = (
            config.checkpoint_dir / "checkpoint.pkl"
            if config.checkpoint_dir is not None
            else None
        )
        config.configure_jax_cache()
        _method_t0 = time.perf_counter()

        def _validated_initial_particles(pos):
            arr = jnp.asarray(pos)
            if arr.ndim != 2 or arr.shape != (n_live, self.n_dims):
                raise ValueError(
                    f"initial_position must have shape ({n_live}, {self.n_dims}), "
                    f"got {arr.shape}."
                )
            return arr

        nested_sampler = self._build_nested_sampler(n_delete, mesh)

        # Bypass BlackJAX's jax.vmap(init_state_fn) to avoid peak-memory OOM.
        # A full vmap over all live particles materialises O(n_live) concurrent
        # intermediate buffers, which can exceed available GPU memory for expensive
        # likelihoods. lax.map with n_delete particles per batch bounds peak memory
        # to n_delete/n_live of the full-vmap cost at no extra computation.
        _single_init_fn = partial(
            _init_state_strategy,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
        )

        def _batched_nss_init(positions):
            if mesh is not None:
                positions = jax.device_put(
                    positions, NamedSharding(mesh, P(_LIVE_AXIS))
                )

            def _batched_fn(pos):
                return jax.lax.map(_single_init_fn, pos, batch_size=n_delete)

            state = _ns_adaptive_init(
                positions,
                init_state_fn=_batched_fn,
                update_inner_kernel_params_fn=self._update_inner_kernel_params_fn,
            )
            return place_state(state, mesh) if mesh is not None else state

        # Resume from checkpoint if one exists.
        if (
            ckpt_path is not None
            and config.checkpoint_interval > 0
            and ckpt_path.exists()
        ):
            _initial_rng_key = rng_key
            try:
                with open(ckpt_path, "rb") as _f:
                    _ckpt = pickle.load(_f)
                self._validate_checkpoint(_ckpt)
                state = _ckpt["state"]
                dead = _ckpt["dead"]
                rng_key = _ckpt["rng_key"]
                if mesh is not None:
                    state = place_state(state, mesh)
                    rng_key = place_key(rng_key, mesh)
                n_iter = _ckpt["n_iter"]
                self._prev_elapsed = float(_ckpt["elapsed_time"])
                logger.info(
                    "%s: resumed from checkpoint at n_iter=%d (%s)",
                    self.sampler_name,
                    n_iter,
                    ckpt_path,
                )
            except (
                OSError,
                EOFError,
                KeyError,
                TypeError,
                ValueError,
                pickle.UnpicklingError,
            ) as _e:
                logger.warning(
                    "%s: incompatible or corrupt checkpoint at %s (%s) — starting fresh.",
                    self.sampler_name,
                    ckpt_path,
                    _e,
                )
                rng_key = _initial_rng_key
                state = _batched_nss_init(
                    _validated_initial_particles(initial_position)
                )
                dead = []
                n_iter = 0
                self._prev_elapsed = 0.0
        else:
            state = _batched_nss_init(_validated_initial_particles(initial_position))
            dead = []
            n_iter = 0

        if mesh is not None:
            rng_key = place_key(rng_key, mesh)

        def _terminate(state: AdaptiveNSState) -> bool:
            dlogz = jnp.logaddexp(0, state.integrator.logZ_live - state.integrator.logZ)
            return bool(jnp.isfinite(dlogz) and dlogz < config.termination_dlogz)

        step_fn = jax.jit(nested_sampler.step)
        _last_ckpt_t = time.perf_counter()

        while not _terminate(state):
            rng_key, subkey = jax.random.split(rng_key)
            state, dead_info = step_fn(subkey, state)
            dead.append(dead_info)
            n_iter += 1
            if (
                ckpt_path is not None
                and config.checkpoint_interval > 0
                and time.perf_counter() - _last_ckpt_t >= config.checkpoint_interval
            ):
                _last_ckpt_t = config.write_checkpoint(
                    {
                        "state": jax.device_get(state),
                        "dead": jax.device_get(dead),
                        "rng_key": jax.device_get(rng_key),
                        "n_iter": n_iter,
                        "sampler_name": self.sampler_name,
                        "elapsed_time": self._prev_elapsed
                        + (time.perf_counter() - _method_t0),
                    },
                    self.sampler_name,
                )

        final_state = finalise(state, dead)  # type: ignore[arg-type]  # AdaptiveNSState structurally satisfies NSState (.particles field)
        self._final_state = jax.device_get(final_state)
        self._n_iterations = n_iter

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
        if ckpt_path is not None:
            ckpt_path.unlink(missing_ok=True)
        if config.checkpoint_dir is not None:
            shutil.rmtree(config.checkpoint_dir / "jax_cache", ignore_errors=True)
            jax.config.update("jax_compilation_cache_dir", None)

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
