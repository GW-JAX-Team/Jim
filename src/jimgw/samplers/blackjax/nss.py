"""BlackJAX Nested Slice Sampling (NSS)."""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from anesthetic.samples import NestedSamples
from jaxtyping import Array, Float, Key

import blackjax
from blackjax.ns.utils import finalise
from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax._imports import (
    require_nested_sampling,
    require_nss,
)
from jimgw.samplers.config import BlackJAXNSSConfig
from jimgw.samplers.periodic import to_prior_space_stepper

logger = logging.getLogger(__name__)

require_nested_sampling(blackjax)
require_nss(blackjax)


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
    _stepper_fn: Callable
    _final_state: Any
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
        self._stepper_fn = to_prior_space_stepper(periodic, n_dims)

    def _sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_live n_dims"],
    ) -> None:
        """Run the BlackJAX NSS sampler.

        If ``config.checkpoint_path`` is set, a ``.pkl`` checkpoint is written
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
        num_inner_steps = config.num_inner_steps_per_dim * self.n_dims

        arr = jnp.asarray(initial_position)
        if arr.ndim != 2 or arr.shape != (n_live, self.n_dims):
            raise ValueError(
                f"initial_position must have shape ({n_live}, {self.n_dims}), "
                f"got {arr.shape}."
            )
        initial_particles = arr

        nested_sampler = blackjax.nss(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            num_delete=n_delete,
            num_inner_steps=num_inner_steps,
            stepper_fn=self._stepper_fn,
        )

        ckpt_path: Optional[Path] = config.checkpoint_path

        # Resume from checkpoint if one exists.
        if ckpt_path is not None and ckpt_path.exists():
            try:
                with open(ckpt_path, "rb") as _f:
                    _ckpt = pickle.load(_f)
                state = _ckpt["state"]
                dead = _ckpt["dead"]
                rng_key = _ckpt["rng_key"]
                n_iter = _ckpt["n_iter"]
                logger.info(
                    "NSS: resumed from checkpoint at n_iter=%d (%s)", n_iter, ckpt_path
                )
            except Exception as _e:
                logger.warning(
                    "NSS: corrupt checkpoint at %s (%s) — starting fresh.",
                    ckpt_path,
                    _e,
                )
                state = nested_sampler.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                dead = []
                n_iter = 0
        else:
            state = nested_sampler.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
            dead = []
            n_iter = 0

        def _terminate(state: Any) -> bool:
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
                and time.perf_counter() - _last_ckpt_t >= config.checkpoint_interval
            ):
                _ckpt_data = {
                    "state": state,
                    "dead": dead,
                    "rng_key": rng_key,
                    "n_iter": n_iter,
                }
                _tmp = ckpt_path.with_suffix(".pkl.tmp")
                with open(_tmp, "wb") as _f:
                    pickle.dump(_ckpt_data, _f)
                _tmp.replace(ckpt_path)
                _last_ckpt_t = time.perf_counter()
                logger.debug("NSS: checkpoint saved at n_iter=%d", n_iter)

        self._final_state = finalise(state, dead)
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
        ui = self._final_state.update_info  # SliceInfo concatenated across all steps
        total_steps = int(jnp.sum(ui.num_steps))
        total_shrink = int(jnp.sum(ui.num_shrink))

        log_Z = np.asarray(self._nested_samples.logZ()).item()
        log_Z_error = np.std(np.asarray(self._nested_samples.logZ(nsamples=100))).item()

        return {
            "n_likelihood_evaluations": total_steps + total_shrink,
            "n_iterations": self._n_iterations,
            "n_stepping_out_history": np.asarray(ui.num_steps),
            "n_shrinking_history": np.asarray(ui.num_shrink),
            "n_likelihood_evaluations_stepping_out": total_steps,
            "n_likelihood_evaluations_shrinking": total_shrink,
            "acceptance_history": np.asarray(ui.is_accepted),
            "log_Z": log_Z,
            "log_Z_error": log_Z_error,
        }
