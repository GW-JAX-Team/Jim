"""BlackJAX SMC samplers for Jim.

Supports four mode combinations selected by
[`BlackJAXSMCConfig`][jimgw.samplers.config.BlackJAXSMCConfig]:

* ``persistent_sampling=True,  temperature_ladder=None``  → adaptive persistent SMC
* ``persistent_sampling=True,  temperature_ladder=given`` → fixed-ladder persistent SMC
* ``persistent_sampling=False, temperature_ladder=None``  → adaptive tempered SMC
* ``persistent_sampling=False, temperature_ladder=given`` → fixed-ladder tempered SMC
"""

import logging
import pickle
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from blackjax import (
    adaptive_persistent_sampling_smc,
    adaptive_tempered_smc,
    inner_kernel_tuning,
    persistent_sampling_smc,
    rmh,
    tempered_smc,
)
from blackjax.mcmc import random_walk
from blackjax.smc import extend_params, persistent_sampling, tempered
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.smc.persistent_sampling import (
    compute_log_persistent_weights,
    compute_persistent_ess,
)
from blackjax.smc.resampling import systematic
from jaxtyping import Array, Float, Key

from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax._de_move import (
    de_proposal_scale,
    sample_de_gamma,
    sample_two_distinct_indices,
)
from jimgw.samplers.config import BlackJAXSMCConfig
from jimgw.samplers.periodic import to_displacement_wrapper

logger = logging.getLogger(__name__)

# Fixed key used for post-sampling resampling in get_samples().
_RESAMPLE_KEY = jax.random.key(123)


class BlackJAXSMCSampler(Sampler):
    """BlackJAX SMC sampler.

    The inner kernel is set by ``config.inner_kernel``: ``"GRW"`` (default) is a
    Gaussian random walk with adaptive covariance; ``"DE"`` uses the
    differential-evolution proposal shared with the NS acceptance-walk kernel.

    Supports checkpoint/resume via ``config.checkpoint_dir``: a ``checkpoint.pkl``
    checkpoint is written atomically after each tempering iteration (subject
    to ``config.checkpoint_interval``) and the sampler resumes from it if one
    already exists at that path.

    Operates on flat ``(n_dims,)`` arrays.

    Args:
        n_dims: Dimension of the sampling space.
        log_prior_fn: Log-prior callable ``(arr,) -> float``.
        log_likelihood_fn: Log-likelihood callable ``(arr,) -> float``.
        log_posterior_fn: Log-posterior callable ``(arr,) -> float``.
        config: Optional ``BlackJAXSMCConfig``; defaults to all-default values.
        periodic: Optional periodic-parameter spec in index space,
            ``dict[int, (lo, hi)]`` where the key is the dimension index and
            the value is the ``(lower, upper)`` period bounds.  ``None`` means
            no periodic parameters.  Provided by Jim after resolving names.
    """

    _config: BlackJAXSMCConfig
    _displacement_wrapper: Callable
    _final_state: Any
    _n_iterations: int
    # Per-mode diagnostics stashes (set in the corresponding _run_* method).
    _acceptance_history: (
        np.ndarray
    )  # adaptive/persistent: per-step mean acceptance rate
    _cov_scale_history: (
        np.ndarray
    )  # persistent adaptive only: per-step covariance scale
    _tempering_schedule: np.ndarray  # adaptive tempered only: per-step temperature
    _is_weights_history: (
        np.ndarray
    )  # tempered (non-persistent) modes: per-step IS weights

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: Optional[BlackJAXSMCConfig] = None,
        periodic: Optional[dict[int, tuple[float, float]]] = None,
    ) -> None:
        if config is None:
            config = BlackJAXSMCConfig()
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )
        self._displacement_wrapper = to_displacement_wrapper(periodic, n_dims)

    @property
    def sampler_name(self) -> str:
        return "BlackJAX SMC"

    @property
    def mode(self) -> str:
        """SMC implementation selected by the sampler configuration."""
        if self._config.persistent_sampling:
            return "fp" if self._config.temperature_ladder is not None else "ap"
        return "ft" if self._config.temperature_ladder is not None else "at"

    def _validate_checkpoint(self, checkpoint: dict) -> None:
        """Raise when a checkpoint is incompatible with this SMC configuration."""
        super()._validate_checkpoint(checkpoint)
        checkpoint_mode = checkpoint.get("mode")
        if checkpoint_mode != self.mode:
            raise ValueError(
                "checkpoint belongs to a different SMC mode: "
                f"{checkpoint_mode or 'an unknown mode'}, not {self.mode}"
            )
        checkpoint_kernel = checkpoint.get("inner_kernel")
        if checkpoint_kernel != self._config.inner_kernel:
            raise ValueError(
                "checkpoint belongs to a different SMC inner kernel: "
                f"{checkpoint_kernel}, not {self._config.inner_kernel}"
            )

    def _build_inner_kernel_step(self):
        """Build the transition step for the configured inner kernel.

        The returned function accepts ``(key, state, logdensity, parameter)``.
        Its parameter is a covariance matrix for GRW or a reference ensemble for
        differential evolution.
        """
        displacement_wrapper = self._displacement_wrapper
        additive_step = random_walk.build_additive_step()

        if self._config.inner_kernel == "GRW":

            def grw_step(random_key, state, logdensity, cov):
                covariance = cov

                def proposal_distribution(proposal_key, position):
                    raw_disp = jax.random.multivariate_normal(
                        proposal_key, jnp.zeros_like(position), covariance
                    )
                    return displacement_wrapper(raw_disp, position)

                return additive_step(
                    random_key, state, logdensity, proposal_distribution
                )

            return grw_step

        proposal_scale = de_proposal_scale(self.n_dims)
        small_step_probability = 0.5

        def de_step(random_key, state, logdensity, ensemble):
            reference_ensemble = ensemble
            n_particles = reference_ensemble.shape[0]

            def proposal_distribution(proposal_key, position):
                pair_key, multiplier_key = jax.random.split(proposal_key)
                first_index, second_index = sample_two_distinct_indices(
                    pair_key, n_particles
                )
                ensemble_difference = (
                    reference_ensemble[first_index] - reference_ensemble[second_index]
                )
                multiplier = sample_de_gamma(
                    multiplier_key, small_step_probability, proposal_scale
                )
                return displacement_wrapper(multiplier * ensemble_difference, position)

            return additive_step(random_key, state, logdensity, proposal_distribution)

        return de_step

    def _build_adaptive_inner_kernel_parameters(self, initial_particles):
        """Build the parameter updater and initial values for adaptive SMC.

        GRW receives a covariance estimated from the particle cloud. Differential
        evolution receives that cloud itself as its proposal reference ensemble.
        """
        if self._config.inner_kernel == "DE":

            def update_ensemble_parameters(_key, smc_state, _info):
                return extend_params({"ensemble": smc_state.particles})  # type: ignore[arg-type]

            return update_ensemble_parameters, extend_params(
                {"ensemble": initial_particles}  # type: ignore[arg-type]
            )

        initial_covariance = (
            jnp.atleast_2d(jnp.cov(initial_particles.T))
            * self._config.grw.initial_cov_scale
        )

        def update_covariance_parameters(_key, smc_state, _info):
            covariance = jnp.atleast_2d(jnp.cov(smc_state.particles.T))
            return extend_params({"cov": covariance})  # type: ignore[arg-type]

        return update_covariance_parameters, extend_params(
            {"cov": initial_covariance}  # type: ignore[arg-type]
        )

    def _build_fixed_ladder_smc(
        self,
        uses_persistent_sampling: bool,
        inner_kernel_step,
        initial_particles,
        n_temperature_steps: int,
        n_mcmc_steps: int,
    ):
        """Build initialization and transition functions for a fixed temperature ladder.

        GRW uses BlackJAX's high-level SMC algorithms. The DE path builds the
        underlying transition directly so each step can use the latest particle
        cloud as its proposal reference ensemble.
        """
        config = self._config

        if config.inner_kernel == "DE":
            build_smc_transition = (
                persistent_sampling.build_kernel
                if uses_persistent_sampling
                else tempered.build_kernel
            )
            smc_transition = build_smc_transition(
                logprior_fn=self._log_prior_fn,
                loglikelihood_fn=self._log_likelihood_fn,
                mcmc_step_fn=inner_kernel_step,
                mcmc_init_fn=rmh.init,
                resampling_fn=systematic,
                batch_size=config.batch_size,
            )

            def initialize_de_state(particles):
                if uses_persistent_sampling:
                    return persistent_sampling.init(
                        particles,
                        self._log_likelihood_fn,
                        n_temperature_steps,
                        batch_size=config.batch_size,
                    )
                return tempered.init(particles)

            def step_with_current_ensemble(random_key, state, tempering_parameter):
                return smc_transition(
                    random_key,
                    state,
                    n_mcmc_steps,
                    tempering_parameter,
                    extend_params({"ensemble": state.particles}),  # type: ignore[arg-type]
                )

            return initialize_de_state, step_with_current_ensemble

        initial_covariance = (
            jnp.atleast_2d(jnp.cov(initial_particles.T)) * config.grw.initial_cov_scale
        )
        smc_kwargs = {
            "logprior_fn": self._log_prior_fn,
            "loglikelihood_fn": self._log_likelihood_fn,
            "mcmc_step_fn": inner_kernel_step,
            "mcmc_init_fn": rmh.init,
            "mcmc_parameters": extend_params({"cov": initial_covariance}),  # type: ignore[arg-type]
            "resampling_fn": systematic,
            "num_mcmc_steps": n_mcmc_steps,
            "batch_size": config.batch_size,
        }
        smc_algorithm = (
            persistent_sampling_smc(n_schedule=n_temperature_steps, **smc_kwargs)
            if uses_persistent_sampling
            else tempered_smc(**smc_kwargs)
        )
        return smc_algorithm.init, smc_algorithm.step

    def _checkpoint_safe_state(self, config: BlackJAXSMCConfig, state: Any) -> Any:
        """Drop the DE reference ensemble from ``state`` before checkpointing.

        For ``inner_kernel="DE"`` in the adaptive (inner-kernel-tuning) modes,
        ``state.parameter_override["ensemble"]`` is always an exact copy of
        ``state.sampler_state.particles`` — see
        ``_build_adaptive_inner_kernel_parameters``'s ``update_ensemble_parameters``,
        which sets it to ``extend_params({"ensemble": smc_state.particles})`` every
        step. Persisting it doubles the particle population's footprint in every
        checkpoint file for no benefit, since ``_restore_checkpoint_ensemble``
        recomputes it from ``sampler_state.particles`` on resume.
        """
        if config.inner_kernel == "DE" and isinstance(
            state, StateWithParameterOverride
        ):
            return state._replace(
                parameter_override={
                    key: value
                    for key, value in state.parameter_override.items()
                    if key != "ensemble"
                }
            )
        return state

    def _restore_checkpoint_ensemble(
        self, config: BlackJAXSMCConfig, state: Any
    ) -> Any:
        """Reconstruct the DE ensemble dropped by ``_checkpoint_safe_state``."""
        if (
            config.inner_kernel == "DE"
            and isinstance(state, StateWithParameterOverride)
            and "ensemble" not in state.parameter_override
        ):
            return state._replace(
                parameter_override={
                    **state.parameter_override,
                    **extend_params(
                        {"ensemble": state.sampler_state.particles}  # type: ignore[arg-type]
                    ),
                }
            )
        return state

    def _load_or_initialize_state(
        self,
        checkpoint_path: Optional[Path],
        config: BlackJAXSMCConfig,
        rng_key: Key,
        initial_particles,
        initialize_state: Callable,
        initial_mode_data: dict[str, Any],
        load_mode_data: Callable[[dict], dict[str, Any]],
        n_temperature_steps: Optional[int] = None,
    ) -> tuple[Any, Key, int, dict[str, Any]]:
        """Load a compatible checkpoint or initialize a new SMC state.

        ``load_mode_data`` restores histories that are specific to each SMC
        mode. ``initial_mode_data`` is used when no usable checkpoint exists or
        when a fixed-ladder checkpoint exceeds the current number of temperature
        steps. Only trusted checkpoint files may be loaded because pickle can
        execute arbitrary code.
        """
        if not (
            checkpoint_path is not None
            and config.checkpoint_interval > 0
            and checkpoint_path.exists()
        ):
            self._prev_elapsed = 0.0
            return (
                initialize_state(initial_particles),
                rng_key,
                0,
                dict(initial_mode_data),
            )

        initial_rng_key = rng_key
        try:
            with open(checkpoint_path, "rb") as checkpoint_file:
                checkpoint = pickle.load(checkpoint_file)
            self._validate_checkpoint(checkpoint)
            state = checkpoint["state"]
            rng_key = checkpoint["rng_key"]
            n_completed_iterations = checkpoint["n_iter"]
            mode_data = load_mode_data(checkpoint)
            if (
                n_temperature_steps is not None
                and n_completed_iterations > n_temperature_steps
            ):
                logger.warning(
                    "%s: checkpoint n_iter=%d exceeds current schedule length=%d — starting fresh.",
                    f"{self.sampler_name} ({self.mode.upper()})",
                    n_completed_iterations,
                    n_temperature_steps,
                )
                rng_key = initial_rng_key
                state = initialize_state(initial_particles)
                n_completed_iterations = 0
                mode_data = dict(initial_mode_data)
                self._prev_elapsed = 0.0
            else:
                self._prev_elapsed = float(checkpoint["elapsed_time"])
                state = self._restore_checkpoint_ensemble(config, state)
                logger.info(
                    "%s: resumed from checkpoint at n_iter=%d (%s)",
                    f"{self.sampler_name} ({self.mode.upper()})",
                    n_completed_iterations,
                    checkpoint_path,
                )
            return state, rng_key, n_completed_iterations, mode_data
        except (
            OSError,
            EOFError,
            KeyError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as error:
            logger.warning(
                "%s: incompatible or corrupt checkpoint at %s (%s) — starting fresh.",
                f"{self.sampler_name} ({self.mode.upper()})",
                checkpoint_path,
                error,
            )
            self._prev_elapsed = 0.0
            return (
                initialize_state(initial_particles),
                initial_rng_key,
                0,
                dict(initial_mode_data),
            )

    def _prepare_checkpointing(
        self, config: BlackJAXSMCConfig
    ) -> tuple[Optional[Path], float]:
        """Enable the JAX compile cache and return the checkpoint path and start time."""
        checkpoint_path = (
            config.checkpoint_dir / "checkpoint.pkl"
            if config.checkpoint_dir is not None
            else None
        )
        config.configure_jax_cache()
        return checkpoint_path, time.perf_counter()

    def _save_checkpoint_if_due(
        self,
        config: BlackJAXSMCConfig,
        checkpoint_path: Optional[Path],
        last_checkpoint_at: float,
        run_started_at: float,
        state: Any,
        rng_key: Key,
        n_completed_iterations: int,
        mode_data: dict[str, Any],
    ) -> float:
        """Save a checkpoint when its interval has elapsed.

        ``mode_data`` contains mode-specific histories to merge into the common
        checkpoint data. Returns the last checkpoint time.
        """
        if not (
            checkpoint_path is not None
            and config.checkpoint_interval > 0
            and time.perf_counter() - last_checkpoint_at >= config.checkpoint_interval
        ):
            return last_checkpoint_at
        return config.write_checkpoint(
            {
                "state": self._checkpoint_safe_state(config, state),
                "rng_key": rng_key,
                "n_iter": n_completed_iterations,
                "mode": self.mode,
                "inner_kernel": config.inner_kernel,
                "sampler_name": self.sampler_name,
                "elapsed_time": self._prev_elapsed
                + (time.perf_counter() - run_started_at),
                **mode_data,
            },
            f"{self.sampler_name} ({self.mode.upper()})",
        )

    def _remove_checkpoint_and_jax_cache(
        self, config: BlackJAXSMCConfig, checkpoint_path: Optional[Path]
    ) -> None:
        """Remove a completed run's checkpoint, JAX cache, and cache setting."""
        if checkpoint_path is not None:
            checkpoint_path.unlink(missing_ok=True)
        if config.checkpoint_dir is not None:
            shutil.rmtree(config.checkpoint_dir / "jax_cache", ignore_errors=True)
            jax.config.update("jax_compilation_cache_dir", None)

    def _run_adaptive_persistent(self, rng_key: Key, initial_particles) -> None:
        """Run adaptive persistent SMC with the configured inner kernel."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config._resolve_target_ess_fraction()
        checkpoint_path, run_started_at = self._prepare_checkpointing(config)

        inner_kernel_step = self._build_inner_kernel_step()
        update_mcmc_parameters, initial_mcmc_parameters = (
            self._build_adaptive_inner_kernel_parameters(initial_particles)
        )

        smc_algorithm = inner_kernel_tuning(
            smc_algorithm=adaptive_persistent_sampling_smc,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            max_iterations=1000,
            mcmc_step_fn=inner_kernel_step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=update_mcmc_parameters,
            initial_parameter_value=initial_mcmc_parameters,
            num_mcmc_steps=n_mcmc_steps,
            target_ess=target_ess,
            batch_size=config.batch_size,
        )

        covariance_scale = float(config.grw.initial_cov_scale)
        state, rng_key, n_completed_iterations, mode_checkpoint_data = (
            self._load_or_initialize_state(
                checkpoint_path,
                config,
                rng_key,
                initial_particles,
                smc_algorithm.init,
                initial_mode_data={
                    "cov_scale": covariance_scale,
                    "accept_history": [],
                    "cov_scale_history": [],
                },
                load_mode_data=lambda checkpoint: {
                    "cov_scale": float(checkpoint.get("cov_scale", covariance_scale)),
                    "accept_history": list(checkpoint["accept_history"]),
                    "cov_scale_history": list(checkpoint["cov_scale_history"]),
                },
            )
        )
        covariance_scale = mode_checkpoint_data["cov_scale"]
        acceptance_history: list[float] = mode_checkpoint_data["accept_history"]
        covariance_scale_history: list[float] = mode_checkpoint_data[
            "cov_scale_history"
        ]

        run_adaptive_step = jax.jit(smc_algorithm.step)
        last_checkpoint_at = time.perf_counter()

        while state.sampler_state.tempering_param < 1.0:  # type: ignore[attr-defined]
            rng_key, step_key = jax.random.split(rng_key)
            state, info = run_adaptive_step(step_key, state)

            acceptance_rate = float(info.update_info.acceptance_rate.mean())

            if config.inner_kernel == "GRW":
                sampler_state = state.sampler_state
                updated_covariance_scale = covariance_scale * float(
                    jnp.exp(
                        config.grw.scale_adaptation_gain
                        * (acceptance_rate - config.grw.target_acceptance_rate)
                    )
                )
                current_covariance = state.parameter_override["cov"]
                updated_parameters = extend_params(
                    {"cov": current_covariance[0] * updated_covariance_scale}  # type: ignore[arg-type]
                )
                state = StateWithParameterOverride(sampler_state, updated_parameters)  # type: ignore[arg-type]
                covariance_scale_history.append(updated_covariance_scale)
                covariance_scale = updated_covariance_scale

            acceptance_history.append(acceptance_rate)
            n_completed_iterations += 1

            last_checkpoint_at = self._save_checkpoint_if_due(
                config,
                checkpoint_path,
                last_checkpoint_at,
                run_started_at,
                state,
                rng_key,
                n_completed_iterations,
                {
                    "cov_scale": covariance_scale,
                    "accept_history": acceptance_history.copy(),
                    "cov_scale_history": covariance_scale_history.copy(),
                },
            )

        self._final_state = state
        self._n_iterations = n_completed_iterations
        self._acceptance_history = np.asarray(acceptance_history)
        self._cov_scale_history = np.asarray(covariance_scale_history)
        self._remove_checkpoint_and_jax_cache(config, checkpoint_path)

    def _run_fixed_persistent(
        self, rng_key: Key, initial_particles, ladder: list[float]
    ) -> None:
        """Run fixed-ladder persistent SMC with the configured inner kernel."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        tempering_parameters = ladder[1:]
        n_temperature_steps = len(tempering_parameters)
        checkpoint_path, run_started_at = self._prepare_checkpointing(config)

        inner_kernel_step = self._build_inner_kernel_step()
        initialize_state, temperature_step = self._build_fixed_ladder_smc(
            uses_persistent_sampling=True,
            inner_kernel_step=inner_kernel_step,
            initial_particles=initial_particles,
            n_temperature_steps=n_temperature_steps,
            n_mcmc_steps=n_mcmc_steps,
        )

        state, rng_key, n_completed_iterations, mode_checkpoint_data = (
            self._load_or_initialize_state(
                checkpoint_path,
                config,
                rng_key,
                initial_particles,
                initialize_state,
                initial_mode_data={"accept_history": []},
                load_mode_data=lambda checkpoint: {
                    "accept_history": list(checkpoint["accept_history"])
                },
                n_temperature_steps=n_temperature_steps,
            )
        )
        acceptance_history: list[float] = mode_checkpoint_data["accept_history"]

        run_temperature_step = jax.jit(temperature_step)
        last_checkpoint_at = time.perf_counter()

        for tempering_parameter in tempering_parameters[n_completed_iterations:]:
            rng_key, step_key = jax.random.split(rng_key)
            state, info = run_temperature_step(step_key, state, tempering_parameter)
            acceptance_history.append(float(info.update_info.acceptance_rate.mean()))
            n_completed_iterations += 1
            last_checkpoint_at = self._save_checkpoint_if_due(
                config,
                checkpoint_path,
                last_checkpoint_at,
                run_started_at,
                state,
                rng_key,
                n_completed_iterations,
                {"accept_history": acceptance_history.copy()},
            )

        self._final_state = state
        self._n_iterations = n_temperature_steps
        self._acceptance_history = np.asarray(acceptance_history)
        self._remove_checkpoint_and_jax_cache(config, checkpoint_path)

    def _run_adaptive_tempered(self, rng_key: Key, initial_particles) -> None:
        """Run adaptive tempered SMC with the configured inner kernel."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config._resolve_target_ess_fraction()
        checkpoint_path, run_started_at = self._prepare_checkpointing(config)

        inner_kernel_step = self._build_inner_kernel_step()
        update_mcmc_parameters, initial_mcmc_parameters = (
            self._build_adaptive_inner_kernel_parameters(initial_particles)
        )

        smc_algorithm = inner_kernel_tuning(
            smc_algorithm=adaptive_tempered_smc,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            mcmc_step_fn=inner_kernel_step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=update_mcmc_parameters,
            initial_parameter_value=initial_mcmc_parameters,
            num_mcmc_steps=n_mcmc_steps,
            target_ess=target_ess,
            batch_size=config.batch_size,
        )

        state, rng_key, n_completed_iterations, mode_checkpoint_data = (
            self._load_or_initialize_state(
                checkpoint_path,
                config,
                rng_key,
                initial_particles,
                smc_algorithm.init,
                initial_mode_data={
                    "accept_history": [],
                    "tempering_schedule": [],
                    "is_weights_history": [],
                },
                load_mode_data=lambda checkpoint: {
                    "accept_history": list(checkpoint["accept_history"]),
                    "tempering_schedule": list(checkpoint["tempering_schedule"]),
                    "is_weights_history": list(checkpoint["is_weights_history"]),
                },
            )
        )
        acceptance_history: list[float] = mode_checkpoint_data["accept_history"]
        tempering_schedule: list[float] = mode_checkpoint_data["tempering_schedule"]
        importance_weight_history: list[np.ndarray] = mode_checkpoint_data[
            "is_weights_history"
        ]

        run_adaptive_step = jax.jit(smc_algorithm.step)
        last_checkpoint_at = time.perf_counter()

        while state.sampler_state.tempering_param < 1.0:
            rng_key, step_key = jax.random.split(rng_key)
            state, info = run_adaptive_step(step_key, state)

            acceptance_history.append(float(info.update_info.acceptance_rate.mean()))
            tempering_schedule.append(float(state.sampler_state.tempering_param))
            importance_weight_history.append(np.asarray(state.sampler_state.weights))
            n_completed_iterations += 1

            last_checkpoint_at = self._save_checkpoint_if_due(
                config,
                checkpoint_path,
                last_checkpoint_at,
                run_started_at,
                state,
                rng_key,
                n_completed_iterations,
                {
                    "accept_history": acceptance_history.copy(),
                    "tempering_schedule": tempering_schedule.copy(),
                    "is_weights_history": np.stack(importance_weight_history),
                },
            )

        self._final_state = state
        self._n_iterations = n_completed_iterations
        self._acceptance_history = np.asarray(acceptance_history)
        self._tempering_schedule = np.asarray(tempering_schedule)
        self._is_weights_history = (
            np.stack(importance_weight_history)
            if importance_weight_history
            else np.empty((0, initial_particles.shape[0]))
        )
        self._remove_checkpoint_and_jax_cache(config, checkpoint_path)

    def _run_fixed_tempered(
        self, rng_key: Key, initial_particles, ladder: list[float]
    ) -> None:
        """Run fixed-ladder tempered SMC with the configured inner kernel."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        tempering_parameters = ladder[1:]
        n_temperature_steps = len(tempering_parameters)
        checkpoint_path, run_started_at = self._prepare_checkpointing(config)

        inner_kernel_step = self._build_inner_kernel_step()
        initialize_state, temperature_step = self._build_fixed_ladder_smc(
            uses_persistent_sampling=False,
            inner_kernel_step=inner_kernel_step,
            initial_particles=initial_particles,
            n_temperature_steps=n_temperature_steps,
            n_mcmc_steps=n_mcmc_steps,
        )

        state, rng_key, n_completed_iterations, mode_checkpoint_data = (
            self._load_or_initialize_state(
                checkpoint_path,
                config,
                rng_key,
                initial_particles,
                initialize_state,
                initial_mode_data={"accept_history": [], "is_weights_history": []},
                load_mode_data=lambda checkpoint: {
                    "accept_history": list(checkpoint["accept_history"]),
                    "is_weights_history": list(checkpoint["is_weights_history"]),
                },
                n_temperature_steps=n_temperature_steps,
            )
        )
        acceptance_history: list[float] = mode_checkpoint_data["accept_history"]
        importance_weight_history: list[np.ndarray] = mode_checkpoint_data[
            "is_weights_history"
        ]

        run_temperature_step = jax.jit(temperature_step)
        last_checkpoint_at = time.perf_counter()

        for tempering_parameter in tempering_parameters[n_completed_iterations:]:
            rng_key, step_key = jax.random.split(rng_key)
            state, info = run_temperature_step(step_key, state, tempering_parameter)
            acceptance_history.append(float(info.update_info.acceptance_rate.mean()))
            importance_weight_history.append(np.asarray(state.weights))
            n_completed_iterations += 1
            last_checkpoint_at = self._save_checkpoint_if_due(
                config,
                checkpoint_path,
                last_checkpoint_at,
                run_started_at,
                state,
                rng_key,
                n_completed_iterations,
                {
                    "accept_history": acceptance_history.copy(),
                    "is_weights_history": np.stack(importance_weight_history),
                },
            )

        self._final_state = state
        self._n_iterations = n_temperature_steps
        self._acceptance_history = np.asarray(acceptance_history)
        self._is_weights_history = (
            np.stack(importance_weight_history)
            if importance_weight_history
            else np.empty((0, initial_particles.shape[0]))
        )
        self._remove_checkpoint_and_jax_cache(config, checkpoint_path)

    def _sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_particles n_dims"],
    ) -> None:
        """Run the BlackJAX SMC sampler.

        If ``config.checkpoint_dir`` is set, a ``checkpoint.pkl`` is written
        atomically after each tempering iteration (subject to
        ``config.checkpoint_interval``) and the sampler resumes from the
        checkpoint if one already exists at that path.

        Args:
            rng_key: JAX PRNG key.
            initial_position: Starting particles in the sampling space,
                shape ``(n_particles, n_dims)``.  Must match ``config.n_particles``.
                Ignored when resuming from a checkpoint.

        Raises:
            ValueError: If ``initial_position`` shape does not match
                ``(n_particles, n_dims)``.
        """
        config = self._config
        n_particles = config.n_particles

        arr = jnp.asarray(initial_position)
        if arr.ndim != 2 or arr.shape != (n_particles, self.n_dims):
            raise ValueError(
                f"initial_position must have shape ({n_particles}, {self.n_dims}), "
                f"got {arr.shape}."
            )
        initial_particles = arr

        ladder = config.temperature_ladder
        mode = self.mode

        if mode == "ap":
            self._run_adaptive_persistent(rng_key, initial_particles)
        elif mode == "fp":
            assert ladder is not None
            self._run_fixed_persistent(rng_key, initial_particles, ladder)
        elif mode == "at":
            self._run_adaptive_tempered(rng_key, initial_particles)
        else:
            assert ladder is not None
            self._run_fixed_tempered(rng_key, initial_particles, ladder)

    def get_samples(self) -> dict[str, np.ndarray]:
        """Return posterior samples.

        When ``persistent_sampling=True``: samples are drawn with replacement
        from all-temperature particles weighted by the persistent-sampling
        weight formula.  The number of returned samples approximately equals
        the effective sample size ``1 / max(weights)``.

        When ``persistent_sampling=False``: returns all final-temperature
        particles with equal weight.

        Returns:
            Dict with keys ``"samples"`` (shape ``(n, n_dims)``) and
            ``"log_likelihood"`` (shape ``(n,)``).
        """
        if not self._sampled:
            raise RuntimeError("get_samples() called before sample()")

        mode = self.mode
        state = self._final_state

        if mode in ("ap", "fp"):
            ps = state.sampler_state if mode == "ap" else state
            n_iter = int(ps.iteration)

            all_particles = np.asarray(ps.persistent_particles[: n_iter + 1]).reshape(
                -1, self.n_dims
            )
            all_log_likelihoods = np.asarray(
                ps.persistent_log_likelihoods[: n_iter + 1]
            ).reshape(-1)

            log_w, _ = compute_log_persistent_weights(
                ps.persistent_log_likelihoods,
                ps.persistent_log_Z,
                ps.tempering_schedule,
                ps.iteration,
                include_current=True,
            )
            weights = np.asarray(jax.nn.softmax(log_w[: n_iter + 1].reshape(-1)))

            n_available = all_particles.shape[0]
            n_target = max(1, int(1.0 / float(np.max(weights))))
            n_target = min(n_target, n_available)
            indices = np.array(
                jax.random.choice(
                    _RESAMPLE_KEY,
                    n_available,
                    shape=(n_target,),
                    replace=True,
                    p=weights,
                )
            )
            return {
                "samples": all_particles[indices],
                "log_likelihood": all_log_likelihoods[indices],
            }

        elif mode == "at":
            ps = state.sampler_state
            final_particles = np.array(ps.particles)
            pbs = self._config.batch_size
            log_likelihoods = np.array(
                jax.lax.map(self._log_likelihood_fn, ps.particles, batch_size=pbs)
                if pbs > 0
                else jax.vmap(self._log_likelihood_fn)(ps.particles)
            )
            return {"samples": final_particles, "log_likelihood": log_likelihoods}

        else:  # mode == "ft"
            final_particles = np.array(state.particles)
            pbs = self._config.batch_size
            log_likelihoods = np.array(
                jax.lax.map(self._log_likelihood_fn, state.particles, batch_size=pbs)
                if pbs > 0
                else jax.vmap(self._log_likelihood_fn)(state.particles)
            )
            return {"samples": final_particles, "log_likelihood": log_likelihoods}

    def _get_diagnostics(self) -> dict[str, Any]:
        """Return SMC run diagnostics.

        Returns a dict with the following keys (not all present for all modes):

        * ``"n_likelihood_evaluations"`` — total likelihood calls.
        * ``"acceptance_history"`` — per-iteration mean acceptance rate; length ``n_iterations``.
        * ``"n_iterations"`` — total SMC iterations (adaptive modes only).
        * ``"tempering_schedule"`` — inverse temperature at each iteration; length ``n_iterations`` (adaptive modes only).
        * ``"cov_scale_history"`` — covariance scale per iteration for
          adaptive-persistent GRW; empty for adaptive-persistent DE.
        * ``"ess_history"`` — ESS per iteration (all modes: persistent ESS for ap/fp, Kish ESS for at/ft); length ``n_iterations``.
        * ``"persistent_log_Z"`` — cumulative log-Z after each iteration; length ``n_iterations`` (persistent modes only).
        * ``"log_Z"`` — final log Bayesian evidence (persistent modes only).
        * ``"log_Z_error"`` — standard deviation of log Z from delta-method IS weight variance (all modes).
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")

        cfg = self._config
        mode = self.mode
        n_mcmc = cfg.n_mcmc_steps_per_dim * self.n_dims
        n_iter = self._n_iterations

        result: dict[str, Any] = {
            "n_likelihood_evaluations": n_mcmc * n_iter * cfg.n_particles,
        }

        if mode in ("ap", "at"):
            result["n_iterations"] = n_iter
            result["acceptance_history"] = self._acceptance_history
            if mode == "ap":
                result["cov_scale_history"] = self._cov_scale_history
                ps = self._final_state.sampler_state
                n = int(ps.iteration)
                result["tempering_schedule"] = np.asarray(
                    ps.tempering_schedule[1 : n + 1]
                )
                log_Z_traj = np.asarray(ps.persistent_log_Z[1 : n + 1])
                result["persistent_log_Z"] = log_Z_traj
                result["log_Z"] = float(log_Z_traj[-1])
            else:  # mode == "at"
                result["tempering_schedule"] = self._tempering_schedule
        elif mode in ("fp", "ft"):
            result["acceptance_history"] = self._acceptance_history
            if mode == "fp":
                ps = self._final_state
                n = int(ps.iteration)
                log_Z_traj = np.asarray(ps.persistent_log_Z[1 : n + 1])
                result["persistent_log_Z"] = log_Z_traj
                result["log_Z"] = float(log_Z_traj[-1])

        if mode in ("ap", "fp"):
            ps = self._final_state.sampler_state if mode == "ap" else self._final_state
            n = self._n_iterations
            ess_hist = np.zeros(n)
            for t in range(1, n + 1):
                log_w, _ = compute_log_persistent_weights(
                    ps.persistent_log_likelihoods,
                    ps.persistent_log_Z,
                    ps.tempering_schedule,
                    t,
                    include_current=True,
                )
                ess_hist[t - 1] = float(
                    compute_persistent_ess(log_w.reshape(-1), normalize_weights=True)
                )
            result["ess_history"] = ess_hist
            # Delta-method log_Z error bar.
            # At step k, IS weights exp(Δβ·logL) over all k batches of accumulated particles.
            # Var(log Z_k) = Var(w) / (N_eff · E[w]²) summed across steps.
            var_list = []
            for k in range(1, n + 1):
                delta_beta = float(ps.tempering_schedule[k]) - float(
                    ps.tempering_schedule[k - 1]
                )
                log_L_accum = np.asarray(ps.persistent_log_likelihoods[:k]).reshape(-1)
                log_w_k = delta_beta * log_L_accum
                m = float(np.max(log_w_k))
                u = np.exp(log_w_k - m)
                mean_u = float(np.mean(u))
                if mean_u > 0:
                    var_list.append(float(np.var(u)) / (len(log_w_k) * mean_u**2))
            result["log_Z_error"] = float(np.sqrt(np.sum(var_list)))

        if mode in ("at", "ft"):
            # IS weights are already normalized; Kish ESS = 1/sum(w^2)
            w = self._is_weights_history
            n_particles = w.shape[1]
            result["ess_history"] = 1.0 / np.sum(w**2, axis=-1)
            # Delta-method log_Z error bar: Var(log Z_k) = sum(p²) - 1/N for normalized weights.
            var_per_step = np.sum(w**2, axis=-1) - 1.0 / n_particles
            result["log_Z_error"] = float(
                np.sqrt(float(np.clip(np.sum(var_per_step), 0.0, None)))
            )

        return result
