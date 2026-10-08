"""Mode AT: adaptive tempered SMC (``adaptive_tempered_smc`` + ``inner_kernel_tuning``)."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from blackjax import adaptive_tempered_smc, inner_kernel_tuning, rmh
from blackjax.smc import extend_params
from blackjax.smc.resampling import systematic
from jaxtyping import Key

from jimgw.samplers.blackjax.smc import diagnostics
from jimgw.samplers.blackjax.smc.base import _BlackJAXSMCBase
from jimgw.samplers.blackjax.utils import (
    load_or_initialize_checkpoint,
    prepare_checkpointing,
    remove_checkpoint_and_jax_cache,
    save_checkpoint_if_due,
)


class AdaptiveTemperedSMCSampler(_BlackJAXSMCBase):
    """BlackJAX SMC, adaptive temperature schedule, plain (non-persistent) tempering."""

    mode = "at"
    _tempering_schedule: np.ndarray  # per-step temperature
    _is_weights_history: np.ndarray  # per-step normalized IS weights

    def _run(self, rng_key: Key, initial_particles) -> None:
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config._resolve_target_ess_fraction()
        checkpoint_path, run_start_time = prepare_checkpointing(config)
        log_label = f"{self.sampler_name} ({self.mode.upper()})"

        mcmc_step = self._build_mcmc_step()
        initial_covariance = (
            jnp.atleast_2d(jnp.cov(initial_particles.T)) * config.initial_cov_scale
        )

        def mcmc_parameter_update_fn(_key, state, _info):
            return extend_params({"cov": jnp.atleast_2d(jnp.cov(state.particles.T))})  # type: ignore[arg-type]  # blackjax stubs: extend_params accepts dict

        smc_algorithm = inner_kernel_tuning(
            smc_algorithm=adaptive_tempered_smc,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params({"cov": initial_covariance}),  # type: ignore[arg-type]  # blackjax stubs: extend_params accepts dict
            num_mcmc_steps=n_mcmc_steps,
            target_ess=target_ess,
            batch_size=config.batch_size,
        )

        state, rng_key, n_completed_iterations, mode_checkpoint_data = (
            load_or_initialize_checkpoint(
                self,
                checkpoint_path,
                config,
                rng_key,
                initial_particles,
                smc_algorithm.init,
                initial_extra={
                    "accept_history": [],
                    "tempering_schedule": [],
                    "is_weights_history": [],
                },
                load_extra=lambda checkpoint: {
                    "accept_history": list(checkpoint["accept_history"]),
                    "tempering_schedule": list(checkpoint["tempering_schedule"]),
                    "is_weights_history": list(checkpoint["is_weights_history"]),
                },
                log_label=log_label,
            )
        )
        acceptance_history: list[float] = mode_checkpoint_data["accept_history"]
        tempering_schedule: list[float] = mode_checkpoint_data["tempering_schedule"]
        importance_weight_history: list[np.ndarray] = mode_checkpoint_data[
            "is_weights_history"
        ]

        run_adaptive_step = jax.jit(smc_algorithm.step)
        last_checkpoint_write_time = time.perf_counter()

        while state.sampler_state.tempering_param < 1.0:
            rng_key, step_key = jax.random.split(rng_key)
            state, info = run_adaptive_step(step_key, state)

            acceptance_history.append(float(info.update_info.acceptance_rate.mean()))
            tempering_schedule.append(float(state.sampler_state.tempering_param))
            importance_weight_history.append(np.asarray(state.sampler_state.weights))
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
                self._checkpoint_extra(
                    accept_history=acceptance_history.copy(),
                    tempering_schedule=tempering_schedule.copy(),
                    is_weights_history=np.stack(importance_weight_history),
                ),
                log_label=log_label,
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
        remove_checkpoint_and_jax_cache(config, checkpoint_path)

    def get_samples(self) -> dict[str, np.ndarray]:
        """Return posterior samples: all final-temperature particles, equal weight.

        Returns:
            Dict with keys ``"samples"`` (shape ``(n, n_dims)``) and
            ``"log_likelihood"`` (shape ``(n,)``).
        """
        if not self._sampled:
            raise RuntimeError("get_samples() called before sample()")
        particles = self._final_state.sampler_state.particles
        pbs = self._config.batch_size
        log_likelihoods = np.array(
            jax.lax.map(self._log_likelihood_fn, particles, batch_size=pbs)
            if pbs > 0
            else jax.vmap(self._log_likelihood_fn)(particles)
        )
        return {"samples": np.array(particles), "log_likelihood": log_likelihoods}

    def _get_diagnostics(self) -> dict[str, object]:
        """Return SMC run diagnostics.

        Returns a dict with keys ``"n_likelihood_evaluations"``,
        ``"n_iterations"``, ``"acceptance_history"``, ``"tempering_schedule"``,
        ``"ess_history"``, ``"log_Z_error"``.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")
        cfg = self._config
        n_mcmc = cfg.n_mcmc_steps_per_dim * self.n_dims
        n_iter = self._n_iterations

        return {
            "n_likelihood_evaluations": n_mcmc * n_iter * cfg.n_particles,
            "n_iterations": n_iter,
            "acceptance_history": self._acceptance_history,
            "tempering_schedule": self._tempering_schedule,
            "ess_history": diagnostics.kish_ess_history(self._is_weights_history),
            "log_Z_error": diagnostics.kish_log_z_error(self._is_weights_history),
        }
