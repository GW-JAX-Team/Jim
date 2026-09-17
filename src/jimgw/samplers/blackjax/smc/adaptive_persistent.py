"""Mode AP: adaptive persistent SMC (``adaptive_persistent_sampling_smc`` + ``inner_kernel_tuning``)."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from blackjax import adaptive_persistent_sampling_smc, inner_kernel_tuning, rmh
from blackjax.smc import extend_params
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.smc.resampling import systematic
from jaxtyping import Array, Key

from jimgw.samplers.blackjax.smc import diagnostics, precondition
from jimgw.samplers.blackjax.smc.base import _BlackJAXSMCBase
from jimgw.samplers.blackjax.utils import (
    load_or_initialize_checkpoint,
    prepare_checkpointing,
    remove_checkpoint_and_jax_cache,
    save_checkpoint_if_due,
)


class AdaptivePersistentSMCSampler(_BlackJAXSMCBase):
    """BlackJAX SMC, adaptive temperature schedule with persistent sampling."""

    mode = "ap"
    _cov_scale_history: np.ndarray  # per-step covariance scale

    def _run(self, rng_key: Key, initial_particles) -> None:
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config._resolve_target_ess_fraction()
        checkpoint_path, run_start_time = prepare_checkpointing(config)
        log_label = f"{self.sampler_name} ({self.mode.upper()})"

        rng_key, precond = self._setup_precondition_for_run(rng_key, initial_particles)
        mcmc_step = (
            precond.mcmc_step if precond is not None else self._build_mcmc_step()
        )
        initial_covariance = (
            precondition.weighted_covariance(initial_particles)
            * config.initial_cov_scale
        )
        initial_parameters: dict[str, Array] = {"cov": initial_covariance}
        if precond is not None:
            initial_parameters["cov"] = (
                self._latent_covariance(precond.flow, initial_particles)
                * config.initial_cov_scale
            )
            initial_parameters["flow_params"] = precond.flat_params

        def mcmc_parameter_update_fn(_key, state, _info):
            return extend_params({"cov": jnp.atleast_2d(jnp.cov(state.particles.T))})  # type: ignore[arg-type]  # blackjax stubs: extend_params accepts dict

        smc_algorithm = inner_kernel_tuning(
            smc_algorithm=adaptive_persistent_sampling_smc,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            max_iterations=1000,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params(initial_parameters),  # type: ignore[arg-type]  # blackjax stubs: extend_params accepts dict
            num_mcmc_steps=n_mcmc_steps,
            target_ess=target_ess,
            batch_size=config.batch_size,
        )

        covariance_scale = float(config.initial_cov_scale)
        state, rng_key, n_completed_iterations, mode_checkpoint_data = (
            load_or_initialize_checkpoint(
                self,
                checkpoint_path,
                config,
                rng_key,
                initial_particles,
                smc_algorithm.init,
                initial_extra={
                    "cov_scale": covariance_scale,
                    "accept_history": [],
                    "cov_scale_history": [],
                    "precondition_optimizer_state": None,
                },
                load_extra=lambda checkpoint: {
                    "cov_scale": float(checkpoint.get("cov_scale", covariance_scale)),
                    "accept_history": list(checkpoint["accept_history"]),
                    "cov_scale_history": list(checkpoint["cov_scale_history"]),
                    "precondition_optimizer_state": checkpoint.get(
                        "precondition_optimizer_state"
                    ),
                },
                log_label=log_label,
            )
        )
        covariance_scale = mode_checkpoint_data["cov_scale"]
        acceptance_history: list[float] = mode_checkpoint_data["accept_history"]
        covariance_scale_history: list[float] = mode_checkpoint_data[
            "cov_scale_history"
        ]
        if precond is not None:
            precond.flow, precond.flat_params = self._reconstruct_precondition_flow(
                state, precond.unravel_fn, precond.static
            )
        self._restore_precondition_optimizer_state(
            precond, mode_checkpoint_data, n_completed_iterations
        )

        run_adaptive_step = jax.jit(smc_algorithm.step)
        last_checkpoint_write_time = time.perf_counter()

        while state.sampler_state.tempering_param < 1.0:  # type: ignore[attr-defined]  # blackjax stubs
            rng_key, step_key = jax.random.split(rng_key)
            state, info = run_adaptive_step(step_key, state)

            sampler_state = state.sampler_state
            acceptance_rate = float(info.update_info.acceptance_rate.mean())
            updated_covariance_scale = covariance_scale * float(
                jnp.exp(
                    config.scale_adaptation_gain
                    * (acceptance_rate - config.target_acceptance_rate)
                )
            )
            current_covariance = state.parameter_override["cov"]
            new_parameters: dict[str, Array] = {
                "cov": current_covariance[0] * updated_covariance_scale
            }
            if precond is not None:
                is_terminal_iteration = bool(sampler_state.tempering_param >= 1.0)
                rng_key, precond_parameters = self._precondition_iteration_update(
                    rng_key,
                    precond,
                    sampler_state,
                    is_terminal_iteration=is_terminal_iteration,
                    n_completed_iterations=n_completed_iterations,
                    needs_resampling=False,
                    covariance_scale=updated_covariance_scale,
                )
                new_parameters.update(precond_parameters)
            updated_parameters = extend_params(new_parameters)  # type: ignore[arg-type]  # blackjax stubs
            state = StateWithParameterOverride(sampler_state, updated_parameters)  # type: ignore[arg-type]  # blackjax stubs

            acceptance_history.append(acceptance_rate)
            covariance_scale_history.append(updated_covariance_scale)
            covariance_scale = updated_covariance_scale
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
                    cov_scale=covariance_scale,
                    accept_history=acceptance_history.copy(),
                    cov_scale_history=covariance_scale_history.copy(),
                    precondition_optimizer_state=(
                        precond.optimizer.optim_state if precond is not None else None
                    ),
                ),
                log_label=log_label,
            )

        self._final_state = state
        self._n_iterations = n_completed_iterations
        self._acceptance_history = np.asarray(acceptance_history)
        self._cov_scale_history = np.asarray(covariance_scale_history)
        remove_checkpoint_and_jax_cache(config, checkpoint_path)

    def get_samples(self) -> dict[str, np.ndarray]:
        """Return posterior samples.

        Samples are drawn with replacement from all-temperature particles
        weighted by the persistent-sampling weight formula. The number of
        returned samples approximately equals the effective sample size
        ``1 / max(weights)``.

        Returns:
            Dict with keys ``"samples"`` (shape ``(n, n_dims)``) and
            ``"log_likelihood"`` (shape ``(n,)``).
        """
        if not self._sampled:
            raise RuntimeError("get_samples() called before sample()")
        ps = self._final_state.sampler_state
        return diagnostics.resample_persistent_particles(ps, self.n_dims)

    def _get_diagnostics(self) -> dict[str, object]:
        """Return SMC run diagnostics.

        Returns a dict with keys ``"n_likelihood_evaluations"``,
        ``"n_iterations"``, ``"acceptance_history"``, ``"cov_scale_history"``,
        ``"tempering_schedule"``, ``"persistent_log_Z"``, ``"log_Z"``,
        ``"ess_history"``, ``"log_Z_error"``.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")
        cfg = self._config
        n_mcmc = cfg.n_mcmc_steps_per_dim * self.n_dims
        n_iter = self._n_iterations

        ps = self._final_state.sampler_state
        n = int(ps.iteration)
        log_Z_traj = np.asarray(ps.persistent_log_Z[1 : n + 1])

        return {
            "n_likelihood_evaluations": n_mcmc * n_iter * cfg.n_particles,
            "n_iterations": n_iter,
            "acceptance_history": self._acceptance_history,
            "cov_scale_history": self._cov_scale_history,
            "tempering_schedule": np.asarray(ps.tempering_schedule[1 : n + 1]),
            "persistent_log_Z": log_Z_traj,
            "log_Z": float(log_Z_traj[-1]),
            "ess_history": diagnostics.persistent_ess_history(ps, self._n_iterations),
            "log_Z_error": diagnostics.persistent_log_z_error(ps, self._n_iterations),
        }
