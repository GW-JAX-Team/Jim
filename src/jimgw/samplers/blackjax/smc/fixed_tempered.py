"""Mode FT: plain (non-persistent) tempered SMC over an explicit ladder (``tempered_smc``)."""

import time

import jax
import numpy as np
from blackjax import inner_kernel_tuning, rmh, tempered_smc
from blackjax.smc import extend_params
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.smc.resampling import systematic
from jaxtyping import Key

from jimgw.samplers.blackjax.smc import diagnostics, precondition
from jimgw.samplers.blackjax.smc.base import _BlackJAXSMCBase
from jimgw.samplers.blackjax.utils import (
    load_or_initialize_checkpoint,
    prepare_checkpointing,
    remove_checkpoint_and_jax_cache,
    save_checkpoint_if_due,
)


class FixedTemperedSMCSampler(_BlackJAXSMCBase):
    """BlackJAX SMC, fixed temperature ladder, plain (non-persistent) tempering."""

    mode = "ft"
    _is_weights_history: np.ndarray  # per-step normalized IS weights

    def _run(self, rng_key: Key, initial_particles) -> None:
        config = self._config
        ladder = config.temperature_ladder
        assert ladder is not None
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        tempering_parameters = ladder[1:]  # skip 0.0
        n_temperature_steps = len(tempering_parameters)
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

        if precond is not None:
            initial_latent_cov = (
                self._latent_covariance(precond.flow, initial_particles)
                * config.initial_cov_scale
            )
            initial_parameters = {
                "cov": initial_latent_cov,
                "flow_params": precond.flat_params,
            }

            def mcmc_parameter_update_fn(
                _key,
                _state,
                _info,
                _cov=initial_latent_cov,
                _flow_params=precond.flat_params,
            ):
                # Placeholder only, same reasoning as FP mode's mcmc_parameter_update_fn.
                return extend_params({"cov": _cov, "flow_params": _flow_params})  # type: ignore[arg-type]  # blackjax stubs

            smc_algorithm = inner_kernel_tuning(
                smc_algorithm=tempered_smc,
                logprior_fn=self._log_prior_fn,
                loglikelihood_fn=self._log_likelihood_fn,
                mcmc_step_fn=mcmc_step,
                mcmc_init_fn=rmh.init,
                resampling_fn=systematic,
                mcmc_parameter_update_fn=mcmc_parameter_update_fn,
                initial_parameter_value=extend_params(initial_parameters),  # type: ignore[arg-type]  # blackjax stubs
                num_mcmc_steps=n_mcmc_steps,
                batch_size=config.batch_size,
            )
        else:
            smc_algorithm = tempered_smc(
                logprior_fn=self._log_prior_fn,
                loglikelihood_fn=self._log_likelihood_fn,
                mcmc_step_fn=mcmc_step,
                mcmc_init_fn=rmh.init,
                mcmc_parameters=extend_params({"cov": initial_covariance}),  # type: ignore[arg-type]  # blackjax stubs: extend_params accepts dict
                resampling_fn=systematic,
                num_mcmc_steps=n_mcmc_steps,
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
                    "is_weights_history": [],
                    "precondition_optimizer_state": None,
                },
                load_extra=lambda checkpoint: {
                    "accept_history": list(checkpoint["accept_history"]),
                    "is_weights_history": list(checkpoint["is_weights_history"]),
                    "precondition_optimizer_state": checkpoint.get(
                        "precondition_optimizer_state"
                    ),
                },
                log_label=log_label,
                is_stale=lambda n_iter: n_iter > n_temperature_steps,
                stale_message="checkpoint n_iter exceeds current schedule length"
                f"={n_temperature_steps}",
            )
        )
        acceptance_history: list[float] = mode_checkpoint_data["accept_history"]
        importance_weight_history: list[np.ndarray] = mode_checkpoint_data[
            "is_weights_history"
        ]
        if precond is not None:
            precond.flow, precond.flat_params = self._reconstruct_precondition_flow(
                state, precond.unravel_fn, precond.static
            )
        self._restore_precondition_optimizer_state(
            precond, mode_checkpoint_data, n_completed_iterations
        )

        run_temperature_step = jax.jit(smc_algorithm.step)
        last_checkpoint_write_time = time.perf_counter()

        for tempering_parameter in tempering_parameters[n_completed_iterations:]:
            rng_key, step_key = jax.random.split(rng_key)
            if precond is not None:
                state, info = run_temperature_step(
                    step_key, state, tempering_param=tempering_parameter
                )
            else:
                state, info = run_temperature_step(step_key, state, tempering_parameter)

            if precond is not None:
                sampler_state = state.sampler_state
                acceptance_history.append(
                    float(info.update_info.acceptance_rate.mean())
                )
                importance_weight_history.append(np.asarray(sampler_state.weights))
                is_terminal_iteration = tempering_parameter == tempering_parameters[-1]
                rng_key, new_parameters = self._precondition_iteration_update(
                    rng_key,
                    precond,
                    sampler_state,
                    is_terminal_iteration=is_terminal_iteration,
                    n_completed_iterations=n_completed_iterations,
                    needs_resampling=True,
                )
                state = StateWithParameterOverride(
                    sampler_state,
                    extend_params(new_parameters),  # type: ignore[arg-type]  # blackjax stubs
                )
            else:
                acceptance_history.append(
                    float(info.update_info.acceptance_rate.mean())
                )
                importance_weight_history.append(np.asarray(state.weights))

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
                    is_weights_history=np.stack(importance_weight_history),
                    precondition_optimizer_state=(
                        precond.optimizer.optim_state if precond is not None else None
                    ),
                ),
                log_label=log_label,
            )

        self._final_state = state.sampler_state if precond is not None else state
        self._n_iterations = n_temperature_steps
        self._acceptance_history = np.asarray(acceptance_history)
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
        particles = self._final_state.particles
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
        ``"acceptance_history"``, ``"ess_history"``, ``"log_Z_error"``.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")
        cfg = self._config
        n_mcmc = cfg.n_mcmc_steps_per_dim * self.n_dims
        n_iter = self._n_iterations

        return {
            "n_likelihood_evaluations": n_mcmc * n_iter * cfg.n_particles,
            "acceptance_history": self._acceptance_history,
            "ess_history": diagnostics.kish_ess_history(self._is_weights_history),
            "log_Z_error": diagnostics.kish_log_z_error(self._is_weights_history),
        }
