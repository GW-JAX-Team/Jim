"""Mode AT: adaptive tempered SMC (``adaptive_tempered_smc`` + ``inner_kernel_tuning``)."""

import time

import jax
import numpy as np
from blackjax import adaptive_tempered_smc, inner_kernel_tuning
from jaxtyping import Array, Key

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

    def _run(self, rng_key: Key[Array, ""], initial_particles) -> None:
        config = self._config
        checkpoint_path, run_start_time = prepare_checkpointing(config)

        smc_algorithm = inner_kernel_tuning(
            smc_algorithm=adaptive_tempered_smc,
            target_ess=config._resolve_target_ess_fraction(),
            **self._common_arguments(),
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
                    "mutation_history": self._new_mutation_history(),
                    "tempering_schedule": [],
                    "is_weights_history": [],
                },
                load_extra=lambda checkpoint: {
                    "mutation_history": checkpoint["mutation_history"],
                    "tempering_schedule": list(checkpoint["tempering_schedule"]),
                    "is_weights_history": list(checkpoint["is_weights_history"]),
                },
                log_label=self._log_label,
            )
        )
        mutation_history: dict[str, list] = mode_checkpoint_data["mutation_history"]
        tempering_schedule: list[float] = mode_checkpoint_data["tempering_schedule"]
        importance_weight_history: list[np.ndarray] = mode_checkpoint_data[
            "is_weights_history"
        ]

        run_adaptive_step = jax.jit(smc_algorithm.step)
        last_checkpoint_write_time = time.perf_counter()

        while state.sampler_state.tempering_param < 1.0:
            rng_key, step_key = jax.random.split(rng_key)
            step_start_time = time.perf_counter()
            state, info = run_adaptive_step(step_key, state)
            n_completed_iterations += 1
            tempering_schedule.append(float(state.sampler_state.tempering_param))
            importance_weight_history.append(np.asarray(state.sampler_state.weights))
            self._record_mutation_history(
                mutation_history,
                info.update_info,
                tempering_schedule[-1],
                n_completed_iterations,
                step_start_time,
            )

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
                    mutation_history=self._copy_mutation_history(mutation_history),
                    tempering_schedule=tempering_schedule.copy(),
                    is_weights_history=np.stack(importance_weight_history),
                ),
                log_label=self._log_label,
            )

        self._final_state = state
        self._n_iterations = n_completed_iterations
        self._mutation_history = mutation_history
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
        ``"n_iterations"``, ``"acceptance_history"``, ``"n_steps_history"``,
        ``"scale_history"``, ``"training_loss_history"`` (with a flow),
        ``"tempering_schedule"``, ``"ess_history"``, ``"log_Z_error"``.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")

        return {
            # Per iteration: one pass where BlackJAX solves for the next
            # temperature, one over the mutation's starting particles and one
            # where BlackJAX reweights the particles.
            "n_likelihood_evaluations": self._n_likelihood_evaluations(
                n_initial_passes=0, n_extra_passes_per_iteration=3
            ),
            "n_iterations": self._n_iterations,
            **self._mutation_history_diagnostics(),
            "tempering_schedule": self._tempering_schedule,
            "ess_history": diagnostics.kish_ess_history(self._is_weights_history),
            "log_Z_error": diagnostics.kish_log_z_error(self._is_weights_history),
        }
