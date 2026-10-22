"""Mode FP: persistent SMC over an explicit temperature ladder (``persistent_sampling_smc`` + ``inner_kernel_tuning``)."""

import time

import jax
import numpy as np
from blackjax import inner_kernel_tuning, persistent_sampling_smc
from jaxtyping import Array, Key

from jimgw.samplers.blackjax.smc import diagnostics
from jimgw.samplers.blackjax.smc.base import _BlackJAXSMCBase
from jimgw.samplers.blackjax.utils import (
    load_or_initialize_checkpoint,
    prepare_checkpointing,
    remove_checkpoint_and_jax_cache,
    save_checkpoint_if_due,
)


class FixedPersistentSMCSampler(_BlackJAXSMCBase):
    """BlackJAX SMC, fixed temperature ladder with persistent sampling."""

    mode = "fp"

    def _run(self, rng_key: Key[Array, ""], initial_particles) -> None:
        config = self._config
        ladder = config.temperature_ladder
        assert ladder is not None
        tempering_parameters = ladder[1:]  # skip 0.0 (already in init state)
        n_temperature_steps = len(tempering_parameters)
        checkpoint_path, run_start_time = prepare_checkpointing(config)

        smc_algorithm = inner_kernel_tuning(
            smc_algorithm=persistent_sampling_smc,
            n_schedule=n_temperature_steps,
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
                initial_extra={"mutation_history": self._new_mutation_history()},
                load_extra=lambda checkpoint: {
                    "mutation_history": checkpoint["mutation_history"]
                },
                log_label=self._log_label,
                is_stale=lambda n_iter: n_iter > n_temperature_steps,
                stale_message="checkpoint n_iter exceeds current schedule length"
                f"={n_temperature_steps}",
            )
        )
        mutation_history: dict[str, list] = mode_checkpoint_data["mutation_history"]

        run_temperature_step = jax.jit(smc_algorithm.step)
        last_checkpoint_write_time = time.perf_counter()

        for tempering_parameter in tempering_parameters[n_completed_iterations:]:
            rng_key, step_key = jax.random.split(rng_key)
            step_start_time = time.perf_counter()
            state, info = run_temperature_step(
                step_key, state, lmbda=tempering_parameter
            )
            n_completed_iterations += 1
            self._record_mutation_history(
                mutation_history,
                info.update_info,
                tempering_parameter,
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
                    mutation_history=self._copy_mutation_history(mutation_history)
                ),
                log_label=self._log_label,
            )

        self._final_state = state
        self._n_iterations = n_temperature_steps
        self._mutation_history = mutation_history
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
        return diagnostics.resample_persistent_particles(
            self._final_state.sampler_state, self.n_dims
        )

    def _get_diagnostics(self) -> dict[str, object]:
        """Return SMC run diagnostics.

        Returns a dict with keys ``"n_likelihood_evaluations"``,
        ``"acceptance_history"``, ``"n_steps_history"``, ``"scale_history"``,
        ``"training_loss_history"`` (with a flow), ``"persistent_log_Z"``,
        ``"log_Z"``, ``"ess_history"``, ``"log_Z_error"``.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")
        ps = self._final_state.sampler_state
        n = int(ps.iteration)
        log_Z_traj = np.asarray(ps.persistent_log_Z[1 : n + 1])

        return {
            # One pass over the initial particles; per iteration, one over the
            # mutation's starting particles and one where BlackJAX re-evaluates
            # the moved particles.
            "n_likelihood_evaluations": self._n_likelihood_evaluations(
                n_initial_passes=1, n_extra_passes_per_iteration=2
            ),
            **self._mutation_history_diagnostics(),
            "persistent_log_Z": log_Z_traj,
            "log_Z": float(log_Z_traj[-1]),
            "ess_history": diagnostics.persistent_ess_history(ps, self._n_iterations),
            "log_Z_error": diagnostics.persistent_log_z_error(ps, self._n_iterations),
        }
