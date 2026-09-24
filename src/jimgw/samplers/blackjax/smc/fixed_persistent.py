"""Mode FP: persistent SMC over an explicit temperature ladder (``persistent_sampling_smc``)."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from blackjax import persistent_sampling_smc, rmh
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


class FixedPersistentSMCSampler(_BlackJAXSMCBase):
    """BlackJAX SMC, fixed temperature ladder with persistent sampling."""

    mode = "fp"

    def _run(self, rng_key: Key, initial_particles) -> None:
        config = self._config
        ladder = config.temperature_ladder
        assert ladder is not None
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        tempering_parameters = ladder[1:]  # skip 0.0 (already in init state)
        n_temperature_steps = len(tempering_parameters)
        checkpoint_path, run_start_time = prepare_checkpointing(config)
        log_label = f"{self.sampler_name} ({self.mode.upper()})"

        mcmc_step = self._build_mcmc_step()
        initial_covariance = (
            jnp.atleast_2d(jnp.cov(initial_particles.T)) * config.initial_cov_scale
        )

        smc_algorithm = persistent_sampling_smc(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            n_schedule=n_temperature_steps,
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
                initial_extra={"accept_history": []},
                load_extra=lambda checkpoint: {
                    "accept_history": list(checkpoint["accept_history"])
                },
                log_label=log_label,
                is_stale=lambda n_iter: n_iter > n_temperature_steps,
                stale_message="checkpoint n_iter exceeds current schedule length"
                f"={n_temperature_steps}",
            )
        )
        acceptance_history: list[float] = mode_checkpoint_data["accept_history"]

        run_temperature_step = jax.jit(smc_algorithm.step)
        last_checkpoint_write_time = time.perf_counter()

        for tempering_parameter in tempering_parameters[n_completed_iterations:]:
            rng_key, step_key = jax.random.split(rng_key)
            state, info = run_temperature_step(step_key, state, tempering_parameter)
            acceptance_history.append(float(info.update_info.acceptance_rate.mean()))
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
                self._checkpoint_extra(accept_history=acceptance_history.copy()),
                log_label=log_label,
            )

        self._final_state = state
        self._n_iterations = n_temperature_steps
        self._acceptance_history = np.asarray(acceptance_history)
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
        return diagnostics.resample_persistent_particles(self._final_state, self.n_dims)

    def _get_diagnostics(self) -> dict[str, object]:
        """Return SMC run diagnostics.

        Returns a dict with keys ``"n_likelihood_evaluations"``,
        ``"acceptance_history"``, ``"persistent_log_Z"``, ``"log_Z"``,
        ``"ess_history"``, ``"log_Z_error"``.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")
        cfg = self._config
        n_mcmc = cfg.n_mcmc_steps_per_dim * self.n_dims
        n_iter = self._n_iterations

        ps = self._final_state
        n = int(ps.iteration)
        log_Z_traj = np.asarray(ps.persistent_log_Z[1 : n + 1])

        return {
            "n_likelihood_evaluations": n_mcmc * n_iter * cfg.n_particles,
            "acceptance_history": self._acceptance_history,
            "persistent_log_Z": log_Z_traj,
            "log_Z": float(log_Z_traj[-1]),
            "ess_history": diagnostics.persistent_ess_history(ps, self._n_iterations),
            "log_Z_error": diagnostics.persistent_log_z_error(ps, self._n_iterations),
        }
