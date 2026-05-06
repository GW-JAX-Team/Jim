"""BlackJAX SMC samplers for Jim.

Supports four mode combinations selected by
[`BlackJAXSMCConfig`][jimgw.samplers.config.BlackJAXSMCConfig]:

* ``persistent_sampling=True,  temperature_ladder=None``  → adaptive persistent SMC
* ``persistent_sampling=True,  temperature_ladder=given`` → fixed-ladder persistent SMC
* ``persistent_sampling=False, temperature_ladder=None``  → adaptive tempered SMC
* ``persistent_sampling=False, temperature_ladder=given`` → fixed-ladder tempered SMC
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

from blackjax import (
    adaptive_persistent_sampling_smc,
    adaptive_tempered_smc,
    inner_kernel_tuning,
    persistent_sampling_smc,
    rmh,
    tempered_smc,
)
from blackjax.mcmc import random_walk
from blackjax.smc import extend_params
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.smc.persistent_sampling import (
    compute_log_persistent_weights,
    compute_persistent_ess,
)
from blackjax.smc.resampling import systematic

from jimgw.samplers.base import Sampler
from jimgw.samplers.config import BlackJAXSMCConfig
from jimgw.samplers.periodic import to_displacement_wrapper

# Fixed key used for post-sampling resampling in get_samples().
_RESAMPLE_KEY = jax.random.key(123)


class BlackJAXSMCSampler(Sampler):
    """BlackJAX SMC sampler.

    Uses a Gaussian random-walk MCMC inner kernel with initial covariance
    estimated from the starting particles.  With adaptive temperature
    selection the covariance is re-estimated at each step.

    Operates on flat ``(n_dims,)`` arrays.

    Args:
        n_dims: Dimension of the sampling space.
        log_prior_fn: Log-prior callable ``(arr,) -> float``.
        log_likelihood_fn: Log-likelihood callable ``(arr,) -> float``.
        log_posterior_fn: Log-posterior callable ``(arr,) -> float``.
        config: Optional :class:`BlackJAXSMCConfig`; defaults to all-default values.
        periodic: Optional periodic-parameter spec in index space,
            ``dict[int, (lo, hi)]`` where the key is the dimension index and
            the value is the ``(lower, upper)`` period bounds.  ``None`` means
            no periodic parameters.  Provided by Jim after resolving names.
    """

    _config: BlackJAXSMCConfig
    _displacement_wrapper: Callable
    _final_state: Any
    # Mode tag set in sample() so get_samples() / get_diagnostics() know which path was taken.
    _mode: str  # "ap" | "fp" | "at" | "ft"
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
    _scan_infos: Any  # fixed ladder: stacked SMCInfo from jax.lax.scan

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mcmc_step(self):
        """Return a GRW step callable ``(key, state, logdensity, cov) -> (state, info)``."""
        displacement_wrapper = self._displacement_wrapper
        kernel = random_walk.build_additive_step()

        def step(key, state, logdensity, cov):
            def proposal_distribution(key, position):
                raw_disp = jax.random.multivariate_normal(
                    key, jnp.zeros_like(position), cov
                )
                return displacement_wrapper(raw_disp, position)

            return kernel(key, state, logdensity, proposal_distribution)

        return step

    # ------------------------------------------------------------------
    # Mode runners
    # ------------------------------------------------------------------

    def _run_adaptive_persistent(self, rng_key: Key, initial_particles) -> None:
        """Mode AP: adaptive_persistent_sampling_smc + inner_kernel_tuning + while_loop."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config._resolve_target_ess_fraction()
        max_iterations = 1000

        mcmc_step = self._build_mcmc_step()
        cov0 = jnp.atleast_2d(jnp.cov(initial_particles.T)) * config.initial_cov_scale

        def mcmc_parameter_update_fn(_key, state, _info):
            return extend_params({"cov": jnp.atleast_2d(jnp.cov(state.particles.T))})  # type: ignore[arg-type]  # blackjax fork stubs: extend_params accepts dict

        smc_alg = inner_kernel_tuning(
            smc_algorithm=adaptive_persistent_sampling_smc,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            max_iterations=max_iterations,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params({"cov": cov0}),  # type: ignore[arg-type]  # blackjax fork stubs: extend_params accepts dict
            num_mcmc_steps=n_mcmc_steps,
            target_ess=target_ess,
        )

        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API: init takes only particles

        accept_history = jnp.zeros(max_iterations)
        cov_scale_history = jnp.zeros(max_iterations)

        def cond_fn(carry: tuple) -> Any:
            s = carry[0]
            return s.sampler_state.tempering_param < 1.0

        def body_fn(carry: tuple) -> tuple:
            s, key, cov_scale, n_iter, accept_h, cov_scale_h = carry
            key, subkey = jax.random.split(key)
            s, info = smc_alg.step(subkey, s)

            ps = s.sampler_state  # type: ignore[attr-defined]  # blackjax fork stubs
            acceptance_rate = info.update_info.acceptance_rate.mean()  # type: ignore[attr-defined]  # blackjax fork stubs

            new_scale = jnp.exp(
                jnp.log(cov_scale)
                + config.scale_adaptation_gain
                * (acceptance_rate - config.target_acceptance_rate)
            )
            current_cov = s.parameter_override["cov"]  # type: ignore[attr-defined]  # blackjax fork stubs
            new_params = extend_params({"cov": current_cov[0] * new_scale})  # type: ignore[arg-type]  # blackjax fork stubs
            s = StateWithParameterOverride(ps, new_params)  # type: ignore[arg-type]  # blackjax fork stubs

            accept_h = accept_h.at[n_iter].set(acceptance_rate)
            cov_scale_h = cov_scale_h.at[n_iter].set(new_scale)

            return (s, key, new_scale, n_iter + 1, accept_h, cov_scale_h)

        state, _, _, n_iter, accept_h, cov_scale_h = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                state,
                rng_key,
                jnp.array(config.initial_cov_scale),
                jnp.array(0),
                accept_history,
                cov_scale_history,
            ),
        )

        n_iter_int = int(n_iter)
        self._final_state = state
        self._mode = "ap"
        self._n_iterations = n_iter_int
        self._acceptance_history = np.asarray(accept_h[:n_iter_int])
        self._cov_scale_history = np.asarray(cov_scale_h[:n_iter_int])

    def _run_fixed_persistent(
        self, rng_key: Key, initial_particles, ladder: list[float]
    ) -> None:
        """persistent_sampling=True, fixed temperature ladder."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        lambdas = jnp.array(ladder[1:])  # skip 0.0 (already in init state)
        n_schedule = len(ladder) - 1

        mcmc_step = self._build_mcmc_step()
        cov0 = jnp.atleast_2d(jnp.cov(initial_particles.T)) * config.initial_cov_scale

        smc_alg = persistent_sampling_smc(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            n_schedule=n_schedule,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            mcmc_parameters=extend_params({"cov": cov0}),  # type: ignore[arg-type]  # blackjax fork stubs: extend_params accepts dict
            resampling_fn=systematic,
            num_mcmc_steps=n_mcmc_steps,
        )

        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API

        def scan_body(carry, lmbda):
            s, key = carry
            key, subkey = jax.random.split(key)
            s, info = smc_alg.step(subkey, s, lmbda)  # type: ignore[call-arg]  # blackjax fork API: step accepts extra arg
            return (s, key), info

        (state, _), scan_infos = jax.lax.scan(scan_body, (state, rng_key), lambdas)

        self._final_state = state
        self._mode = "fp"
        self._n_iterations = n_schedule
        self._scan_infos = scan_infos

    def _run_adaptive_tempered(self, rng_key: Key, initial_particles) -> None:
        """persistent_sampling=False, adaptive temperature selection."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config._resolve_target_ess_fraction()
        max_iterations = 1000

        mcmc_step = self._build_mcmc_step()
        cov0 = jnp.atleast_2d(jnp.cov(initial_particles.T)) * config.initial_cov_scale

        def mcmc_parameter_update_fn(_key, state, _info):
            return extend_params({"cov": jnp.atleast_2d(jnp.cov(state.particles.T))})  # type: ignore[arg-type]  # blackjax fork stubs: extend_params accepts dict

        smc_alg = inner_kernel_tuning(
            smc_algorithm=adaptive_tempered_smc,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params({"cov": cov0}),  # type: ignore[arg-type]  # blackjax fork stubs: extend_params accepts dict
            num_mcmc_steps=n_mcmc_steps,
            target_ess=target_ess,
        )

        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API

        accept_history = jnp.zeros(max_iterations)
        temp_history = jnp.zeros(max_iterations)
        is_weights_history = jnp.zeros((max_iterations, initial_particles.shape[0]))

        def cond_fn(carry: tuple) -> Any:
            s = carry[0]
            return s.sampler_state.tempering_param < 1.0

        def body_fn(carry: tuple) -> tuple:
            s, key, n_iter, accept_h, temp_h, is_weights_h = carry
            key, subkey = jax.random.split(key)
            s, info = smc_alg.step(subkey, s)

            acceptance_rate = info.update_info.acceptance_rate.mean()  # type: ignore[attr-defined]  # blackjax fork stubs
            tempering_param = s.sampler_state.tempering_param  # type: ignore[attr-defined]  # blackjax fork stubs

            accept_h = accept_h.at[n_iter].set(acceptance_rate)
            temp_h = temp_h.at[n_iter].set(tempering_param)
            is_weights_h = is_weights_h.at[n_iter].set(s.sampler_state.weights)  # type: ignore[attr-defined]

            return (s, key, n_iter + 1, accept_h, temp_h, is_weights_h)

        state, _, n_iter, accept_h, temp_h, is_weights_h = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                state,
                rng_key,
                jnp.array(0),
                accept_history,
                temp_history,
                is_weights_history,
            ),
        )

        n_iter_int = int(n_iter)
        self._final_state = state
        self._mode = "at"
        self._n_iterations = n_iter_int
        self._acceptance_history = np.asarray(accept_h[:n_iter_int])
        self._tempering_schedule = np.asarray(temp_h[:n_iter_int])
        self._is_weights_history = np.asarray(is_weights_h[:n_iter_int])

    def _run_fixed_tempered(
        self, rng_key: Key, initial_particles, ladder: list[float]
    ) -> None:
        """Mode FT: tempered_smc + scan over explicit temperature ladder."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        lambdas = jnp.array(ladder[1:])  # skip 0.0

        mcmc_step = self._build_mcmc_step()
        cov0 = jnp.atleast_2d(jnp.cov(initial_particles.T)) * config.initial_cov_scale

        smc_alg = tempered_smc(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            mcmc_parameters=extend_params({"cov": cov0}),  # type: ignore[arg-type]  # blackjax fork stubs: extend_params accepts dict
            resampling_fn=systematic,
            num_mcmc_steps=n_mcmc_steps,
        )

        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API

        def scan_body(carry, lmbda):
            s, key = carry
            key, subkey = jax.random.split(key)
            s, info = smc_alg.step(subkey, s, lmbda)  # type: ignore[call-arg]  # blackjax fork API: step accepts extra arg
            return (s, key), (info, s.weights)

        (state, _), (scan_infos, is_weights_h) = jax.lax.scan(
            scan_body, (state, rng_key), lambdas
        )

        self._final_state = state
        self._mode = "ft"
        self._n_iterations = len(ladder) - 1
        self._scan_infos = scan_infos
        self._is_weights_history = np.asarray(is_weights_h)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_particles n_dims"],
    ) -> None:
        """Run the BlackJAX SMC sampler.

        Args:
            rng_key: JAX PRNG key.
            initial_position: Starting particles in the sampling space,
                shape ``(n_particles, n_dims)``.  Must match ``config.n_particles``.

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
        persistent = config.persistent_sampling

        if persistent and ladder is None:
            self._run_adaptive_persistent(rng_key, initial_particles)
        elif persistent and ladder is not None:
            self._run_fixed_persistent(rng_key, initial_particles, ladder)
        elif not persistent and ladder is None:
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

        mode = self._mode
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
            log_likelihoods = np.array(jax.vmap(self._log_likelihood_fn)(ps.particles))
            return {"samples": final_particles, "log_likelihood": log_likelihoods}

        else:  # mode == "ft"
            final_particles = np.array(state.particles)
            log_likelihoods = np.array(
                jax.vmap(self._log_likelihood_fn)(state.particles)
            )
            return {"samples": final_particles, "log_likelihood": log_likelihoods}

    def _get_diagnostics(self) -> dict[str, Any]:
        """Return SMC run diagnostics.

        Returns a dict with the following keys (not all present for all modes):

        * ``"n_likelihood_evaluations"`` — total likelihood calls.
        * ``"acceptance_history"`` — per-iteration mean acceptance rate; length ``n_iterations``.
        * ``"n_iterations"`` — total SMC iterations (adaptive modes only).
        * ``"tempering_schedule"`` — inverse temperature at each iteration; length ``n_iterations`` (adaptive modes only).
        * ``"cov_scale_history"`` — covariance scale per iteration (adaptive-persistent only); length ``n_iterations``.
        * ``"ess_history"`` — ESS per iteration (all modes: persistent ESS for ap/fp, Kish ESS for at/ft); length ``n_iterations``.
        * ``"persistent_log_Z"`` — cumulative log-Z after each iteration; length ``n_iterations`` (persistent modes only).
        * ``"log_Z"`` — final log Bayesian evidence (persistent modes only).
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")

        cfg = self._config
        mode = self._mode
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
            result["acceptance_history"] = np.asarray(
                self._scan_infos.update_info.acceptance_rate.mean(axis=-1)
            )
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
                    ps.persistent_log_likelihoods,  # type: ignore[attr-defined]
                    ps.persistent_log_Z,  # type: ignore[attr-defined]
                    ps.tempering_schedule,  # type: ignore[attr-defined]
                    t,
                    include_current=True,
                )
                ess_hist[t - 1] = float(
                    compute_persistent_ess(log_w.reshape(-1), normalize_weights=True)
                )
            result["ess_history"] = ess_hist

        if mode in ("at", "ft"):
            # IS weights are already normalized; Kish ESS = 1/sum(w^2)
            w = self._is_weights_history
            result["ess_history"] = 1.0 / np.sum(w**2, axis=-1)

        return result
