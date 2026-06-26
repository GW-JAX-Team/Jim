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

logger = logging.getLogger(__name__)

# Fixed key used for post-sampling resampling in get_samples().
_RESAMPLE_KEY = jax.random.key(123)


class BlackJAXSMCSampler(Sampler):
    """BlackJAX SMC sampler.

    Uses a Gaussian random-walk MCMC inner kernel with initial covariance
    estimated from the starting particles.  With adaptive temperature
    selection the covariance is re-estimated at each step.

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
        """Mode AP: adaptive_persistent_sampling_smc + inner_kernel_tuning + Python while."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config._resolve_target_ess_fraction()
        ckpt_path = (
            config.checkpoint_dir / "checkpoint.pkl"
            if config.checkpoint_dir is not None
            else None
        )
        config.configure_jax_cache()
        _method_t0 = time.perf_counter()

        mcmc_step = self._build_mcmc_step()
        cov0 = jnp.atleast_2d(jnp.cov(initial_particles.T)) * config.initial_cov_scale

        def mcmc_parameter_update_fn(_key, state, _info):
            return extend_params({"cov": jnp.atleast_2d(jnp.cov(state.particles.T))})  # type: ignore[arg-type]  # blackjax fork stubs: extend_params accepts dict

        smc_alg = inner_kernel_tuning(
            smc_algorithm=adaptive_persistent_sampling_smc,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            max_iterations=1000,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params({"cov": cov0}),  # type: ignore[arg-type]  # blackjax fork stubs: extend_params accepts dict
            num_mcmc_steps=n_mcmc_steps,
            target_ess=target_ess,
            batch_size=config.batch_size,
        )

        accept_list: list[float] = []
        cov_scale_list: list[float] = []
        cov_scale = float(config.initial_cov_scale)
        n_iter = 0

        # Resume from checkpoint if one exists.
        if (
            ckpt_path is not None
            and config.checkpoint_interval > 0
            and ckpt_path.exists()
        ):
            try:
                with open(ckpt_path, "rb") as _f:
                    _ckpt = pickle.load(
                        _f
                    )  # Only load trusted checkpoints — pickle executes arbitrary code.
                if _ckpt.get("mode") == "ap":
                    state = _ckpt["state"]
                    rng_key = _ckpt["rng_key"]
                    n_iter = _ckpt["n_iter"]
                    cov_scale = float(_ckpt.get("cov_scale", cov_scale))
                    accept_list = list(_ckpt["accept_history"])
                    cov_scale_list = list(_ckpt["cov_scale_history"])
                    self._prev_elapsed = float(_ckpt["elapsed_time"])
                    logger.info(
                        "SMC-AP: resumed from checkpoint at n_iter=%d (%s)",
                        n_iter,
                        ckpt_path,
                    )
                else:
                    state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                    self._prev_elapsed = 0.0
            except (
                OSError,
                EOFError,
                KeyError,
                ValueError,
                pickle.UnpicklingError,
            ) as _e:
                logger.warning(
                    "SMC-AP: corrupt checkpoint at %s (%s) — starting fresh.",
                    ckpt_path,
                    _e,
                )
                state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                self._prev_elapsed = 0.0
        else:
            state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API

        step_fn = jax.jit(smc_alg.step)
        _last_ckpt_t = time.perf_counter()

        while state.sampler_state.tempering_param < 1.0:  # type: ignore[attr-defined]  # blackjax fork stubs
            rng_key, subkey = jax.random.split(rng_key)
            state, info = step_fn(subkey, state)

            ps = state.sampler_state  # type: ignore[attr-defined]  # blackjax fork stubs
            acceptance_rate = float(info.update_info.acceptance_rate.mean())  # type: ignore[attr-defined]  # blackjax fork stubs
            new_scale = cov_scale * float(
                jnp.exp(
                    config.scale_adaptation_gain
                    * (acceptance_rate - config.target_acceptance_rate)
                )
            )
            current_cov = state.parameter_override["cov"]  # type: ignore[attr-defined]  # blackjax fork stubs
            new_params = extend_params({"cov": current_cov[0] * new_scale})  # type: ignore[arg-type]  # blackjax fork stubs
            state = StateWithParameterOverride(ps, new_params)  # type: ignore[arg-type]  # blackjax fork stubs

            accept_list.append(acceptance_rate)
            cov_scale_list.append(new_scale)
            cov_scale = new_scale
            n_iter += 1

            if (
                ckpt_path is not None
                and config.checkpoint_interval > 0
                and time.perf_counter() - _last_ckpt_t >= config.checkpoint_interval
            ):
                _last_ckpt_t = config.write_checkpoint(
                    {
                        "state": state,
                        "rng_key": rng_key,
                        "n_iter": n_iter,
                        "mode": "ap",
                        "elapsed_time": self._prev_elapsed
                        + (time.perf_counter() - _method_t0),
                        "cov_scale": cov_scale,
                        "accept_history": accept_list.copy(),
                        "cov_scale_history": cov_scale_list.copy(),
                    },
                    "SMC-AP",
                )

        self._final_state = state
        self._mode = "ap"
        self._n_iterations = n_iter
        self._acceptance_history = np.asarray(accept_list)
        self._cov_scale_history = np.asarray(cov_scale_list)
        if ckpt_path is not None:
            ckpt_path.unlink(missing_ok=True)
        if config.checkpoint_dir is not None:
            shutil.rmtree(config.checkpoint_dir / "jax_cache", ignore_errors=True)
            jax.config.update("jax_compilation_cache_dir", None)

    def _run_fixed_persistent(
        self, rng_key: Key, initial_particles, ladder: list[float]
    ) -> None:
        """Mode FP: persistent_sampling_smc over an explicit temperature ladder — Python for loop."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        ladder_values = ladder[1:]  # skip 0.0 (already in init state)
        n_schedule = len(ladder_values)
        ckpt_path = (
            config.checkpoint_dir / "checkpoint.pkl"
            if config.checkpoint_dir is not None
            else None
        )
        config.configure_jax_cache()
        _method_t0 = time.perf_counter()

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
            batch_size=config.batch_size,
        )

        accept_list: list[float] = []
        n_iter = 0

        # Resume from checkpoint if one exists.
        if (
            ckpt_path is not None
            and config.checkpoint_interval > 0
            and ckpt_path.exists()
        ):
            try:
                with open(ckpt_path, "rb") as _f:
                    _ckpt = pickle.load(_f)
                if _ckpt.get("mode") == "fp":
                    state = _ckpt["state"]
                    rng_key = _ckpt["rng_key"]
                    n_iter = _ckpt["n_iter"]
                    accept_list = list(_ckpt["accept_history"])
                    if n_iter > n_schedule:
                        logger.warning(
                            "SMC-FP: checkpoint n_iter=%d exceeds current schedule length=%d — starting fresh.",
                            n_iter,
                            n_schedule,
                        )
                        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                        n_iter = 0
                        accept_list = []
                        self._prev_elapsed = 0.0
                    else:
                        self._prev_elapsed = float(_ckpt["elapsed_time"])
                        logger.info(
                            "SMC-FP: resumed from checkpoint at n_iter=%d (%s)",
                            n_iter,
                            ckpt_path,
                        )
                else:
                    state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                    self._prev_elapsed = 0.0
            except (
                OSError,
                EOFError,
                KeyError,
                ValueError,
                pickle.UnpicklingError,
            ) as _e:
                logger.warning(
                    "SMC-FP: corrupt checkpoint at %s (%s) — starting fresh.",
                    ckpt_path,
                    _e,
                )
                state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                self._prev_elapsed = 0.0
        else:
            state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API

        step_fn = jax.jit(smc_alg.step)
        _last_ckpt_t = time.perf_counter()

        for lmbda in ladder_values[n_iter:]:
            rng_key, subkey = jax.random.split(rng_key)
            state, info = step_fn(subkey, state, lmbda)  # type: ignore[call-arg]  # blackjax fork API: step accepts extra arg
            accept_list.append(float(info.update_info.acceptance_rate.mean()))
            n_iter += 1
            if (
                ckpt_path is not None
                and config.checkpoint_interval > 0
                and time.perf_counter() - _last_ckpt_t >= config.checkpoint_interval
            ):
                _last_ckpt_t = config.write_checkpoint(
                    {
                        "state": state,
                        "rng_key": rng_key,
                        "n_iter": n_iter,
                        "mode": "fp",
                        "elapsed_time": self._prev_elapsed
                        + (time.perf_counter() - _method_t0),
                        "accept_history": accept_list.copy(),
                    },
                    "SMC-FP",
                )

        self._final_state = state
        self._mode = "fp"
        self._n_iterations = n_schedule
        self._acceptance_history = np.asarray(accept_list)
        if ckpt_path is not None:
            ckpt_path.unlink(missing_ok=True)
        if config.checkpoint_dir is not None:
            shutil.rmtree(config.checkpoint_dir / "jax_cache", ignore_errors=True)
            jax.config.update("jax_compilation_cache_dir", None)

    def _run_adaptive_tempered(self, rng_key: Key, initial_particles) -> None:
        """Mode AT: adaptive_tempered_smc + inner_kernel_tuning + Python while."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config._resolve_target_ess_fraction()
        ckpt_path = (
            config.checkpoint_dir / "checkpoint.pkl"
            if config.checkpoint_dir is not None
            else None
        )
        config.configure_jax_cache()
        _method_t0 = time.perf_counter()

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
            batch_size=config.batch_size,
        )

        accept_list: list[float] = []
        temp_list: list[float] = []
        is_weights_list: list[np.ndarray] = []
        n_iter = 0

        # Resume from checkpoint if one exists.
        if (
            ckpt_path is not None
            and config.checkpoint_interval > 0
            and ckpt_path.exists()
        ):
            try:
                with open(ckpt_path, "rb") as _f:
                    _ckpt = pickle.load(_f)
                if _ckpt.get("mode") == "at":
                    state = _ckpt["state"]
                    rng_key = _ckpt["rng_key"]
                    n_iter = _ckpt["n_iter"]
                    accept_list = list(_ckpt["accept_history"])
                    temp_list = list(_ckpt["tempering_schedule"])
                    is_weights_list = list(_ckpt["is_weights_history"])
                    self._prev_elapsed = float(_ckpt["elapsed_time"])
                    logger.info(
                        "SMC-AT: resumed from checkpoint at n_iter=%d (%s)",
                        n_iter,
                        ckpt_path,
                    )
                else:
                    state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                    self._prev_elapsed = 0.0
            except (
                OSError,
                EOFError,
                KeyError,
                ValueError,
                pickle.UnpicklingError,
            ) as _e:
                logger.warning(
                    "SMC-AT: corrupt checkpoint at %s (%s) — starting fresh.",
                    ckpt_path,
                    _e,
                )
                state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                self._prev_elapsed = 0.0
        else:
            state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API

        step_fn = jax.jit(smc_alg.step)
        _last_ckpt_t = time.perf_counter()

        while state.sampler_state.tempering_param < 1.0:  # type: ignore[attr-defined]  # blackjax fork stubs
            rng_key, subkey = jax.random.split(rng_key)
            state, info = step_fn(subkey, state)

            accept_list.append(float(info.update_info.acceptance_rate.mean()))  # type: ignore[attr-defined]  # blackjax fork stubs
            temp_list.append(float(state.sampler_state.tempering_param))  # type: ignore[attr-defined]  # blackjax fork stubs
            is_weights_list.append(np.asarray(state.sampler_state.weights))  # type: ignore[attr-defined]
            n_iter += 1

            if (
                ckpt_path is not None
                and config.checkpoint_interval > 0
                and time.perf_counter() - _last_ckpt_t >= config.checkpoint_interval
            ):
                _last_ckpt_t = config.write_checkpoint(
                    {
                        "state": state,
                        "rng_key": rng_key,
                        "n_iter": n_iter,
                        "mode": "at",
                        "elapsed_time": self._prev_elapsed
                        + (time.perf_counter() - _method_t0),
                        "accept_history": accept_list.copy(),
                        "tempering_schedule": temp_list.copy(),
                        "is_weights_history": np.stack(is_weights_list),
                    },
                    "SMC-AT",
                )

        self._final_state = state
        self._mode = "at"
        self._n_iterations = n_iter
        self._acceptance_history = np.asarray(accept_list)
        self._tempering_schedule = np.asarray(temp_list)
        self._is_weights_history = (
            np.stack(is_weights_list)
            if is_weights_list
            else np.empty((0, initial_particles.shape[0]))
        )
        if ckpt_path is not None:
            ckpt_path.unlink(missing_ok=True)
        if config.checkpoint_dir is not None:
            shutil.rmtree(config.checkpoint_dir / "jax_cache", ignore_errors=True)
            jax.config.update("jax_compilation_cache_dir", None)

    def _run_fixed_tempered(
        self, rng_key: Key, initial_particles, ladder: list[float]
    ) -> None:
        """Mode FT: tempered_smc + Python for loop over explicit temperature ladder."""
        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        ladder_values = ladder[1:]  # skip 0.0
        n_schedule = len(ladder_values)
        ckpt_path = (
            config.checkpoint_dir / "checkpoint.pkl"
            if config.checkpoint_dir is not None
            else None
        )
        config.configure_jax_cache()
        _method_t0 = time.perf_counter()

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
            batch_size=config.batch_size,
        )

        accept_list: list[float] = []
        is_weights_list: list[np.ndarray] = []
        n_iter = 0

        # Resume from checkpoint if one exists.
        if (
            ckpt_path is not None
            and config.checkpoint_interval > 0
            and ckpt_path.exists()
        ):
            try:
                with open(ckpt_path, "rb") as _f:
                    _ckpt = pickle.load(_f)
                if _ckpt.get("mode") == "ft":
                    state = _ckpt["state"]
                    rng_key = _ckpt["rng_key"]
                    n_iter = _ckpt["n_iter"]
                    accept_list = list(_ckpt["accept_history"])
                    is_weights_list = list(_ckpt["is_weights_history"])
                    if n_iter > n_schedule:
                        logger.warning(
                            "SMC-FT: checkpoint n_iter=%d exceeds current schedule length=%d — starting fresh.",
                            n_iter,
                            n_schedule,
                        )
                        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                        n_iter = 0
                        accept_list = []
                        is_weights_list = []
                        self._prev_elapsed = 0.0
                    else:
                        self._prev_elapsed = float(_ckpt["elapsed_time"])
                        logger.info(
                            "SMC-FT: resumed from checkpoint at n_iter=%d (%s)",
                            n_iter,
                            ckpt_path,
                        )
                else:
                    state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                    self._prev_elapsed = 0.0
            except (
                OSError,
                EOFError,
                KeyError,
                ValueError,
                pickle.UnpicklingError,
            ) as _e:
                logger.warning(
                    "SMC-FT: corrupt checkpoint at %s (%s) — starting fresh.",
                    ckpt_path,
                    _e,
                )
                state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API
                self._prev_elapsed = 0.0
        else:
            state = smc_alg.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API

        step_fn = jax.jit(smc_alg.step)
        _last_ckpt_t = time.perf_counter()

        for lmbda in ladder_values[n_iter:]:
            rng_key, subkey = jax.random.split(rng_key)
            state, info = step_fn(subkey, state, lmbda)  # type: ignore[call-arg]  # blackjax fork API: step accepts extra arg
            accept_list.append(float(info.update_info.acceptance_rate.mean()))
            is_weights_list.append(np.asarray(state.weights))
            n_iter += 1
            if (
                ckpt_path is not None
                and config.checkpoint_interval > 0
                and time.perf_counter() - _last_ckpt_t >= config.checkpoint_interval
            ):
                _last_ckpt_t = config.write_checkpoint(
                    {
                        "state": state,
                        "rng_key": rng_key,
                        "n_iter": n_iter,
                        "mode": "ft",
                        "elapsed_time": self._prev_elapsed
                        + (time.perf_counter() - _method_t0),
                        "accept_history": accept_list.copy(),
                        "is_weights_history": np.stack(is_weights_list),
                    },
                    "SMC-FT",
                )

        self._final_state = state
        self._mode = "ft"
        self._n_iterations = n_schedule
        self._acceptance_history = np.asarray(accept_list)
        self._is_weights_history = (
            np.stack(is_weights_list)
            if is_weights_list
            else np.empty((0, initial_particles.shape[0]))
        )
        if ckpt_path is not None:
            ckpt_path.unlink(missing_ok=True)
        if config.checkpoint_dir is not None:
            shutil.rmtree(config.checkpoint_dir / "jax_cache", ignore_errors=True)
            jax.config.update("jax_compilation_cache_dir", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        * ``"cov_scale_history"`` — covariance scale per iteration (adaptive-persistent only); length ``n_iterations``.
        * ``"ess_history"`` — ESS per iteration (all modes: persistent ESS for ap/fp, Kish ESS for at/ft); length ``n_iterations``.
        * ``"persistent_log_Z"`` — cumulative log-Z after each iteration; length ``n_iterations`` (persistent modes only).
        * ``"log_Z"`` — final log Bayesian evidence (persistent modes only).
        * ``"log_Z_error"`` — standard deviation of log Z from delta-method IS weight variance (all modes).
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
