"""FlowMC-backed sampler for Jim.

Wraps the flowMC `Sampler` configured with a rational-quadratic
spline normalizing flow and a choice of local MCMC kernel, with optional
parallel tempering.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
from flowMC.resource_strategy_bundle.RQSpline_GRW import RQSpline_GRW_Bundle
from flowMC.resource_strategy_bundle.RQSpline_GRW_PT import RQSpline_GRW_PT_Bundle
from flowMC.resource_strategy_bundle.RQSpline_HMC import RQSpline_HMC_Bundle
from flowMC.resource_strategy_bundle.RQSpline_HMC_PT import RQSpline_HMC_PT_Bundle
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle
from flowMC.Sampler import Sampler as FlowMCSamplerBackend
from jaxtyping import Array, Float, Key

from jimgw.samplers.base import Sampler
from jimgw.samplers.config import FlowMCConfig

logger = logging.getLogger(__name__)

# Maps (local_kernel, pt_enabled) → bundle class.
_BUNDLE: dict[tuple[str, bool], Type] = {
    ("MALA", False): RQSpline_MALA_Bundle,
    ("MALA", True): RQSpline_MALA_PT_Bundle,
    ("HMC", False): RQSpline_HMC_Bundle,
    ("HMC", True): RQSpline_HMC_PT_Bundle,
    ("GRW", False): RQSpline_GRW_Bundle,
    ("GRW", True): RQSpline_GRW_PT_Bundle,
}


class FlowMCSampler(Sampler):
    """flowMC sampler backend.

    Wraps the flowMC `Sampler` with a rational-quadratic spline NF
    and a configurable local MCMC kernel (MALA, HMC, or GRW) with optional
    parallel tempering.  The flowMC bundle is built lazily inside
    `sample` so the PRNG key from Jim is used correctly (no duplication
    of the seed).

    Configured via [`FlowMCConfig`][jimgw.samplers.config.FlowMCConfig].

    Args:
        n_dims: Dimension of the sampling space.
        log_prior_fn: Log-prior callable ``(arr,) -> float``.
        log_likelihood_fn: Log-likelihood callable ``(arr,) -> float``.
        log_posterior_fn: Log-posterior callable ``(arr,) -> float``.
        config: Optional :class:`FlowMCConfig`; defaults to all-default values.
        periodic: Optional periodic-parameter spec in index space,
            ``dict[int, (lo, hi)]`` where the key is the dimension index.
            ``None`` means no periodic parameters.  Provided by Jim after
            resolving parameter names to indices.
    """

    _config: FlowMCConfig
    _periodic_index_dict: Optional[dict]
    _flowmc_sampler: Optional[FlowMCSamplerBackend]

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: Optional[FlowMCConfig] = None,
        periodic: Optional[dict[int, tuple[float, float]]] = None,
    ) -> None:
        if config is None:
            config = FlowMCConfig()
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )
        self._periodic_index_dict = periodic
        self._flowmc_sampler = None

        # Pre-compute strategy order for use before sampling.
        order = ["local_sampler", "normalizing_flow"]
        if config.parallel_tempering is not None:
            order.append("parallel_tempering")
        self._strategy_order_from_config: list[str] = order

    # flowMC expects callables with signature (params, data) -> Float.
    def _logpdf_flowmc(self, params: Float[Array, " n_dims"], _data: dict) -> Float:  # noqa: F722
        return self._log_posterior_fn(params)

    def _logprior_flowmc(self, params: Float[Array, " n_dims"], _data: dict) -> Float:  # noqa: F722
        return self._log_prior_fn(params)

    @property
    def strategy_order(self) -> list[str]:
        """Ordered list of flowMC strategies."""
        if self._flowmc_sampler is not None:
            order = self._flowmc_sampler.strategy_order
            if order is not None:
                return order
        return self._strategy_order_from_config

    def _sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_chains n_dims"],
    ) -> None:
        """Run the flowMC sampler.

        The flowMC bundle (NF + chosen local kernel + optional PT) is built
        here so that the PRNG key is derived from the key Jim passes in.

        Args:
            rng_key: JAX PRNG key for both bundle initialisation and sampling.
            initial_position: Starting positions in the sampling space.
                Accepted shapes:

                - ``(n_dims,)`` — broadcast to all chains.
                - ``(n_chains, n_dims)`` — one position per chain.
        """
        config = self._config
        rng_key, bundle_key, sampler_key = jax.random.split(rng_key, 3)

        bundle_cls = _BUNDLE[
            (config.local_kernel, config.parallel_tempering is not None)
        ]

        # Common kwargs for every bundle.
        common_kwargs: dict = dict(
            rng_key=bundle_key,
            n_chains=config.n_chains,
            n_dims=self.n_dims,
            logpdf=self._logpdf_flowmc,
            n_local_steps=config.n_local_steps,
            n_global_steps=config.n_global_steps,
            n_training_loops=config.n_training_loops,
            n_production_loops=config.n_production_loops,
            n_epochs=config.n_epochs,
            periodic=self._periodic_index_dict,
            rq_spline_hidden_units=config.rq_spline_hidden_units,
            rq_spline_n_bins=config.rq_spline_n_bins,
            rq_spline_n_layers=config.rq_spline_n_layers,
            n_NFproposal_batch_size=config.n_NFproposal_batch_size,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            n_max_examples=config.n_max_examples,
            history_window=config.history_window,
            chain_batch_size=config.chain_batch_size,
            local_thinning=config.local_thinning,
            global_thinning=config.global_thinning,
            early_stopping=config.early_stopping,
            early_stopping_tolerance=config.early_stopping_tolerance,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_acceptance=config.early_stopping_min_acceptance,
            verbose=config.verbose,
        )

        # Kernel-specific kwargs.
        if config.local_kernel == "MALA":
            common_kwargs["mala_step_size"] = config.mala.step_size
        elif config.local_kernel == "HMC":
            common_kwargs["hmc_step_size"] = config.hmc.step_size
            common_kwargs["hmc_n_leapfrog"] = config.hmc.n_leapfrog_steps
            common_kwargs["condition_matrix"] = config.hmc.condition_matrix
        elif config.local_kernel == "GRW":
            common_kwargs["grw_step_size"] = config.grw.step_size

        # PT-specific kwargs (only for PT bundles).
        if config.parallel_tempering is not None:
            pt = config.parallel_tempering
            common_kwargs["n_temperatures"] = pt.n_temperatures
            common_kwargs["max_temperature"] = pt.max_temperature
            common_kwargs["n_tempered_steps"] = pt.n_tempered_steps
            common_kwargs["logprior"] = self._logprior_flowmc

        resource_strategy_bundle = bundle_cls(**common_kwargs)

        self._flowmc_sampler = FlowMCSamplerBackend(
            self.n_dims,
            config.n_chains,
            sampler_key,
            resource_strategy_bundles=resource_strategy_bundle,
        )

        initial_position = jnp.asarray(initial_position)
        if initial_position.ndim == 1:
            if initial_position.shape[0] != self.n_dims:
                raise ValueError(
                    f"initial_position must have shape (n_dims,) or "
                    f"(n_chains, n_dims). Got shape {initial_position.shape}."
                )
            logger.info("1D initial_position provided. Broadcasting to all chains.")
            initial_position = jnp.broadcast_to(
                initial_position, (config.n_chains, self.n_dims)
            )
        elif initial_position.ndim == 2:
            if initial_position.shape != (config.n_chains, self.n_dims):
                raise ValueError(
                    f"initial_position must have shape (n_dims,) or "
                    f"(n_chains, n_dims). Got shape {initial_position.shape}."
                )
        else:
            raise ValueError(
                f"initial_position must have shape (n_dims,) or "
                f"(n_chains, n_dims). Got shape {initial_position.shape}."
            )

        self._flowmc_sampler.rng_key = rng_key
        self._flowmc_sampler.sample(initial_position, {})

    def get_samples(self) -> dict[str, np.ndarray]:
        """Return all production samples with their log-likelihoods.

        Production samples are flat arrays in sampling space, shape
        ``(N, n_dims)`` where ``N = n_chains * n_production_loops * n_total_steps``.
        Log-likelihoods are recovered from the stored log-posterior values as
        ``log_likelihood = log_posterior - log_prior``, avoiding a second
        evaluation of the likelihood function.

        Returns:
            Dict with keys ``"samples"`` (shape ``(N, n_dims)``) and
            ``"log_likelihood"`` (shape ``(N,)``).
        """
        if not self._sampled or self._flowmc_sampler is None:
            raise RuntimeError(
                "get_samples() called before sample(). Run sample() first."
            )
        resources = self._flowmc_sampler.resources

        pos_raw = resources["positions_production"].data  # type: ignore[union-attr]  # flowMC stubs
        pos_jnp = jnp.array(pos_raw).reshape(-1, self.n_dims)

        log_posterior = np.array(
            resources["log_prob_production"].data  # type: ignore[union-attr]  # flowMC stubs
        ).reshape(-1)
        log_prior = np.array(jax.vmap(self._log_prior_fn)(pos_jnp))
        log_likelihood = log_posterior - log_prior
        prod_positions = np.array(pos_jnp)

        return {"samples": prod_positions, "log_likelihood": log_likelihood}

    def _get_diagnostics(self) -> dict[str, Any]:
        """Return flowMC run diagnostics.

        Returns a dict with the following keys:

        * ``"n_likelihood_evaluations"`` — total likelihood calls.
        * ``"n_training_loops_actual"`` — training loops actually run
          (may be fewer than configured if early stopping triggered).
        * ``"training_loss_history"`` — normalizing-flow loss per epoch.
        * ``"acceptance_training_local"`` — per-loop local acceptance (training).
        * ``"acceptance_training_global"`` — per-loop global acceptance (training).
        * ``"acceptance_production_local"`` — per-loop local acceptance (production).
        * ``"acceptance_production_global"`` — per-loop global acceptance (production).
        """
        if not self._sampled or self._flowmc_sampler is None:
            raise RuntimeError("get_diagnostics() called before sample()")
        cfg = self._config
        res = self._flowmc_sampler.resources

        actual_training_loops = (
            self._flowmc_sampler.strategies["check_early_stop"]._call_count  # type: ignore[attr-defined]  # flowMC stubs: Strategy._call_count
            if "check_early_stop" in self._flowmc_sampler.strategies
            else cfg.n_training_loops
        )

        n_evals = int(
            cfg.n_chains
            * (cfg.n_local_steps + cfg.n_global_steps)
            * (actual_training_loops + cfg.n_production_loops)
        )

        return {
            "n_likelihood_evaluations": n_evals,
            "n_training_loops_actual": actual_training_loops,
            "training_loss_history": np.asarray(res["loss_buffer"].data),  # type: ignore[union-attr]  # flowMC stubs
            "acceptance_training_local": np.asarray(res["local_accs_training"].data),  # type: ignore[union-attr]  # flowMC stubs
            "acceptance_training_global": np.asarray(res["global_accs_training"].data),  # type: ignore[union-attr]  # flowMC stubs
            "acceptance_production_local": np.asarray(
                res["local_accs_production"].data  # type: ignore[union-attr]  # flowMC stubs
            ),
            "acceptance_production_global": np.asarray(
                res["global_accs_production"].data  # type: ignore[union-attr]  # flowMC stubs
            ),
        }
