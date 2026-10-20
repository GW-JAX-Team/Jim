"""Shared machinery for the BlackJAX SMC sampler modes.

``_BlackJAXSMCBase`` holds everything that doesn't depend on which of the four
SMC modes (adaptive/fixed temperature schedule x persistent/tempered
resampling) a concrete sampler implements: construction, checkpoint
validation, the mutation and its history. Each mode's own module
(``adaptive_persistent``, ``fixed_persistent``, ``adaptive_tempered``,
``fixed_tempered``) implements ``_run``/``get_samples``/``_get_diagnostics``.
"""

import logging
import time
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Optional

import jax.numpy as jnp
import numpy as np
from blackjax.smc import extend_params
from blackjax.smc.resampling import systematic
from jaxtyping import Array, Float, Key

from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax.smc.mutation import EnsembleMutation
from jimgw.samplers.config import BlackJAXSMCConfig
from jimgw.typing import FloatScalar

logger = logging.getLogger(__name__)


class _BlackJAXSMCBase(Sampler):
    """Shared base for the four BlackJAX SMC mode samplers.

    Each tempering iteration moves the particles with an
    [`EnsembleMutation`][jimgw.samplers.blackjax.smc.mutation.EnsembleMutation]:
    the configured kernel (t-pCN or a Gaussian random walk), optionally in the
    latent space of a normalizing flow trained on the particles.

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

    mode: ClassVar[str]
    _config: BlackJAXSMCConfig
    _mutation: EnsembleMutation
    # Any: one of several heterogeneous, duck-typed blackjax SMC state types; object breaks attribute access, since blackjax's own stubs don't expose them precisely either.
    _final_state: Any
    _n_iterations: int
    # Per-iteration record of each mutation quantity, keyed by diagnostics name.
    _mutation_history: dict[str, list]

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable[[Float[Array, " n_dims"]], FloatScalar],
        log_likelihood_fn: Callable[[Float[Array, " n_dims"]], FloatScalar],
        log_posterior_fn: Callable[[Float[Array, " n_dims"]], FloatScalar],
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
        self._mutation = EnsembleMutation(
            n_dims=n_dims,
            periodic=periodic,
            kernel_config=config.mcmc_kernel,
            flow_config=config.flow,
        )

    @property
    def sampler_name(self) -> str:
        return "BlackJAX SMC"

    def _validate_checkpoint(self, checkpoint: dict) -> None:
        """Raise when a checkpoint is incompatible with this SMC configuration."""
        super()._validate_checkpoint(checkpoint)
        checkpoint_mode = checkpoint.get("mode")
        if checkpoint_mode != self.mode:
            raise ValueError(
                "checkpoint belongs to a different SMC mode: "
                f"{checkpoint_mode or 'an unknown mode'}, not {self.mode}"
            )
        if checkpoint.get("mutation") != self._mutation_settings():
            raise ValueError(
                "checkpoint was written with different mcmc_kernel or flow settings"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _log_label(self) -> str:
        return f"{self.sampler_name} ({self.mode.upper()})"

    def _mutation_settings(self) -> dict[str, object]:
        """The kernel and flow settings that a checkpoint must match."""
        config = self._config
        return {
            "mcmc_kernel": config.mcmc_kernel.model_dump(),
            "flow": False if config.flow is False else config.flow.model_dump(),
        }

    def _checkpoint_extra(self, **mode_specific: object) -> dict[str, object]:
        """Checkpoint payload's mode metadata, plus mode-specific extras."""
        return {
            "mode": self.mode,
            "mutation": self._mutation_settings(),
            **mode_specific,
        }

    def _common_arguments(self) -> dict[str, Any]:
        """Arguments common to every mode's ``blackjax.inner_kernel_tuning`` call."""
        mutation = self._mutation
        return {
            "logprior_fn": self._log_prior_fn,
            "loglikelihood_fn": self._log_likelihood_fn,
            "mcmc_step_fn": mutation.read_parameters,
            "mcmc_init_fn": mutation.mcmc_init_fn,
            "resampling_fn": systematic,
            "mcmc_parameter_update_fn": mutation.mcmc_parameter_update_fn,
            "initial_parameter_value": extend_params(
                mutation.initial_parameter_value()  # ty: ignore[invalid-argument-type]  # blackjax stubs: extend_params accepts dict
            ),
            "num_mcmc_steps": mutation.max_steps,
            "update_strategy": mutation.update_strategy,
            "batch_size": self._config.batch_size,
        }

    def _new_mutation_history(self) -> dict[str, list]:
        """An empty mutation history, keyed by diagnostics name."""
        history: dict[str, list] = {
            "acceptance_history": [],
            "n_steps_history": [],
            "scale_history": [],
        }
        if self._config.flow is not False:
            history["training_loss_history"] = []
        return history

    def _copy_mutation_history(self, history: dict[str, list]) -> dict[str, list]:
        """A copy of the mutation history for a checkpoint."""
        return {name: list(values) for name, values in history.items()}

    def _record_mutation_history(
        self,
        history: dict[str, list],
        update_info: dict,
        tempering_parameter: float,
        iteration: int,
        start_time: float,
    ) -> None:
        """Append one mutation's numbers to the mutation history.

        Also logs one INFO line for the iteration; ``tempering_parameter``,
        ``iteration`` and ``start_time`` are only used there.
        """
        summary = self._mutation.summary(update_info)
        history["acceptance_history"].append(summary["acceptance"])
        history["n_steps_history"].append(summary["n_steps"])
        history["scale_history"].append(summary["scale"])
        if summary["training_loss"] is not None:
            history["training_loss_history"].append(summary["training_loss"])

        parts = [
            f"iteration {iteration}",
            f"lambda {tempering_parameter:.3e}",
            f"{summary['n_steps']} steps ({summary['stop_reason']})",
            f"acceptance {summary['acceptance']:.2f}",
            f"{summary['scale_name']} {summary['scale']:.3g}",
        ]
        if not np.isnan(summary["nu"]):
            parts.append(f"nu {summary['nu']:.3g}")
        parts.append(f"{summary['n_unique']} unique particles")
        if summary["training_loss"] is not None:
            parts.append(f"training loss {np.min(summary['training_loss']):.3f}")
        parts.append(f"{time.perf_counter() - start_time:.1f} s")
        logger.info("%s: %s", self._log_label, ", ".join(parts))

    def _mutation_history_diagnostics(self) -> dict[str, np.ndarray]:
        """The mutation history as diagnostics arrays."""
        history = self._mutation_history
        result = {name: np.asarray(values) for name, values in history.items()}
        result["n_steps_history"] = result["n_steps_history"].astype(int)
        return result

    def _n_likelihood_evaluations(
        self, *, n_initial_passes: int, n_extra_passes_per_iteration: int
    ) -> int:
        """Number of likelihood evaluations in the run.

        A pass evaluates the likelihood once for every particle.  Each MCMC
        step of a mutation is one pass.  On top of those, a mode makes
        ``n_initial_passes`` passes before the first iteration and
        ``n_extra_passes_per_iteration`` passes in every iteration.  The total
        is ``n_particles * (n_initial_passes + sum over iterations of
        (n_steps + n_extra_passes_per_iteration))``.
        """
        n_steps = np.asarray(self._mutation_history["n_steps_history"], dtype=int)
        n_passes = n_initial_passes + np.sum(n_steps + n_extra_passes_per_iteration)
        return int(self._config.n_particles * n_passes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _sample(
        self,
        rng_key: Key[Array, ""],
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
        self._run(rng_key, arr)

    @abstractmethod
    def _run(
        self,
        rng_key: Key[Array, ""],
        initial_particles: Float[Array, "n_particles n_dims"],
    ) -> None:
        """Run this mode's SMC loop; sets ``_final_state``/``_n_iterations``/``_mutation_history``."""
