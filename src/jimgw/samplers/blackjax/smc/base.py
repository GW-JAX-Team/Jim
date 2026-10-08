"""Shared machinery for the BlackJAX SMC sampler modes.

``_BlackJAXSMCBase`` holds everything that doesn't depend on which of the four
SMC modes (adaptive/fixed temperature schedule x persistent/tempered
resampling) a concrete sampler implements: construction, checkpoint
validation, and the mutation kernel. Each mode's own module
(``adaptive_persistent``, ``fixed_persistent``, ``adaptive_tempered``,
``fixed_tempered``) implements ``_run``/``get_samples``/``_get_diagnostics``.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.mcmc import random_walk
from jaxtyping import Array, Float, Key

from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax.smc.typing import PlainMCMCStep
from jimgw.samplers.config import BlackJAXSMCConfig
from jimgw.samplers.periodic import to_displacement_wrapper
from jimgw.typing import FloatScalar


class _BlackJAXSMCBase(Sampler):
    """Shared base for the four BlackJAX SMC mode samplers.

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

    mode: ClassVar[str]
    _config: BlackJAXSMCConfig
    _displacement_wrapper: Callable[
        [Float[Array, " n_dim"], Float[Array, " n_dim"]], Float[Array, " n_dim"]
    ]
    # Any: one of several heterogeneous, duck-typed blackjax SMC state types; object breaks attribute access, since blackjax's own stubs don't expose them precisely either.
    _final_state: Any
    _n_iterations: int
    _acceptance_history: np.ndarray  # per-step mean acceptance rate, all modes

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
        self._displacement_wrapper = to_displacement_wrapper(periodic, n_dims)

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mcmc_step(self) -> PlainMCMCStep:
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

    def _checkpoint_extra(self, **mode_specific: object) -> dict[str, object]:
        """Checkpoint payload's mode metadata, plus mode-specific extras."""
        return {"mode": self.mode, **mode_specific}

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
        self._run(rng_key, arr)

    @abstractmethod
    def _run(
        self, rng_key: Key, initial_particles: Float[Array, "n_particles n_dims"]
    ) -> None:
        """Run this mode's SMC loop; sets ``_final_state``/``_n_iterations``/diagnostics stashes."""
