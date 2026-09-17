"""Shared machinery for the BlackJAX SMC sampler modes.

``_BlackJAXSMCBase`` holds everything that doesn't depend on which of the four
SMC modes (adaptive/fixed temperature schedule x persistent/tempered
resampling) a concrete sampler implements: construction, checkpoint
validation, the plain (non-preconditioned) mutation kernel, and the
normalizing-flow preconditioning machinery. Each mode's own module
(``adaptive_persistent``, ``fixed_persistent``, ``adaptive_tempered``,
``fixed_tempered``) implements ``_run``/``get_samples``/``_get_diagnostics``.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.mcmc import random_walk
from flowMC.resource.model.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from jaxtyping import Array, Float, Key, PyTree

from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax.smc import precondition
from jimgw.samplers.blackjax.smc.typing import PlainMCMCStep, PreconditionedMCMCStep
from jimgw.samplers.config import BlackJAXSMCConfig
from jimgw.samplers.periodic import to_displacement_wrapper, to_position_wrapper
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
    _position_wrapper: Callable[[Float[Array, " n_dim"]], Float[Array, " n_dim"]]
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
        self._position_wrapper = to_position_wrapper(periodic, n_dims)

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
        checkpoint_n_particles = checkpoint.get("n_particles")
        if (
            checkpoint_n_particles is not None
            and checkpoint_n_particles != self._config.n_particles
        ):
            raise ValueError(
                f"checkpoint was created with n_particles={checkpoint_n_particles}, "
                f"but this sampler has n_particles={self._config.n_particles}; "
                "resuming with a different value is not supported."
            )
        checkpoint_n_dims = checkpoint.get("n_dims")
        if checkpoint_n_dims is not None and checkpoint_n_dims != self.n_dims:
            raise ValueError(
                f"checkpoint was created with n_dims={checkpoint_n_dims}, but "
                f"this sampler has n_dims={self.n_dims}; resuming with a "
                "different sampling-space dimension is not supported."
            )
        if self._config.temperature_ladder is not None:
            checkpoint_ladder = checkpoint.get("temperature_ladder")
            current_ladder = tuple(self._config.temperature_ladder)
            if (
                checkpoint_ladder is not None
                and tuple(checkpoint_ladder) != current_ladder
            ):
                raise ValueError(
                    f"checkpoint was created with temperature_ladder="
                    f"{checkpoint_ladder}, but this sampler has "
                    f"temperature_ladder={current_ladder}; resuming with a "
                    "different ladder is not supported."
                )
        checkpoint_precondition = checkpoint.get("precondition", False)
        current_precondition = self._config.precondition is not None
        if checkpoint_precondition != current_precondition:
            raise ValueError(
                f"checkpoint was created with precondition={checkpoint_precondition}, "
                f"but this sampler has precondition={current_precondition}; resuming "
                "with a different value is not supported."
            )
        if self._config.precondition:
            checkpoint_architecture = checkpoint.get("precondition_architecture")
            current_architecture = (
                self._config.precondition.rq_spline_n_layers,
                tuple(self._config.precondition.rq_spline_hidden_units),
                self._config.precondition.rq_spline_n_bins,
            )
            if (
                checkpoint_architecture is not None
                and checkpoint_architecture != current_architecture
            ):
                raise ValueError(
                    "checkpoint was created with a different preconditioning flow "
                    f"architecture {checkpoint_architecture} than the current config "
                    f"{current_architecture} (n_layers, hidden_units, n_bins); restore "
                    "the original precondition.rq_spline_* values or start a fresh run."
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

    def _init_precondition(
        self, key: Key
    ) -> tuple[
        MaskedCouplingRQSpline,
        Optimizer,
        PreconditionedMCMCStep,
        Float[Array, " n_flat"],
        Callable[[Float[Array, " n_flat"]], PyTree],
        MaskedCouplingRQSpline,
    ]:
        """Build a fresh flow + optimizer for preconditioning and its mutation kernel.

        Returns ``(flow, optimizer, mcmc_step, flat_params, unravel_fn, static)``
        -- see ``precondition.build_preconditioned_mcmc_step`` for why
        ``mcmc_step`` is safe to reuse across iterations. ``unravel_fn``/
        ``static`` are also returned here so a resumed run can rebuild ``flow``
        from a checkpointed ``flow_params`` array, via
        ``_reconstruct_precondition_flow``.
        """
        precondition_config = self._config.precondition
        assert precondition_config is not None, "precondition config must be set"
        flow = precondition.build_flow(self.n_dims, precondition_config, key)
        optimizer = precondition.build_optimizer(flow, precondition_config)
        flat_params, unravel_fn, static = precondition.partition_flow(flow)
        mcmc_step = precondition.build_preconditioned_mcmc_step(
            unravel_fn, static, self._position_wrapper
        )
        return flow, optimizer, mcmc_step, flat_params, unravel_fn, static

    def _retrain_precondition_flow(self, rng_key, flow, optimizer, training_particles):
        """Retrain the preconditioning flow (outside JIT) and re-flatten its weights."""
        precondition_config = self._config.precondition
        assert precondition_config is not None, "precondition config must be set"
        rng_key, flow, optimizer, _ = precondition.train_flow(
            rng_key, flow, optimizer, training_particles, precondition_config
        )
        flat_params = precondition.flatten_flow(flow)
        return rng_key, flow, optimizer, flat_params

    def _latent_covariance(self, flow, particles, weights=None):
        """Empirical covariance of ``particles`` mapped into the flow's latent space.

        Used as the preconditioned kernel's proposal covariance: the flow already
        whitens sampling-space particles internally, so a sampling-space covariance
        would double-scale (or badly mis-scale) the latent-space step -- pocoMC's
        own preconditioned kernel fits its proposal covariance the same way, on
        flow-transformed particles, not raw sampling-space ones.

        ``weights`` fits a weights-aware covariance (``jnp.cov``'s ``aweights``),
        matching pocoMC's own ``Geometry.fit``: ``np.cov(theta.T,
        aweights=weights)``, which its preconditioned-RWM kernel reads directly
        as ``geometry.normal_cov``. Pass the tempered modes' (at/ft) actual
        particle weights here; leave ``None`` for the persistent modes (ap/fp),
        whose particles are already equal-weight. Guarded against effective
        sample size collapse -- see ``precondition.weighted_covariance``.
        """
        latent_particles = jax.vmap(lambda x: precondition.to_latent(flow, x)[0])(
            particles
        )
        return precondition.weighted_covariance(latent_particles, weights)

    def _reconstruct_precondition_flow(self, state, unravel_fn, static):
        """Rebuild the flow from ``state``'s flow_params (via ``precondition.combine_flow``).

        A no-op on a fresh run; recovers the checkpointed weights on a resumed
        one. Pair with ``_restore_precondition_optimizer_state``, which restores
        the optimizer state this doesn't carry.
        """
        flat_params = state.parameter_override["flow_params"][0]
        flow = precondition.combine_flow(flat_params, unravel_fn, static)
        return flow, flat_params

    def _restore_precondition_optimizer_state(
        self, precond, mode_checkpoint_data, n_completed_iterations: int
    ) -> None:
        """Restore the optimizer's Adam state from a checkpoint, if resumed.

        Without this, a resumed run's first retrain would warm-start from
        momentum accumulated while training on *this* invocation's
        ``initial_particles`` during ``_setup_precondition_for_run`` -- momentum
        unrelated to the actual checkpointed flow weights, and dependent on an
        argument ``_sample`` documents as ignored on resume. pocoMC's own
        checkpoint/resume persists its whole flow object (including whatever
        optimizer state it carries) verbatim (``save_state``/``load_state``
        pickle ``self.__dict__`` wholesale); this mirrors that for the one piece
        of state BlackJAX's shared-parameter channel doesn't carry.

        A genuine resume (``n_completed_iterations > 0``) whose checkpoint
        predates this field resets the optimizer fresh from the restored flow
        instead, rather than silently keeping the momentum above; see
        ``precondition.md``. A fresh (non-resumed) run's warm-start momentum
        is untouched.
        """
        if precond is None:
            return
        checkpointed_state = mode_checkpoint_data.get("precondition_optimizer_state")
        if checkpointed_state is not None:
            precond.optimizer.optim_state = checkpointed_state
        elif n_completed_iterations > 0:
            precond.optimizer.optim_state = precond.optimizer.optim.init(
                eqx.filter(precond.flow, eqx.is_inexact_array)
            )

    def _setup_precondition_for_run(
        self, rng_key: Key, initial_particles
    ) -> tuple[Key, Optional[precondition.PreconditionState]]:
        """Build and warm-train this run's flow, or ``None`` if precondition is off.

        Trains once on ``initial_particles`` so the first mutation step already
        uses a fitted flow (matches pocoMC's reweight->train->resample->mutate
        ordering).
        """
        if self._config.precondition is None:
            return rng_key, None
        rng_key, flow_key = jax.random.split(rng_key)
        flow, optimizer, mcmc_step, flat_params, unravel_fn, static = (
            self._init_precondition(flow_key)
        )
        rng_key, flow, optimizer, flat_params = self._retrain_precondition_flow(
            rng_key, flow, optimizer, initial_particles
        )
        precond = precondition.PreconditionState(
            flow=flow,
            optimizer=optimizer,
            flat_params=flat_params,
            unravel_fn=unravel_fn,
            static=static,
            mcmc_step=mcmc_step,
        )
        return rng_key, precond

    def _precondition_iteration_update(
        self,
        rng_key: Key,
        precond: precondition.PreconditionState,
        sampler_state,
        *,
        is_terminal_iteration: bool,
        n_completed_iterations: int,
        needs_resampling: bool,
        covariance_scale: float = 1.0,
    ) -> tuple[Key, dict[str, Array]]:
        """Retrain (unless terminal) and refit the latent-space proposal covariance.

        Mutates ``precond.flow``/``precond.optimizer``/``precond.flat_params`` in
        place when a retrain runs; always returns a fresh ``cov``/``flow_params``
        pair to install as the shared SMC parameter. ``needs_resampling`` draws an
        equal-weight training set first for the weighted-particle tempered modes
        (at/ft), since flowMC's flow.fit has no weighted-loss option (pocoMC's
        own flow.fit trains on the raw weighted population directly); the
        persistent modes (ap/fp) already hand mutation equal-weight particles.
        The covariance, unlike training, is fit on ``sampler_state``'s own raw
        weighted particles/weights (not the resampled training set) -- matching
        pocoMC's ``theta_geometry.fit(flow.forward(u), weights=w)``, using the
        same ``u``/``w`` as ``flow.fit``, not a separately resampled population.
        No further mutation uses a terminal iteration's retrain, so it's skipped
        there as wasted work; the covariance is still refit, since it's only a
        cheap forward pass, not a training loop.

        ``n_completed_iterations`` is the count *before* this iteration, so the
        cadence check uses ``+ 1`` to land on iterations ``train_frequency,
        2 * train_frequency, ...`` -- matching pocoMC's own cadence, whose ``t``
        is incremented at the top of ``_reweight``, before its equivalent check.
        """
        config = self._config
        assert config.precondition is not None
        weights = sampler_state.weights if needs_resampling else None
        if (
            not is_terminal_iteration
            and (n_completed_iterations + 1) % config.precondition.train_frequency == 0
        ):
            if needs_resampling:
                rng_key, resample_key = jax.random.split(rng_key)
                training_particles = precondition.training_particles_from_weighted(
                    resample_key,
                    sampler_state.particles,
                    sampler_state.weights,
                    sampler_state.particles.shape[0],
                )
            else:
                training_particles = sampler_state.particles
            rng_key, precond.flow, precond.optimizer, precond.flat_params = (
                self._retrain_precondition_flow(
                    rng_key, precond.flow, precond.optimizer, training_particles
                )
            )
        new_parameters: dict[str, Array] = {
            "cov": self._latent_covariance(
                precond.flow, sampler_state.particles, weights=weights
            )
            * covariance_scale,
            "flow_params": precond.flat_params,
        }
        return rng_key, new_parameters

    def _checkpoint_extra(self, **mode_specific: object) -> dict[str, object]:
        """Checkpoint payload's mode/precondition metadata, plus mode-specific extras."""
        config = self._config
        architecture = None
        if config.precondition is not None:
            architecture = (
                config.precondition.rq_spline_n_layers,
                tuple(config.precondition.rq_spline_hidden_units),
                config.precondition.rq_spline_n_bins,
            )
        return {
            "mode": self.mode,
            "n_particles": config.n_particles,
            "n_dims": self.n_dims,
            "temperature_ladder": (
                tuple(config.temperature_ladder)
                if config.temperature_ladder is not None
                else None
            ),
            "precondition": config.precondition is not None,
            "precondition_architecture": architecture,
            **mode_specific,
        }

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
