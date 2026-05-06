"""BlackJAX nested sampling with bilby/dynesty-style adaptive DE acceptance-walk kernel."""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key
from anesthetic.samples import NestedSamples
import blackjax

from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax._acceptance_walk_kernel import bilby_adaptive_de_sampler
from jimgw.samplers.blackjax._imports import require_nested_sampling
from jimgw.samplers.config import BlackJAXNSAWConfig
from jimgw.samplers.periodic import to_unit_cube_stepper

require_nested_sampling(blackjax)


class BlackJAXNSAWSampler(Sampler):
    """BlackJAX nested sampler using the bilby/dynesty-style adaptive DE acceptance-walk kernel.

    Samples in the sampling space defined by ``sample_transforms`` (typically
    the unit cube via ``BoundToBound`` transforms).  Operates on flat arrays
    of shape ``(n_dims,)``; the acceptance-walk kernel is pytree-generic and
    works identically with flat arrays.

    !!! note
        This sampler requires the sampling space to be the unit hypercube
        ``[0, 1]^n_dims``.  All ``sample_transforms`` in Jim must map the
        prior support onto ``[0, 1]`` per dimension before sampling.  A
        `ValueError` is raised at construction if the supplied
        ``log_prior_fn`` violates this constraint.

    **Reference:** Prathaban, M., Yallup, D., Alvey, J., Yang, M., Templeton, W.,
    Handley, W., *"Gravitational-wave inference at GPU speed: A bilby-like nested
    sampling kernel within blackjax-ns"*, arXiv:2509.04336 (Sep 2025).

    Configure via [`BlackJAXNSAWConfig`][jimgw.samplers.config.BlackJAXNSAWConfig].

    Args:
        n_dims: Dimension of the sampling space.
        log_prior_fn: Log-prior callable ``(arr,) -> float``.
        log_likelihood_fn: Log-likelihood callable ``(arr,) -> float``.
        log_posterior_fn: Log-posterior callable ``(arr,) -> float``.
        config: Optional :class:`BlackJAXNSAWConfig`; defaults to all-default values.
        periodic: Optional list of dimension indices that are periodic in
            ``[0, 1]`` (unit-cube space).  ``None`` means no periodic
            parameters.  Provided by Jim after resolving parameter names.
    """

    _config: BlackJAXNSAWConfig
    _stepper_fn: Callable
    _final_state: Any
    _nested_samples: NestedSamples
    _n_iterations: int

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: Optional[BlackJAXNSAWConfig] = None,
        periodic: Optional[list[int]] = None,
    ) -> None:
        if config is None:
            config = BlackJAXNSAWConfig()
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )
        self._stepper_fn = to_unit_cube_stepper(periodic, n_dims)
        self._validate_unit_cube_prior(log_prior_fn)

    def _validate_unit_cube_prior(self, log_prior_fn: Callable) -> None:
        """Raise ValueError if log_prior_fn is not the normalized uniform on [0, 1]^n_dims."""
        n = self.n_dims
        # Use interior points only (avoid 0.0 and 1.0 boundaries)
        diag_vs = jnp.linspace(0.0 + 1e-3, 1.0 - 1e-3, 5)
        diag_pts = jnp.stack([jnp.full(n, v) for v in diag_vs])
        random_pts = jax.random.uniform(jax.random.key(123), (5, n))
        # Clip random_pts to avoid exact 0.0 and 1.0 boundaries
        random_pts = jnp.clip(random_pts, 0.0 + 1e-3, 1.0 - 1e-3)
        in_support = jnp.concatenate([diag_pts, random_pts], axis=0)
        out_support = jnp.stack([jnp.full(n, -1e-3), jnp.full(n, 1.0 + 1e-3)])

        log_prior_vmap = jax.vmap(log_prior_fn)
        lp_in = log_prior_vmap(in_support)
        lp_out = log_prior_vmap(out_support)

        if not jnp.allclose(lp_in, jnp.zeros_like(lp_in)):
            raise ValueError(
                "log_prior_fn must return 0.0 for all points in [0, 1]^n_dims. "
            )
        if not jnp.all(jnp.isnan(lp_out) | jnp.isneginf(lp_out)):
            raise ValueError(
                "log_prior_fn must return -inf for all points outside [0, 1]^n_dims. "
            )

    def _sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_live n_dims"],
    ) -> None:
        """Run the BlackJAX NS-AW sampler.

        Args:
            rng_key: JAX PRNG key.
            initial_position: Starting live points in the unit-cube sampling
                space, shape ``(n_live, n_dims)``.  Must match ``config.n_live``.

        Raises:
            ValueError: If ``initial_position`` shape does not match
                ``(n_live, n_dims)``.
        """
        config = self._config
        n_live = config.n_live
        n_delete = int(n_live * config.n_delete_frac)

        arr = jnp.asarray(initial_position)
        if arr.ndim != 2 or arr.shape != (n_live, self.n_dims):
            raise ValueError(
                f"initial_position must have shape ({n_live}, {self.n_dims}), "
                f"got {arr.shape}."
            )
        initial_particles = arr

        nested_sampler = bilby_adaptive_de_sampler(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            nlive=n_live,
            n_target=config.n_target,
            max_mcmc=config.max_mcmc,
            num_delete=n_delete,
            stepper_fn=self._stepper_fn,
            max_proposals=config.max_proposals,
        )

        state = nested_sampler.init(initial_particles)  # type: ignore[call-arg]  # blackjax fork API

        def _terminate(state: Any) -> bool:
            dlogz = jnp.logaddexp(0, state.integrator.logZ_live - state.integrator.logZ)
            return bool(jnp.isfinite(dlogz) and dlogz < config.termination_dlogz)

        step_fn = jax.jit(nested_sampler.step)

        dead = []
        n_iter = 0
        while not _terminate(state):
            rng_key, subkey = jax.random.split(rng_key)
            state, dead_info = step_fn(subkey, state)
            dead.append(dead_info)
            n_iter += 1

        from blackjax.ns.utils import finalise

        self._final_state = finalise(state, dead)
        self._n_iterations = n_iter

        # Build anesthetic NestedSamples for use in get_samples() and get_diagnostics().
        particles_sample = np.array(self._final_state.particles.position)
        log_likelihood = np.array(self._final_state.particles.loglikelihood)
        logL_birth = np.array(self._final_state.particles.loglikelihood_birth)
        logL_birth = np.where(np.isnan(logL_birth), -np.inf, logL_birth)
        self._nested_samples = NestedSamples(
            particles_sample,
            logL=log_likelihood,
            logL_birth=logL_birth,
            logzero=np.nan,
            dtype=np.float64,
        )

    def get_samples(self) -> dict[str, np.ndarray]:
        """Return equally-weighted posterior samples.

        Uses `NestedSamples.posterior_points` to
        resample the nested dead-point collection to a set of equal-weight
        samples.

        Returns:
            Dict with keys ``"samples"`` (shape ``(n, n_dims)``) and
            ``"log_likelihood"`` (shape ``(n,)``).
        """
        if not self._sampled:
            raise RuntimeError("get_samples() called before sample()")
        posterior = self._nested_samples.posterior_points()
        samples = np.asarray(posterior.iloc[:, : self.n_dims])
        log_L = np.asarray(posterior["logL"])
        return {"samples": samples, "log_likelihood": log_L}

    def _get_diagnostics(self) -> dict[str, Any]:
        """Return NS-AW run diagnostics.

        Returns a dict with the following keys:

        * ``"n_likelihood_evaluations"`` — total likelihood calls.
        * ``"n_iterations"`` — total nested-sampling iterations.
        * ``"n_accept_history"`` — accepted proposal count per NS iteration.
        * ``"n_walks_history"`` — completed walk count per NS iteration.
        * ``"n_proposals_history"`` — total proposal count per NS iteration.
        * ``"log_Z"`` — log Bayesian evidence (anesthetic mean estimate).
        * ``"log_Z_error"`` — standard deviation of log Z from 100 bootstrap samples.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")
        ui = self._final_state.update_info  # DEWalkInfo concatenated across all steps
        n_evals = int(jnp.sum(ui.n_likelihood_evals))

        log_Z = np.asarray(self._nested_samples.logZ()).item()
        log_Z_error = np.std(np.asarray(self._nested_samples.logZ(nsamples=100))).item()

        return {
            "n_likelihood_evaluations": n_evals,
            "n_iterations": self._n_iterations,
            "n_accept_history": np.asarray(ui.n_accept),
            "n_walks_history": np.asarray(ui.walks_completed),
            "n_proposals_history": np.asarray(ui.total_proposals),
            "log_Z": log_Z,
            "log_Z_error": log_Z_error,
        }
