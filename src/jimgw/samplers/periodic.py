"""Adapters that translate Jim's periodic-parameter spec into the form each
sampler backend expects.

    periodic_index = {1: (0.0, 2 * jnp.pi), ...}   # key = dimension index

Each backend wants a different shape: flowMC already accepts an index-keyed dict
directly; BlackJAX NS AW needs a stepper function on flat arrays; BlackJAX NSS
needs a ``proposal`` factory with the same interface as
``blackjax.ns.nss.covariance_proposal``; BlackJAX SMC needs a re-centring map
onto the real line.  The adapters below handle those conversions.

All adapters operate on flat JAX arrays of shape ``(n_dims,)``.
"""

from collections.abc import Callable
from numbers import Integral
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logit

# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _build_masks_arrays(
    periodic_index: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build boolean mask, lower-bound, and period arrays from an index-keyed dict.

    Returns ``(mask, lower, period)`` as JAX arrays of shape ``(n_dims,)``.
    """
    periodic_index = periodic_index or {}

    mask = jnp.zeros(n_dims, dtype=bool)
    lower = jnp.zeros(n_dims)
    period = jnp.ones(n_dims)

    for i, (lo, hi) in periodic_index.items():
        if not isinstance(i, Integral):
            raise TypeError(
                f"periodic_index keys must be integers, got {type(i).__name__!r} for key {i!r}."
            )
        if i < 0 or i >= n_dims:
            raise ValueError(
                f"periodic_index key {i} is out of bounds for n_dims={n_dims}."
            )
        lo_f, hi_f = float(lo), float(hi)
        if not jnp.isfinite(lo_f) or not jnp.isfinite(hi_f):
            raise ValueError(
                f"Periodic bounds for dimension {i} must be finite, got ({lo_f}, {hi_f})."
            )
        if hi_f <= lo_f:
            raise ValueError(
                f"Periodic bounds for dimension {i} must satisfy hi > lo, got ({lo_f}, {hi_f})."
            )
        period_f = hi_f - lo_f
        if not jnp.isfinite(period_f):
            raise ValueError(f"Periodic interval for dimension {i} must be finite.")
        mask = mask.at[i].set(True)
        lower = lower.at[i].set(lo_f)
        period = period.at[i].set(period_f)

    return mask, lower, period


# ----------------------------------------------------------------------
# BlackJAX NS AW
# ----------------------------------------------------------------------


def to_unit_cube_stepper(
    periodic_index: Optional[list[int]],
    n_dims: int,
) -> Callable:
    """Stepper function for BlackJAX NS AW (unit-cube space).

    Signature: ``stepper_fn(position, direction, step_size) -> new_position``

    ``periodic_index`` is a list of dimension indices to wrap; bounds are implicit
    because NS AW always operates in ``[0, 1]^n_dims``, so wrapping is always
    ``mod(pos + step_size * dir, 1.0)``.
    """
    mask = jnp.zeros(n_dims, dtype=bool)
    for i in periodic_index or []:
        if not isinstance(i, Integral):
            raise TypeError(
                f"periodic_index entries must be integers, got {type(i).__name__!r} for {i!r}."
            )
        if i < 0 or i >= n_dims:
            raise ValueError(
                f"periodic_index entry {i} is out of bounds for n_dims={n_dims}."
            )
        mask = mask.at[i].set(True)

    def stepper(
        position: jnp.ndarray, direction: jnp.ndarray, step_size: float
    ) -> jnp.ndarray:
        proposed = position + step_size * direction
        return jnp.where(mask, jnp.mod(proposed, 1.0), proposed)

    return stepper


# ----------------------------------------------------------------------
# BlackJAX NSS
# ----------------------------------------------------------------------


def to_prior_space_proposal(
    periodic_index: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
    sample_direction_from_covariance: Callable,
) -> Callable:
    """Return a BlackJAX NSS proposal factory that wraps periodic prior-space dims.

    Has the same interface as ``blackjax.ns.nss.covariance_proposal``:
    ``proposal(init_state_fn, loglikelihood_0, cov) -> proposal_generator``.
    """
    mask, lower, period = _build_masks_arrays(periodic_index, n_dims)

    def proposal(init_state_fn, loglikelihood_0, cov):
        def proposal_generator(rng_key, position, logdensity_fn):
            del logdensity_fn  # NS gates on the recorded loglikelihood.
            direction = sample_direction_from_covariance(rng_key, position, cov)

            def slice_fn(t):
                proposed = position + t * direction
                x = jnp.where(mask, lower + jnp.mod(proposed - lower, period), proposed)
                new_state = init_state_fn(x, loglikelihood_birth=loglikelihood_0)
                return new_state, new_state.loglikelihood > loglikelihood_0

            return slice_fn

        return proposal_generator

    return proposal


# ----------------------------------------------------------------------
# BlackJAX SMC
# ----------------------------------------------------------------------


def to_recentring_map(
    periodic_index: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
) -> tuple[Callable, Callable, Callable]:
    """Re-centring map for BlackJAX SMC (sampling space to the real line).

    For each periodic dimension, the cut goes in the middle of the widest arc
    that holds no particle.  The value is rotated so that the cut sits at the
    ends of the period, then mapped to the real line with a logit.  The map is
    a bijection for fixed cuts, so an MCMC move in ``u`` is exact once the
    log-Jacobian is included.  Other dimensions pass through unchanged.

    Returns:
        ``(cuts_fn, forward_fn, inverse_fn)`` with

        * ``cuts_fn(particles) -> cuts``, particles of shape ``(n, n_dims)``
          and cuts of shape ``(n_dims,)`` (0 for non-periodic dimensions);
        * ``forward_fn(x, cuts) -> u``;
        * ``inverse_fn(u, cuts) -> (x, log_det)``, with ``log_det = log|dx/du|``.
    """
    mask, lower, period = _build_masks_arrays(periodic_index, n_dims)

    def cuts_fn(particles: jnp.ndarray) -> jnp.ndarray:
        positions = jnp.sort(lower + jnp.mod(particles - lower, period), axis=0)
        gaps = jnp.concatenate(
            [jnp.diff(positions, axis=0), positions[:1] + period - positions[-1:]]
        )
        widest = jnp.argmax(gaps, axis=0)[None]
        start = jnp.take_along_axis(positions, widest, axis=0)[0]
        width = jnp.take_along_axis(gaps, widest, axis=0)[0]
        return jnp.where(
            mask, lower + jnp.mod(start + 0.5 * width - lower, period), 0.0
        )

    def forward_fn(x: jnp.ndarray, cuts: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(mask, logit(jnp.mod(x - cuts, period) / period), x)

    def inverse_fn(
        u: jnp.ndarray, cuts: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = lower + jnp.mod(cuts - lower + period * jax.nn.sigmoid(u), period)
        log_det = jnp.log(period) + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u)
        return jnp.where(mask, x, u), jnp.sum(jnp.where(mask, log_det, 0.0))

    return cuts_fn, forward_fn, inverse_fn
