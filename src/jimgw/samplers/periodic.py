"""Adapters that translate Jim's periodic-parameter spec into the form each
sampler backend expects.

    periodic_index = {1: (0.0, 2 * math.pi), ...}   # key = dimension index

Each backend wants a different shape: flowMC already accepts an index-keyed dict
directly; BlackJAX NS-AW needs a stepper function on flat arrays; BlackJAX NSS
needs a stepper returning a ``(position, accepted)`` tuple; BlackJAX SMC needs a
displacement wrapper.  The adapters below handle those conversions.

All adapters operate on flat JAX arrays of shape ``(n_dims,)``.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp


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
        if not isinstance(i, int):
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
        mask = mask.at[i].set(True)
        lower = lower.at[i].set(lo_f)
        period = period.at[i].set(hi_f - lo_f)

    return mask, lower, period


def to_unit_cube_stepper(
    periodic_index: Optional[list[int]],
    n_dims: int,
) -> Callable:
    """Stepper function for BlackJAX NS-AW (unit-cube space).

    Signature: ``stepper_fn(position, direction, step_size) -> new_position``

    ``periodic_index`` is a list of dimension indices to wrap; bounds are implicit
    because NS-AW always operates in ``[0, 1]^n_dims``, so wrapping is always
    ``mod(pos + step_size * dir, 1.0)``.
    """
    mask = jnp.zeros(n_dims, dtype=bool)
    for i in periodic_index or []:
        if not isinstance(i, int):
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


def to_prior_space_stepper(
    periodic_index: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
) -> Callable:
    """Stepper function for BlackJAX NSS (prior space).

    Signature: ``stepper_fn(position, direction, step_size) -> (new_position, accepted)``

    Position and direction are flat JAX arrays of shape ``(n_dims,)``.
    NSS requires the stepper to return a ``(position, bool)`` tuple.
    Periodic parameters are wrapped with
    ``lower + mod(pos + step_size * dir - lower, period)``.
    """
    mask, lower, period = _build_masks_arrays(periodic_index, n_dims)

    def stepper(
        position: jnp.ndarray, direction: jnp.ndarray, step_size: float
    ) -> tuple:
        proposed = position + step_size * direction
        wrapped = jnp.where(mask, lower + jnp.mod(proposed - lower, period), proposed)
        return wrapped, True

    return stepper


def to_displacement_wrapper(
    periodic_index: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
) -> Callable:
    """Displacement wrapper for BlackJAX SMC (prior space).

    Signature: ``wrapper_fn(proposed_displacement, current_position) -> wrapped_displacement``

    Displacement and position are flat JAX arrays of shape ``(n_dims,)``.
    SMC's inner kernel operates on displacements. For periodic parameters the
    displacement is adjusted so that ``current + wrapped_displacement`` stays
    within ``[lower, upper)``:

        wrapped_displacement = lower + mod(current + disp - lower, period) - current
    """
    mask, lower, period = _build_masks_arrays(periodic_index, n_dims)

    def wrapper(
        proposed_displacement: jnp.ndarray, current_position: jnp.ndarray
    ) -> jnp.ndarray:
        new_pos = current_position + proposed_displacement
        wrapped_pos = jnp.where(mask, lower + jnp.mod(new_pos - lower, period), new_pos)
        return wrapped_pos - current_position

    return wrapper
