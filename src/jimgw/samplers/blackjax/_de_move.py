"""Shared proposal primitives for differential-evolution samplers.

The nested-sampling acceptance-walk and SMC inner kernels use the same
proposal, but apply different acceptance rules.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Key

from jimgw.typing import FloatScalar, IntScalar


def sample_two_distinct_indices(
    key: Key, population_size: int
) -> tuple[IntScalar, IntScalar]:
    """Sample two distinct indices uniformly from a population."""
    key_a, key_b = jax.random.split(key)
    idx_a = jax.random.randint(key_a, (), 0, population_size)
    idx_b_raw = jax.random.randint(key_b, (), 0, population_size - 1)
    idx_b = jnp.where(idx_b_raw >= idx_a, idx_b_raw + 1, idx_b_raw)
    return idx_a, idx_b


def sample_de_gamma(
    key: Key,
    small_step_probability: float,
    small_step_scale: FloatScalar,
) -> FloatScalar:
    """Sample a small gamma-distributed multiplier or a unit DE multiplier.

    A small step is used with ``small_step_probability``; otherwise the unit
    multiplier makes a mode-hopping proposal.
    """
    key_mix, key_gamma = jax.random.split(key)
    is_small_step = jax.random.uniform(key_mix) < small_step_probability
    return jnp.where(
        is_small_step,
        small_step_scale * jax.random.gamma(key_gamma, 4.0) * 0.25,
        1.0,
    )


def de_proposal_scale(n_dimensions: int) -> FloatScalar:
    """Return the ter Braak (2006) DE-MC scale for ``n_dimensions``."""
    return 2.38 / jnp.sqrt(2.0 * n_dimensions)
