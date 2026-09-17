"""Type aliases for the BlackJAX SMC backend."""

from collections.abc import Callable
from typing import TypeAlias

from blackjax.mcmc import random_walk
from jaxtyping import Array, Float, Key

from jimgw.typing import FloatScalar

# A function mapping one flat sampling-space position to a scalar log-density.
LogDensityFn: TypeAlias = Callable[[Float[Array, " n_dim"]], FloatScalar]

# The plain (non-preconditioned) GRW mutation step: (key, state, logdensity, cov) -> (state, info).
PlainMCMCStep: TypeAlias = Callable[
    [Key, random_walk.RWState, LogDensityFn, Float[Array, "n_dim n_dim"]],
    tuple[random_walk.RWState, random_walk.RWInfo],
]

# A flow-preconditioned RWMH step: (key, state, logdensity, cov, flow_params) -> (state, info).
PreconditionedMCMCStep: TypeAlias = Callable[
    [
        Key,
        random_walk.RWState,
        LogDensityFn,
        Float[Array, "n_dim n_dim"],
        Float[Array, " n_flat"],
    ],
    tuple[random_walk.RWState, random_walk.RWInfo],
]
