"""Jim's sampler abstraction.

Public API:

* [`Sampler`][jimgw.samplers.base.Sampler] — ABC every backend subclasses.
* `SamplerConfig` — discriminated-union annotation of concrete configs.
* [`build_sampler`][jimgw.samplers.build_sampler] — factory that dispatches to the right concrete class.

The registry uses loaders to defer concrete sampler construction until the
caller requests a backend via `build_sampler`.
"""

from collections.abc import Callable
from typing import Optional

from jimgw.samplers.base import Sampler
from jimgw.samplers.config import (
    BaseSamplerConfig,
    BlackJAXNSAWConfig,
    BlackJAXNSSConfig,
    BlackJAXSMCConfig,
    FlowMCConfig,
    SamplerConfig,
)

__all__ = [
    "Sampler",
    "SamplerConfig",
    "BaseSamplerConfig",
    "FlowMCConfig",
    "BlackJAXNSAWConfig",
    "BlackJAXNSSConfig",
    "BlackJAXSMCConfig",
    "build_sampler",
    "register_sampler",
]


# Each entry is a zero-arg loader returning the concrete sampler's constructor.
SamplerBuilder = Callable[..., Sampler]
_REGISTRY: dict[str, Callable[[], SamplerBuilder]] = {}


def register_sampler(type_str: str, lazy_loader: Callable[[], SamplerBuilder]) -> None:
    """Register a concrete [`Sampler`][jimgw.samplers.base.Sampler] class under ``type_str``.

    ``lazy_loader`` is called (with no args) only when `build_sampler`
    dispatches to this type.
    """
    if type_str in _REGISTRY:
        raise ValueError(
            f"Sampler type {type_str!r} is already registered. "
            "Use a unique type string or remove the existing registration."
        )
    _REGISTRY[type_str] = lazy_loader


def build_sampler(
    config: SamplerConfig,
    *,
    n_dims: int,
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
    log_posterior_fn: Callable,
    periodic: Optional[list[int] | dict[int, tuple[float, float]]] = None,
) -> Sampler:
    """Instantiate the concrete [`Sampler`][jimgw.samplers.base.Sampler] identified by ``config.type``.

    Args:
        config: Typed sampler config; its ``type`` field selects the backend.
        n_dims: Dimension of the sampling space.
        log_prior_fn: Log-prior callable ``(arr,) -> float`` in sampling space.
        log_likelihood_fn: Log-likelihood callable ``(arr,) -> float``.
        log_posterior_fn: Log-posterior callable ``(arr,) -> float``.
        periodic: Periodic-parameter spec already resolved to dimension
            indices by Jim.  For flowMC/NSS/SMC this is a
            ``dict[int, (lo, hi)]``; for NS AW it is a ``list[int]``.

    Raises:
        KeyError: If no sampler is registered for ``config.type``.
    """
    type_str = config.type
    if type_str not in _REGISTRY:
        raise KeyError(
            f"No sampler registered for type {type_str!r}. "
            f"Registered types: {sorted(_REGISTRY)}"
        )
    builder = _REGISTRY[type_str]()
    return builder(
        n_dims=n_dims,
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
        periodic=periodic,
    )


from jimgw.samplers.flowmc import FlowMCSampler  # noqa: E402

register_sampler("flowmc", lambda: FlowMCSampler)

from jimgw.samplers.blackjax.smc import BlackJAXSMCSampler  # noqa: E402

register_sampler("blackjax-smc", lambda: BlackJAXSMCSampler)

from jimgw.samplers.blackjax.ns_aw import BlackJAXNSAWSampler  # noqa: E402

register_sampler("blackjax-ns-aw", lambda: BlackJAXNSAWSampler)

from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler  # noqa: E402

register_sampler("blackjax-nss", lambda: BlackJAXNSSSampler)
