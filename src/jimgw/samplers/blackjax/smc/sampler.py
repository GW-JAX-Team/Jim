"""Factory selecting the right BlackJAX SMC mode class from a ``BlackJAXSMCConfig``."""

from collections.abc import Callable
from typing import Optional

from jaxtyping import Array, Float

from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax.smc.adaptive_persistent import AdaptivePersistentSMCSampler
from jimgw.samplers.blackjax.smc.adaptive_tempered import AdaptiveTemperedSMCSampler
from jimgw.samplers.blackjax.smc.fixed_persistent import FixedPersistentSMCSampler
from jimgw.samplers.blackjax.smc.fixed_tempered import FixedTemperedSMCSampler
from jimgw.samplers.config import BlackJAXSMCConfig
from jimgw.typing import FloatScalar


def build_blackjax_smc_sampler(
    *,
    n_dims: int,
    log_prior_fn: Callable[[Float[Array, " n_dims"]], FloatScalar],
    log_likelihood_fn: Callable[[Float[Array, " n_dims"]], FloatScalar],
    log_posterior_fn: Callable[[Float[Array, " n_dims"]], FloatScalar],
    config: Optional[BlackJAXSMCConfig] = None,
    periodic: Optional[dict[int, tuple[float, float]]] = None,
) -> Sampler:
    """Construct the BlackJAX SMC mode sampler selected by ``config``.

    ``persistent_sampling``/``temperature_ladder`` select one of the four mode
    classes, matching the same combinations ``BlackJAXSMCConfig`` has always
    supported:

    * ``persistent_sampling=True,  temperature_ladder=None``  -> adaptive persistent (ap)
    * ``persistent_sampling=True,  temperature_ladder=given`` -> fixed-ladder persistent (fp)
    * ``persistent_sampling=False, temperature_ladder=None``  -> adaptive tempered (at)
    * ``persistent_sampling=False, temperature_ladder=given`` -> fixed-ladder tempered (ft)
    """
    if config is None:
        config = BlackJAXSMCConfig()
    if config.persistent_sampling:
        mode_cls = (
            FixedPersistentSMCSampler
            if config.temperature_ladder is not None
            else AdaptivePersistentSMCSampler
        )
    else:
        mode_cls = (
            FixedTemperedSMCSampler
            if config.temperature_ladder is not None
            else AdaptiveTemperedSMCSampler
        )
    return mode_cls(
        n_dims=n_dims,
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
        periodic=periodic,
    )
