import logging
from collections.abc import Callable

from ripplegw.interfaces import Waveform

from jimgw.cli._config import WaveformConfig
from jimgw.core.single_event.waveform import (
    RippleIMRPhenomD,
    RippleIMRPhenomD_NRTidalv2,
    RippleIMRPhenomHM,
    RippleIMRPhenomPv2,
    RippleIMRPhenomXAS,
    RippleIMRPhenomXAS_NRTidalv3,
    RippleIMRPhenomXHM,
    RippleIMRPhenomXP,
    RippleIMRPhenomXPHM,
    RippleSineGaussian,
    RippleTaylorF2,
)

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[..., Waveform]] = {
    "TaylorF2": RippleTaylorF2,
    "IMRPhenomD": RippleIMRPhenomD,
    "IMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
    "IMRPhenomHM": RippleIMRPhenomHM,
    "IMRPhenomPv2": RippleIMRPhenomPv2,
    "IMRPhenomXAS": RippleIMRPhenomXAS,
    "IMRPhenomXAS_NRTidalv3": RippleIMRPhenomXAS_NRTidalv3,
    "IMRPhenomXHM": RippleIMRPhenomXHM,
    "IMRPhenomXP": RippleIMRPhenomXP,
    "IMRPhenomXPHM": RippleIMRPhenomXPHM,
}


def build_waveform(cfg: WaveformConfig) -> Waveform:
    """Instantiate the ripple waveform specified by *cfg*."""
    if cfg.approximant == "SineGaussian":
        waveform = RippleSineGaussian()
        logger.info("Built waveform: %s", type(waveform).__name__)
        return waveform

    waveform = _REGISTRY[cfg.approximant](f_ref=cfg.f_ref)
    logger.info("Built waveform: %s(f_ref=%.1f)", type(waveform).__name__, cfg.f_ref)
    return waveform
