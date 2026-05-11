import logging

from jimgw.cli._config import WaveformConfig
from jimgw.core.single_event.waveform import (
    RippleIMRPhenomD,
    RippleIMRPhenomD_NRTidalv2,
    RippleIMRPhenomPv2,
    RippleIMRPhenomXAS,
    RippleIMRPhenomXAS_NRTidalv3,
    RippleIMRPhenomXHM,
    RippleIMRPhenomXPHM,
    RippleSineGaussian,
    RippleTaylorF2,
)

logger = logging.getLogger(__name__)

_REGISTRY = {
    "IMRPhenomD": RippleIMRPhenomD,
    "IMRPhenomPv2": RippleIMRPhenomPv2,
    "TaylorF2": RippleTaylorF2,
    "IMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
    "IMRPhenomXAS": RippleIMRPhenomXAS,
    "IMRPhenomXAS_NRTidalv3": RippleIMRPhenomXAS_NRTidalv3,
    "IMRPhenomXHM": RippleIMRPhenomXHM,
    "IMRPhenomXPHM": RippleIMRPhenomXPHM,
    "SineGaussian": RippleSineGaussian,
}


def build_waveform(cfg: WaveformConfig):
    """Instantiate the ripple waveform specified by *cfg*."""
    cls = _REGISTRY[cfg.approximant]
    waveform = cls(f_ref=cfg.f_ref)
    logger.info("Built waveform: %s(f_ref=%.1f)", type(waveform).__name__, cfg.f_ref)
    return waveform
