"""Ripple waveform models, exposed under jim's historical ``Ripple*`` names.

ripplegw's only construction path is the name-based registry
``ripplegw.waveform(name, **config)`` -- concrete classes (``IMRPhenomD``,
``TaylorF2``, ...) are intentionally not exposed at its top level. These are
thin ``functools.partial`` bindings onto that registry, so the rest of jim can
keep constructing models as ``RippleIMRPhenomD(f_ref=20.0)``.
"""

from functools import partial

import ripplegw

RippleTaylorF2 = partial(ripplegw.waveform, "TaylorF2")
RippleIMRPhenomD = partial(ripplegw.waveform, "IMRPhenomD")
RippleIMRPhenomD_NRTidalv2 = partial(ripplegw.waveform, "IMRPhenomD_NRTidalv2")
RippleIMRPhenomHM = partial(ripplegw.waveform, "IMRPhenomHM")
RippleIMRPhenomPv2 = partial(ripplegw.waveform, "IMRPhenomPv2")
RippleIMRPhenomXAS = partial(ripplegw.waveform, "IMRPhenomXAS")
RippleIMRPhenomXAS_NRTidalv3 = partial(ripplegw.waveform, "IMRPhenomXAS_NRTidalv3")
RippleIMRPhenomXHM = partial(ripplegw.waveform, "IMRPhenomXHM")
RippleIMRPhenomXP = partial(ripplegw.waveform, "IMRPhenomXP")
RippleIMRPhenomXPHM = partial(ripplegw.waveform, "IMRPhenomXPHM")
RippleSineGaussian = partial(ripplegw.waveform, "SineGaussian")

__all__ = [
    "RippleIMRPhenomD",
    "RippleIMRPhenomD_NRTidalv2",
    "RippleIMRPhenomHM",
    "RippleIMRPhenomPv2",
    "RippleIMRPhenomXAS",
    "RippleIMRPhenomXAS_NRTidalv3",
    "RippleIMRPhenomXHM",
    "RippleIMRPhenomXP",
    "RippleIMRPhenomXPHM",
    "RippleSineGaussian",
    "RippleTaylorF2",
]
