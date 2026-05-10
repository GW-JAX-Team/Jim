import logging
from typing import assert_never

import jax.numpy as jnp

from jimgw.cli._config import (
    DataConfig,
    FileDataConfig,
    GWOSCDataConfig,
    InjectionDataConfig,
)
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import GroundBased2G, get_detector_preset
from jimgw.core.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
    SpinAnglesToCartesianSpinTransform,
    SphereSpinToCartesianSpinTransform,
)

logger = logging.getLogger(__name__)


def build_data(
    data_cfg: DataConfig,
    f_min: float,
    f_max: float,
    waveform=None,
) -> list[GroundBased2G]:
    """Construct a list of detectors populated with strain data and PSDs.

    For injection runs, *waveform* is required and is used to inject the signal.
    """
    preset = get_detector_preset()

    if not data_cfg.ifos:
        raise ValueError("data.ifos must be a non-empty list")

    unknown = [ifo for ifo in data_cfg.ifos if ifo not in preset]
    if unknown:
        raise ValueError(
            f"Unknown IFO name(s): {unknown}. Supported: {sorted(preset.keys())}"
        )

    ifos: list[GroundBased2G] = []
    for name in data_cfg.ifos:
        val = preset[name]
        if isinstance(val, list):
            ifos.extend(val)
        else:
            ifos.append(val)

    if isinstance(data_cfg, GWOSCDataConfig):
        _load_gwosc(ifos, data_cfg)
    elif isinstance(data_cfg, InjectionDataConfig):
        _load_injection(ifos, data_cfg, waveform, f_min=f_min, f_max=f_max)
    elif isinstance(data_cfg, FileDataConfig):
        _load_files(ifos, data_cfg)
    else:
        assert_never(data_cfg)

    for ifo in ifos:
        logger.info(
            "%s: %.1f s @ %.0f Hz, PSD shape %s",
            ifo.name,
            ifo.data.duration,
            ifo.data.sampling_frequency,
            ifo.psd.values.shape,
        )

    return ifos


def _load_gwosc(ifos: list[GroundBased2G], cfg: GWOSCDataConfig) -> None:
    # Analysis segment: [trigger - duration + post_trigger, trigger + post_trigger]
    end = cfg.trigger_time + cfg.post_trigger_duration
    start = end - cfg.duration
    # PSD segment: immediately before the analysis segment
    psd_end = start
    psd_start = psd_end - cfg.psd_duration

    for ifo in ifos:
        logger.info("Fetching %s strain [%.1f, %.1f]", ifo.name, start, end)
        strain = Data.from_gwosc(ifo.name, start, end)
        ifo.set_data(strain)

        logger.info("Fetching %s PSD data [%.1f, %.1f]", ifo.name, psd_start, psd_end)
        psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
        nperseg = int(strain.duration * strain.sampling_frequency)
        ifo.set_psd(psd_data.to_psd(nperseg=nperseg))


def to_likelihood_space(
    params: dict[str, float],
    waveform_f_ref: float,
) -> dict[str, float]:
    """Convert injection/reference parameters to likelihood space if needed.

    Handles: q→eta, J-frame spins→Cartesian, spherical spins→Cartesian.

    Raises ``ValueError`` if sampling-space-only parameters (``t_det``,
    ``azimuth``, ``zenith``) are present — use ``t_c``, ``ra``, ``dec``
    instead.
    """
    if "t_det" in params:
        raise ValueError(
            "injection_parameters contains 't_det', which is a sampling-space "
            "parameter tied to a specific detector. Use 't_c' (geocentric arrival "
            "time) instead."
        )
    if "azimuth" in params or "zenith" in params:
        raise ValueError(
            "injection_parameters contains 'azimuth'/'zenith', which are "
            "sampling-space parameters that depend on the detector network "
            "orientation. Use 'ra' and 'dec' instead."
        )

    p = {k: jnp.float64(v) for k, v in params.items()}

    # J-frame spins → Cartesian + iota
    _J_FRAME = {"theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2"}
    if _J_FRAME <= p.keys():
        t = SpinAnglesToCartesianSpinTransform(freq_ref=waveform_f_ref)
        p = dict(t.forward(p))

    # Spherical per-spin → Cartesian
    for label in ("s1", "s2"):
        if {f"{label}_mag", f"{label}_theta", f"{label}_phi"} <= p.keys():
            t = SphereSpinToCartesianSpinTransform(label)
            p = dict(t.forward(p))

    # q → eta
    if "q" in p and "eta" not in p:
        p = dict(MassRatioToSymmetricMassRatioTransform.forward(p))

    return {k: float(v) for k, v in p.items()}


def _load_injection(
    ifos: list[GroundBased2G],
    cfg: InjectionDataConfig,
    waveform,
    *,
    f_min: float,
    f_max: float,
) -> None:
    if waveform is None:
        raise ValueError(
            "waveform is required for injection data — build it before calling build_data."
        )

    parameters = to_likelihood_space(
        cfg.injection_parameters,
        waveform_f_ref=waveform.f_ref,
    )

    for ifo in ifos:
        logger.info("Loading design PSD for %s", ifo.name)
        ifo.load_and_set_psd()

        logger.info("Injecting signal into %s", ifo.name)
        ifo.inject_signal(
            duration=cfg.duration,
            sampling_frequency=cfg.sampling_frequency,
            trigger_time=cfg.trigger_time,
            waveform_model=waveform,
            parameters=parameters,
            f_min=f_min,
            f_max=f_max,
            zero_noise=cfg.zero_noise,
        )


def _load_files(ifos: list[GroundBased2G], cfg: FileDataConfig) -> None:
    for ifo in ifos:
        strain_path = cfg.strain_files.get(ifo.name)
        psd_path = cfg.psd_files.get(ifo.name)

        if strain_path is None:
            raise ValueError(f"No strain file specified for {ifo.name}")
        if psd_path is None:
            raise ValueError(f"No PSD file specified for {ifo.name}")

        logger.info("Loading %s strain from %s", ifo.name, strain_path)
        ifo.set_data(Data.from_file(str(strain_path)))

        logger.info("Loading %s PSD from %s", ifo.name, psd_path)
        ifo.set_psd(PowerSpectrum.from_file(str(psd_path)))
