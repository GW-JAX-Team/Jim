import logging
from typing import Optional, assert_never

import jax

from jimgw.cli._config import (
    DataConfig,
    FileDataConfig,
    GWOSCDataConfig,
    InjectionDataConfig,
)
from jimgw.cli._transforms import to_likelihood_space
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import GroundBased2G, get_detector_preset

logger = logging.getLogger(__name__)


def build_data(
    data_cfg: DataConfig,
    f_min: float,
    f_max: float,
    waveform=None,
    time_frame: str = "detector",
) -> list[GroundBased2G]:
    """Construct a list of detectors populated with strain data and PSDs.

    For injection runs, *waveform* is required and is used to inject the signal.
    """
    preset = get_detector_preset()

    ifos: list[GroundBased2G] = []
    # Interferometer name -> its name in ``data.detectors`` ("ET" covers ET1-ET3).
    # PSD/ASD tables are keyed by the latter.
    config_name: dict[str, str] = {}
    for name in data_cfg.detectors:
        val = preset[name]
        group = val if isinstance(val, list) else [val]
        ifos.extend(group)
        config_name.update({ifo.name: name for ifo in group})

    if isinstance(data_cfg, GWOSCDataConfig):
        _load_gwosc(ifos, data_cfg)
    elif isinstance(data_cfg, InjectionDataConfig):
        _load_injection(
            ifos,
            data_cfg,
            waveform,
            config_name=config_name,
            f_min=f_min,
            f_max=f_max,
            time_frame=time_frame,
        )
    elif isinstance(data_cfg, FileDataConfig):
        _load_files(ifos, data_cfg, config_name=config_name)
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
        nperseg = round(strain.duration * strain.sampling_frequency)
        ifo.set_psd(psd_data.to_psd(nperseg=nperseg))


def _read_psd(
    cfg: InjectionDataConfig | FileDataConfig, detector: str
) -> Optional[PowerSpectrum]:
    """Load the PSD or ASD configured for *detector*, or None if no file is set."""
    if cfg.psd_files is not None:
        path, is_asd = cfg.psd_files[detector], False
    elif cfg.asd_files is not None:
        path, is_asd = cfg.asd_files[detector], True
    else:
        return None
    logger.info("Loading %s %s from %s", detector, "ASD" if is_asd else "PSD", path)
    return PowerSpectrum.from_file(str(path), is_asd=is_asd)


def _load_injection(
    ifos: list[GroundBased2G],
    cfg: InjectionDataConfig,
    waveform,
    *,
    config_name: dict[str, str],
    f_min: float,
    f_max: float,
    time_frame: str = "detector",
) -> None:
    parameters = to_likelihood_space(
        cfg.injection_parameters,
        waveform_f_ref=waveform.f_ref,
        trigger_time=cfg.trigger_time,
        ifos=ifos,
        time_frame=time_frame,
    )

    noise_key = None
    if not cfg.zero_noise:
        assert cfg.noise_seed is not None  # required by InjectionDataConfig
        noise_key = jax.random.key(cfg.noise_seed)
        logger.info("Injected noise seeded with noise_seed=%d", cfg.noise_seed)

    for ifo in ifos:
        # The PSD is set once, here.  ``inject_signal`` draws the noise from it
        # and the likelihood reads it back, so both always use the same PSD.
        psd = _read_psd(cfg, config_name[ifo.name])
        if psd is None:
            # Config validation guarantees a built-in default exists here.
            logger.info("Loading built-in O3 ASD for %s", ifo.name)
            ifo.load_and_set_psd()
        else:
            ifo.set_psd(psd)

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
            rng_key=noise_key,
        )


def _load_files(
    ifos: list[GroundBased2G], cfg: FileDataConfig, *, config_name: dict[str, str]
) -> None:
    for ifo in ifos:
        strain_path = cfg.strain_files[ifo.name]
        channel = cfg.strain_channels.get(ifo.name)

        logger.info("Loading %s strain from %s", ifo.name, strain_path)
        ifo.set_data(Data.from_file(str(strain_path), channel=channel))

        psd = _read_psd(cfg, config_name[ifo.name])
        assert psd is not None  # FileDataConfig requires psd_files or asd_files
        ifo.set_psd(psd)
