import logging
from typing import assert_never

from jimgw.cli._config import (
    DataConfig,
    FileDataConfig,
    GWOSCDataConfig,
    InjectionDataConfig,
)
from jimgw.cli._transforms import to_likelihood_space
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import GroundBased2G, asd_file_dict, get_detector_preset

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
    for name in data_cfg.detectors:
        val = preset[name]
        if isinstance(val, list):
            ifos.extend(val)
        else:
            ifos.append(val)

    if isinstance(data_cfg, GWOSCDataConfig):
        _load_gwosc(ifos, data_cfg)
    elif isinstance(data_cfg, InjectionDataConfig):
        _load_injection(
            ifos, data_cfg, waveform, f_min=f_min, f_max=f_max, time_frame=time_frame
        )
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
        nperseg = int(round(strain.duration * strain.sampling_frequency))
        ifo.set_psd(psd_data.to_psd(nperseg=nperseg))


def _load_injection(
    ifos: list[GroundBased2G],
    cfg: InjectionDataConfig,
    waveform,
    *,
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

    for ifo in ifos:
        if ifo.name not in asd_file_dict:
            raise ValueError(
                f"No default ASD for detector '{ifo.name}'. "
                f"Provide a PSD file via the config. "
                f"Detectors with built-in defaults: {sorted(asd_file_dict)}."
            )
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
        strain_path = cfg.strain_files[ifo.name]
        psd_path = cfg.psd_files[ifo.name]
        channel = cfg.strain_channels.get(ifo.name)
        is_asd = cfg.psd_is_asd.get(ifo.name, False)

        logger.info("Loading %s strain from %s", ifo.name, strain_path)
        ifo.set_data(Data.from_file(str(strain_path), channel=channel))

        logger.info("Loading %s PSD from %s", ifo.name, psd_path)
        ifo.set_psd(PowerSpectrum.from_file(str(psd_path), is_asd=is_asd))
