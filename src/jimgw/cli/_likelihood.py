import logging
from typing import Optional, Union

from ripplegw.interfaces import Waveform

from jimgw.cli._config import (
    CLIInjectionRefParams,
    CLIOptimizerRefParams,
    CLIProvidedRefParams,
    DataConfig,
    LikelihoodConfig,
)
from jimgw.cli._transforms import to_likelihood_space
from jimgw.cli._prior import build_prior
from jimgw.core.prior import CombinePrior
from jimgw.core.single_event.detector import GroundBased2G
from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    TransientLikelihoodFD,
)
from jimgw.core.single_event.marginalization_config import (
    DistanceMargConfig,
    PhaseMargConfig,
    TimeMargConfig,
)
from jimgw.core.transforms import NtoMTransform

logger = logging.getLogger(__name__)


def build_likelihood(
    cfg: LikelihoodConfig,
    ifos: list[GroundBased2G],
    waveform: Waveform,
    trigger_time: float,
    waveform_f_ref: float,
    time_frame: str,
    prior: CombinePrior,
    likelihood_transforms: list[NtoMTransform],
    data_cfg: DataConfig,
) -> Union[TransientLikelihoodFD, HeterodynedTransientLikelihoodFD]:
    """Build a likelihood from the validated likelihood config.

    Uses ``HeterodynedTransientLikelihoodFD`` when ``cfg.heterodyne`` is set,
    otherwise falls back to ``TransientLikelihoodFD``.  ``prior`` and
    ``likelihood_transforms`` are required for the heterodyne case (the optimizer
    needs them to find reference parameters).

    ``data_cfg`` is only used when ``cfg.heterodyne.reference_parameters.type =
    "injection"`` — it must be an ``InjectionDataConfig`` in that case.
    """
    phase_marg = None
    if cfg.phase_marginalization:
        phase_marg = PhaseMargConfig()

    fixed_params = cfg.fixed_parameters if cfg.fixed_parameters else None

    if cfg.heterodyne is not None:
        ref_cfg = cfg.heterodyne.reference_parameters
        reference_params: Optional[dict] = None
        optimizer_popsize = 500
        optimizer_n_steps = 1000

        if isinstance(ref_cfg, CLIOptimizerRefParams):
            optimizer_popsize = ref_cfg.popsize
            optimizer_n_steps = ref_cfg.n_steps
        elif isinstance(ref_cfg, CLIProvidedRefParams):
            reference_params = ref_cfg.values
        elif isinstance(ref_cfg, CLIInjectionRefParams):
            reference_params = to_likelihood_space(
                data_cfg.injection_parameters,  # type: ignore[attr-defined]
                waveform_f_ref=waveform_f_ref,
                trigger_time=trigger_time,
                ifos=ifos,
                time_frame=time_frame,
            )
            logger.info(
                "Using injection parameters as heterodyne reference: %s",
                reference_params,
            )

        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            fixed_parameters=fixed_params,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            trigger_time=trigger_time,
            n_bins=cfg.heterodyne.n_bins,
            optimizer_popsize=optimizer_popsize,
            optimizer_n_steps=optimizer_n_steps,
            prior=prior,
            likelihood_transforms=likelihood_transforms,
            phase_marginalization=phase_marg,
            reference_parameters=reference_params,
        )
        logger.info(
            "Built heterodyne likelihood: f_min=%.1f, f_max=%.1f, n_bins=%d",
            cfg.f_min,
            cfg.f_max,
            cfg.heterodyne.n_bins,
        )
        return likelihood

    time_marg = None
    if cfg.time_marginalization is not None:
        time_marg = TimeMargConfig(tc_range=cfg.time_marginalization.tc_range)

    dist_marg = None
    if cfg.distance_marginalization is not None:
        dist_combined = build_prior(cfg.distance_marginalization.distance_prior)
        dist_marg = DistanceMargConfig(
            distance_prior=dist_combined.base_prior[0],
            n_dist_points=cfg.distance_marginalization.n_dist_points,
            ref_dist=cfg.distance_marginalization.ref_dist,
        )

    likelihood = TransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        fixed_parameters=fixed_params,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        trigger_time=trigger_time,
        phase_marginalization=phase_marg,
        time_marginalization=time_marg,
        distance_marginalization=dist_marg,
    )
    logger.info(
        "Built likelihood: f_min=%.1f, f_max=%.1f, trigger_time=%.3f",
        cfg.f_min,
        cfg.f_max,
        trigger_time,
    )
    return likelihood
