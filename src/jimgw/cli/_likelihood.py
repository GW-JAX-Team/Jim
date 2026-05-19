import logging
from typing import Optional, Union

import jax.numpy as jnp
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
from jimgw.core.constants import EARTH_RADIUS_LIGHT_S
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.detector import GroundBased2G
from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
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
) -> Union[
    TransientLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
]:
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
            # Phase-marginalised heterodyned likelihood with the optimizer: the
            # optimizer needs phase_c in the prior to search over it, but the
            # user should not have to (and must not) include it themselves since
            # it is a marginalised parameter.  Add a default Uniform(0, 2π)
            # component here; the caller's `prior` (without phase_c) is still
            # passed to Jim.__init__ unchanged.
            if cfg.phase_marginalization and "phase_c" not in prior.parameter_names:
                prior = CombinePrior(
                    list(prior.base_prior)
                    + [UniformPrior(0.0, 2 * jnp.pi, ["phase_c"])]
                )
                logger.info(
                    "Added Uniform(0, 2π) prior on phase_c for optimizer reference parameter search"
                )
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

    if cfg.multiband is not None:
        mb = cfg.multiband

        # MultibandedTransientLikelihoodFD._infer_time_offsets only searches for
        # t_c. When the NS-AW sampler is used it renames t_c → t_det in the built
        # prior (same relative-offset bounds). Detect that here and compute
        # time_offset / delta_f_end explicitly so auto-inference works correctly.
        mb_time_offset = mb.time_offset
        mb_delta_f_end = mb.delta_f_end
        if (mb_time_offset is None or mb_delta_f_end is None) and (
            "t_c" not in prior.parameter_names and "t_det" in prior.parameter_names
        ):
            tdet_comp = next(
                (
                    p
                    for p in prior.base_prior
                    if "t_det" in p.parameter_names and hasattr(p, "xmin")
                ),
                None,
            )
            if tdet_comp is not None:
                ref_ifo = (
                    ifos[0]
                    if time_frame == "detector"
                    else next(ifo for ifo in ifos if ifo.name == time_frame)
                )
                t_end = (
                    float(ref_ifo.data.start_time)
                    + float(ref_ifo.data.duration)
                    - trigger_time
                )
                s = EARTH_RADIUS_LIGHT_S
                if mb_time_offset is None:
                    mb_time_offset = t_end - float(getattr(tdet_comp, "xmin")) + s
                    logger.info(
                        "time_offset inferred from t_det prior: %.4f s", mb_time_offset
                    )
                if mb_delta_f_end is None:
                    xmax = float(getattr(tdet_comp, "xmax"))
                    denom = t_end - xmax - s
                    if denom <= 0:
                        raise ValueError(
                            f"Cannot infer delta_f_end from t_det prior: "
                            f"t_end - xmax - s = {t_end:.4f} - {xmax:.4f} - {s:.6f} = {denom:.6f} <= 0. "
                            "Check that the t_det prior upper bound is well within the data segment."
                        )
                    mb_delta_f_end = 100.0 / denom
                    logger.info(
                        "delta_f_end inferred from t_det prior: %.4f Hz", mb_delta_f_end
                    )

        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            fixed_parameters=fixed_params,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            trigger_time=trigger_time,
            highest_mode=mb.highest_mode,
            accuracy_factor=mb.accuracy_factor,
            prior=prior,
            reference_chirp_mass=mb.reference_chirp_mass,
            time_offset=mb_time_offset,
            delta_f_end=mb_delta_f_end,
            max_banding_frequency=mb.max_banding_frequency,
            min_banding_duration=mb.min_banding_duration,
        )
        logger.info(
            "Built multiband likelihood: f_min=%.1f, f_max=%.1f, "
            "reference_chirp_mass=%.4f M_sun",
            cfg.f_min,
            cfg.f_max,
            likelihood.reference_chirp_mass,
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
