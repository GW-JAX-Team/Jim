import logging
import math
from typing import Optional, Union

import jax.numpy as jnp
from ripplegw.interfaces import Waveform

from jimgw.cli._config import (
    CLIInjectionRefParams,
    CLIMultibandedConfig,
    CLIOptimizerRefParams,
    CLIProvidedRefParams,
    DataConfig,
    LikelihoodConfig,
    PowerLawSpec,
    PriorConfig,
    UniformSpec,
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

_DEFAULT_MULTIBAND_TIME_OFFSET = 2.12
_DEFAULT_MULTIBAND_DELTA_F_END = 53.0


def _finite_bounds(
    prior_config: PriorConfig, parameter_name: str
) -> Optional[tuple[float, float]]:
    """Return finite bounds for a CLI prior parameter when they are explicit."""
    spec = prior_config.root.get(parameter_name)
    if not isinstance(spec, (UniformSpec, PowerLawSpec)):
        return None
    if not (math.isfinite(spec.min) and math.isfinite(spec.max)):
        return None
    return float(spec.min), float(spec.max)


def _resolve_multiband_settings(
    config: CLIMultibandedConfig,
    prior_config: PriorConfig,
    ifos: list[GroundBased2G],
    trigger_time: float,
    time_frame: str,
) -> tuple[float, float, float]:
    """Resolve CLI multiband settings without passing the sampling prior to core."""
    reference_chirp_mass = config.reference_chirp_mass
    if reference_chirp_mass is None:
        mc_bounds = _finite_bounds(prior_config, "M_c")
        if mc_bounds is None:
            raise ValueError(
                "multiband.reference_chirp_mass must be set when [prior].M_c "
                "is not a finite bounded uniform or power_law prior."
            )
        reference_chirp_mass = mc_bounds[0]
        logger.info(
            "reference_chirp_mass inferred from M_c prior minimum: %.4f M_sun",
            reference_chirp_mass,
        )

    time_offset = config.time_offset
    delta_f_end = config.delta_f_end
    if time_offset is not None and delta_f_end is not None:
        return reference_chirp_mass, time_offset, delta_f_end

    time_parameter = "t_c"
    time_bounds = _finite_bounds(prior_config, time_parameter)
    if time_bounds is not None:
        t_end = min(
            float(ifo.data.start_time) + float(ifo.data.duration) - trigger_time
            for ifo in ifos
        )
    else:
        time_parameter = "t_det"
        time_bounds = _finite_bounds(prior_config, time_parameter)
        if time_bounds is not None:
            # Preserve the bilby_pipe-style detector-time convention used by the
            # CLI's NS-AW t_c -> t_det adaptation.
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

    if time_bounds is not None:
        time_min, time_max = time_bounds
        if time_offset is None:
            time_offset = t_end - time_min + EARTH_RADIUS_LIGHT_S
            logger.info(
                "time_offset inferred from %s prior: %.4f s",
                time_parameter,
                time_offset,
            )
        if delta_f_end is None:
            denom = t_end - time_max - EARTH_RADIUS_LIGHT_S
            if denom <= 0:
                raise ValueError(
                    f"Cannot infer delta_f_end from {time_parameter} prior: "
                    f"t_end - xmax - s = {t_end:.4f} - {time_max:.4f} - "
                    f"{EARTH_RADIUS_LIGHT_S:.6f} = {denom:.6f} <= 0. "
                    "Check that the time prior upper bound is well within the data segment."
                )
            delta_f_end = 100.0 / denom
            logger.info(
                "delta_f_end inferred from %s prior: %.4f Hz",
                time_parameter,
                delta_f_end,
            )

    if time_offset is None:
        time_offset = _DEFAULT_MULTIBAND_TIME_OFFSET
        logger.warning(
            "time_offset cannot be inferred from prior; using default %.2f s",
            time_offset,
        )
    if delta_f_end is None:
        delta_f_end = _DEFAULT_MULTIBAND_DELTA_F_END
        logger.warning(
            "delta_f_end cannot be inferred from prior; using default %.1f Hz",
            delta_f_end,
        )
    return reference_chirp_mass, time_offset, delta_f_end


def build_likelihood(
    cfg: LikelihoodConfig,
    ifos: list[GroundBased2G],
    waveform: Waveform,
    trigger_time: float,
    waveform_f_ref: float,
    time_frame: str,
    prior: CombinePrior,
    prior_config: PriorConfig,
    likelihood_transforms: list[NtoMTransform],
    data_cfg: DataConfig,
) -> Union[
    TransientLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
]:
    """Build a likelihood from the validated likelihood config.

    Uses ``HeterodynedTransientLikelihoodFD`` when ``cfg.heterodyne`` is set,
    otherwise falls back to ``TransientLikelihoodFD``. ``prior`` and
    ``likelihood_transforms`` are required for the heterodyne case; ``prior_config``
    is used only to resolve automatic multiband settings.

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
            epsilon=cfg.heterodyne.epsilon,
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
            likelihood.n_bins,
        )
        return likelihood

    if cfg.multiband is not None:
        mb = cfg.multiband
        reference_chirp_mass, time_offset, delta_f_end = _resolve_multiband_settings(
            mb, prior_config, ifos, trigger_time, time_frame
        )

        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            reference_chirp_mass=reference_chirp_mass,
            fixed_parameters=fixed_params,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            trigger_time=trigger_time,
            highest_mode=mb.highest_mode,
            accuracy_factor=mb.accuracy_factor,
            time_offset=time_offset,
            delta_f_end=delta_f_end,
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
