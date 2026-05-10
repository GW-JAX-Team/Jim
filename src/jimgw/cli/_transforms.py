"""Transform inference: validated config + prior params → sample/likelihood transforms.

Given the set of parameter names that appear in [prior] and the [sampling]
choices, this module resolves the full transform chain:

    Prior space → (sample transforms) → Sampling space

When Jim evaluates the likelihood it reverses the sample transforms to get back
to prior space, then applies likelihood transforms:

    Sampling space → (reverse sample transforms) → Prior space
                  → (likelihood transforms) → Likelihood space

Likelihood transforms therefore always operate on prior-space parameters, not
sampling-space parameters. No user configuration of transforms is needed —
they are derived automatically from the parametrization.
"""

import logging
from typing import Optional

from jax.numpy import pi as _PI

from jimgw.cli._config import (
    CosineSpec,
    GaussianSpec,
    PowerLawSpec,
    PriorConfig,
    RayleighSpec,
    SamplingConfig,
    SineSpec,
    UniformSpec,
    UniformSphereSpec,
)
from jimgw.core.single_event.detector import GroundBased2G
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
    SpinAnglesToCartesianSpinTransform,
    SphereSpinToCartesianSpinTransform,
)
from jimgw.core.transforms import (
    BijectiveTransform,
    BoundToBound,
    CosineTransform,
    NtoMTransform,
    PowerLawTransform,
    SineTransform,
    reverse_bijective_transform,
)

logger = logging.getLogger(__name__)

_TWO_PI = 2 * _PI

# ---------------------------------------------------------------------------
# Parameter group constants
# ---------------------------------------------------------------------------

_J_FRAME_SPIN_PARAMS = frozenset(
    {"theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2"}
)
_SPHERE_SPIN_LABELS = ("s1", "s2")
_ALIGNED_SPIN_PARAMS = frozenset({"s1_z", "s2_z"})
_CARTESIAN_SPIN_PARAMS = frozenset({"s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z"})

_EQUATORIAL_SKY_PARAMS = frozenset({"ra", "dec"})
_DETECTOR_SKY_PARAMS = frozenset({"azimuth", "zenith"})


def infer_sample_transforms(
    prior_params: frozenset[str],
    trigger_time: float,
    ifos: list[GroundBased2G],
    sampling_cfg: SamplingConfig,
    *,
    unit_cube: bool = False,
    prior_cfg: Optional[PriorConfig] = None,
) -> list[BijectiveTransform]:
    """Infer sample transforms (prior space → sampling space).

    Args:
        prior_params: Set of parameter names appearing in [prior].
        trigger_time: GPS trigger time (from data config).
        ifos: List of detectors (needed for sky/time transforms).
        sampling_cfg: [sampling] section config.
        unit_cube: When ``True``, append BoundToBound/auxiliary transforms
            that map every sampling-space parameter to [0, 1] (required by the
            NS-AW sampler).  ``prior_cfg`` must be supplied in this case.
        prior_cfg: Prior config (required when ``unit_cube=True``).

    Returns:
        List of bijective transforms to pass to Jim as ``sample_transforms``.
    """
    sample_transforms: list[BijectiveTransform] = []

    # --- Time sample transform --------------------------------------------
    has_geocentric_time = "t_c" in prior_params

    if has_geocentric_time and sampling_cfg.time_frame != "geocentric":
        ifo_name = (
            ifos[0].name
            if sampling_cfg.time_frame == "detector"
            else sampling_cfg.time_frame
        )
        ifo_for_time = next((ifo for ifo in ifos if ifo.name == ifo_name), None)
        if ifo_for_time is None:
            raise ValueError(
                f"[sampling] time_frame={sampling_cfg.time_frame!r} is not in the IFO list "
                f"{[i.name for i in ifos]}"
            )
        sample_transforms.append(
            GeocentricArrivalTimeToDetectorArrivalTimeTransform(
                trigger_time, ifo_for_time
            )
        )
        logger.debug(
            "Added GeocentricArrivalTimeToDetectorArrivalTimeTransform(ifo=%s)",
            ifo_name,
        )
    elif has_geocentric_time:
        logger.debug(
            "time_frame='geocentric': sampling t_c directly, no time sample transform"
        )

    # --- Sky sample transform ---------------------------------------------
    has_equatorial_sky = _EQUATORIAL_SKY_PARAMS <= prior_params

    if has_equatorial_sky and sampling_cfg.sky_frame == "detector":
        if len(ifos) < 2:
            raise ValueError(
                "SkyFrameToDetectorFrameSkyPositionTransform requires at least 2 IFOs; "
                f"got {[i.name for i in ifos]}"
            )
        sample_transforms.append(
            SkyFrameToDetectorFrameSkyPositionTransform(trigger_time, ifos)
        )
        logger.debug("Added SkyFrameToDetectorFrameSkyPositionTransform")

    # --- Unit-cube transforms (NS-AW) -------------------------------------
    if unit_cube:
        if prior_cfg is None:
            raise ValueError("prior_cfg must be supplied when unit_cube=True")
        sample_transforms.extend(
            _build_unit_cube_transforms(prior_params, prior_cfg, sampling_cfg)
        )

    logger.info("sample transforms: %s", [type(t).__name__ for t in sample_transforms])
    return sample_transforms


def infer_likelihood_transforms(
    prior_params: frozenset[str],
    trigger_time: float,
    ifos: list[GroundBased2G],
    sampling_cfg: SamplingConfig,
    waveform_f_ref: float,
) -> list[NtoMTransform]:
    """Infer likelihood transforms (prior space → likelihood space).

    Args:
        prior_params: Set of parameter names appearing in [prior].
        trigger_time: GPS trigger time (from data config).
        ifos: List of detectors (needed for sky/time reverse transforms).
        sampling_cfg: [sampling] section config.
        waveform_f_ref: Reference frequency from [waveform] — passed to
            ``SpinAnglesToCartesianSpinTransform`` to match the waveform
            spin-angle convention (same as bilby's ``reference_frequency``).

    Returns:
        List of N-to-M transforms to pass to Jim as ``likelihood_transforms``.
    """
    likelihood_transforms: list[NtoMTransform] = []

    has_j_frame = bool(prior_params & _J_FRAME_SPIN_PARAMS)
    has_sphere_s1 = all(f"s1_{s}" in prior_params for s in ("mag", "theta", "phi"))
    has_sphere_s2 = all(f"s2_{s}" in prior_params for s in ("mag", "theta", "phi"))
    has_equatorial_sky = _EQUATORIAL_SKY_PARAMS <= prior_params
    has_detector_sky = _DETECTOR_SKY_PARAMS <= prior_params
    has_geocentric_time = "t_c" in prior_params
    has_detector_time = "t_det" in prior_params

    # q → eta
    if "q" in prior_params:
        likelihood_transforms.append(MassRatioToSymmetricMassRatioTransform)
        logger.debug("Added MassRatioToSymmetricMassRatioTransform")

    # J-frame spin angles → Cartesian spins + iota
    if has_j_frame:
        likelihood_transforms.append(
            SpinAnglesToCartesianSpinTransform(freq_ref=waveform_f_ref)
        )
        logger.debug(
            "Added SpinAnglesToCartesianSpinTransform(freq_ref=%.1f)",
            waveform_f_ref,
        )

    # Spherical per-spin → Cartesian
    if has_sphere_s1:
        likelihood_transforms.append(SphereSpinToCartesianSpinTransform("s1"))
        logger.debug("Added SphereSpinToCartesianSpinTransform(s1)")
    if has_sphere_s2:
        likelihood_transforms.append(SphereSpinToCartesianSpinTransform("s2"))
        logger.debug("Added SphereSpinToCartesianSpinTransform(s2)")

    # azimuth/zenith directly in prior (no sky sample transform) → ra/dec
    if has_detector_sky and not has_equatorial_sky:
        if len(ifos) < 2:
            raise ValueError(
                "Detector-frame sky prior requires at least 2 IFOs for the reverse transform."
            )
        t_sky = SkyFrameToDetectorFrameSkyPositionTransform(trigger_time, ifos)
        likelihood_transforms.append(reverse_bijective_transform(t_sky))
        logger.debug(
            "Added reverse SkyFrameToDetectorFrameSkyPositionTransform "
            "(detector-frame sky prior → ra/dec)"
        )

    # t_det in prior → reverse GeocentricArrivalTime transform to produce t_c
    if has_detector_time and not has_geocentric_time:
        if sampling_cfg.time_frame in ("detector", "geocentric"):
            ifo_for_time = ifos[0]
        else:
            ifo_for_time = next(
                (ifo for ifo in ifos if ifo.name == sampling_cfg.time_frame), None
            )
            if ifo_for_time is None:
                raise ValueError(
                    f"[sampling] time_frame={sampling_cfg.time_frame!r} is not in the IFO list "
                    f"{[i.name for i in ifos]}"
                )
        t_time = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time, ifo_for_time
        )
        likelihood_transforms.append(reverse_bijective_transform(t_time))
        logger.debug(
            "Added reverse GeocentricArrivalTimeToDetectorArrivalTimeTransform "
            "(t_det prior → t_c for likelihood)"
        )

    logger.info(
        "likelihood transforms: %s",
        [
            t.__name__ if hasattr(t, "__name__") else type(t).__name__
            for t in likelihood_transforms
        ],
    )
    return likelihood_transforms


def validate_config(prior_params: frozenset[str], sampling_cfg: SamplingConfig) -> None:
    """Raise ValueError for any invalid prior/sampling configuration combination."""

    # Spin group mutual exclusivity
    has_j_frame = bool(prior_params & _J_FRAME_SPIN_PARAMS)
    has_sphere_spin = any(
        all(f"{label}_{s}" in prior_params for s in ("mag", "theta", "phi"))
        for label in _SPHERE_SPIN_LABELS
    )
    has_cartesian_spin = bool(prior_params & _CARTESIAN_SPIN_PARAMS)

    active_spin_groups = sum([has_j_frame, has_sphere_spin, has_cartesian_spin])
    if active_spin_groups > 1:
        raise ValueError(
            "Spin parametrizations are mutually exclusive. "
            "Found more than one of: J-frame angles, spherical per-spin, Cartesian/aligned spins. "
            f"Prior parameters: {sorted(prior_params)}"
        )

    if has_j_frame and "iota" in prior_params:
        raise ValueError(
            "J-frame spin angles produce 'iota' — 'iota' must not also appear in [prior]."
        )

    if has_j_frame:
        missing_j = _J_FRAME_SPIN_PARAMS - prior_params
        if missing_j:
            raise ValueError(
                f"J-frame spin parametrization requires all 7 parameters; "
                f"missing from [prior]: {sorted(missing_j)}"
            )
        missing_mass = {"M_c", "q"} - prior_params
        if missing_mass:
            raise ValueError(
                f"SpinAnglesToCartesianSpinTransform requires {missing_mass} in [prior] "
                "as conditioning parameters."
            )

    # Sky group mutual exclusivity
    has_equatorial_sky = bool(prior_params & _EQUATORIAL_SKY_PARAMS)
    has_detector_sky = bool(prior_params & _DETECTOR_SKY_PARAMS)
    if has_equatorial_sky and has_detector_sky:
        raise ValueError(
            "Sky parametrizations are mutually exclusive: "
            "cannot have both ra/dec and azimuth/zenith in [prior]."
        )

    # Time group mutual exclusivity
    has_geocentric_time = "t_c" in prior_params
    has_detector_time = "t_det" in prior_params
    if has_geocentric_time and has_detector_time:
        raise ValueError(
            "Time parametrizations are mutually exclusive: "
            "cannot have both t_c and t_det in [prior]."
        )

    # Prior/sampling consistency
    if has_detector_time and sampling_cfg.time_frame == "geocentric":
        raise ValueError(
            "t_det is in [prior] but time_frame='geocentric' requests geocentric-time sampling. "
            "Either remove t_det from [prior] and use t_c, or set time_frame to a detector name."
        )
    if has_detector_sky and sampling_cfg.sky_frame == "geocentric":
        raise ValueError(
            "azimuth/zenith are in [prior] but sky_frame='geocentric' requests equatorial-sky "
            "sampling. Either remove azimuth/zenith from [prior] and use ra/dec, or set "
            "sky_frame='detector'."
        )


# ---------------------------------------------------------------------------
# Unit-cube transforms (for NS-AW sampler)
# ---------------------------------------------------------------------------


def _build_unit_cube_transforms(
    prior_params: frozenset[str],
    prior_cfg: PriorConfig,
    sampling_cfg: SamplingConfig,
) -> list[BijectiveTransform]:
    """Build BoundToBound (and auxiliary) transforms that map every sampling-space
    parameter to [0, 1], as required by the blackjax-ns-aw sampler.

    Must be called AFTER ``adapt_prior_for_ns_time`` (which converts any ``t_c``
    prior to ``t_det`` with exact bounds) and AFTER ``infer_sample_transforms``.
    The returned list is appended to ``sample_transforms``.

    Raises ``ValueError`` for prior types with infinite support (Gaussian, Rayleigh,
    UniformSphere) that cannot be mapped to a bounded interval.
    """
    unit_transforms: list[BijectiveTransform] = []

    has_equatorial_sky = _EQUATORIAL_SKY_PARAMS <= prior_params
    sky_transform_applied = has_equatorial_sky and sampling_cfg.sky_frame == "detector"

    # ra/dec are consumed by the sky sample transform; unit cube is applied to the
    # transform outputs (azimuth, zenith) instead, since their push-forward priors
    # are flat in those coordinates.
    consumed = _EQUATORIAL_SKY_PARAMS if sky_transform_applied else set()

    # All other prior parameters (including J-frame spin angles and spherical spin
    # components) stay in sampling space for NS-AW; their physics transforms live in
    # likelihood_transforms. Just apply the prior-spec-based unit-cube transform for each.
    for name, spec in prior_cfg.root.items():
        if name in consumed:
            continue
        if isinstance(spec, UniformSphereSpec):
            # Expands to: {name}_mag ~ Uniform(0, 1), {name}_theta ~ Sine, {name}_phi ~ Uniform(0, 2π)
            unit_transforms.extend(
                _unit_cube_for_spec(f"{name}_mag", UniformSpec(min=0.0, max=1.0))
            )
            unit_transforms.extend(_unit_cube_for_spec(f"{name}_theta", SineSpec()))
            unit_transforms.extend(
                _unit_cube_for_spec(
                    f"{name}_phi", UniformSpec(min=0.0, max=float(_TWO_PI))
                )
            )
            continue
        unit_transforms.extend(_unit_cube_for_spec(name, spec))

    if sky_transform_applied:
        # azimuth ∈ [0, 2π] — linear rescale
        unit_transforms.append(
            BoundToBound(
                name_mapping=(["azimuth"], ["azimuth_unit"]),
                original_lower_bound=0.0,
                original_upper_bound=float(_TWO_PI),
                target_lower_bound=0.0,
                target_upper_bound=1.0,
            )
        )
        # zenith ∈ [0, π] — cosine transform then rescale
        unit_transforms.append(
            CosineTransform(name_mapping=(["zenith"], ["cos_zenith"]))
        )
        unit_transforms.append(
            BoundToBound(
                name_mapping=(["cos_zenith"], ["cos_zenith_unit"]),
                original_lower_bound=-1.0,
                original_upper_bound=1.0,
                target_lower_bound=0.0,
                target_upper_bound=1.0,
            )
        )

    logger.info("Added %d unit-cube transforms for NS sampler", len(unit_transforms))
    return unit_transforms


def _unit_cube_for_spec(name: str, spec) -> list:
    """Return the BoundToBound (and auxiliary) transforms for one prior parameter."""
    if isinstance(spec, UniformSpec):
        return [
            BoundToBound(
                name_mapping=([name], [f"{name}_unit"]),
                original_lower_bound=spec.min,
                original_upper_bound=spec.max,
                target_lower_bound=0.0,
                target_upper_bound=1.0,
            )
        ]
    if isinstance(spec, SineSpec):
        # SinePrior: x ∈ [0, π], ∝ sin(x) → CosineTransform → [-1, 1] → [0, 1]
        cos_name = f"cos_{name}"
        return [
            CosineTransform(name_mapping=([name], [cos_name])),
            BoundToBound(
                name_mapping=([cos_name], [f"{cos_name}_unit"]),
                original_lower_bound=-1.0,
                original_upper_bound=1.0,
                target_lower_bound=0.0,
                target_upper_bound=1.0,
            ),
        ]
    if isinstance(spec, CosineSpec):
        # CosinePrior: x ∈ [-π/2, π/2], ∝ cos(x) → SineTransform → [-1, 1] → [0, 1]
        sin_name = f"sin_{name}"
        return [
            SineTransform(name_mapping=([name], [sin_name])),
            BoundToBound(
                name_mapping=([sin_name], [f"{sin_name}_unit"]),
                original_lower_bound=-1.0,
                original_upper_bound=1.0,
                target_lower_bound=0.0,
                target_upper_bound=1.0,
            ),
        ]
    if isinstance(spec, PowerLawSpec):
        # PowerLawPrior: use inverse CDF (PowerLawTransform maps [0,1] → [min,max])
        return [
            reverse_bijective_transform(
                PowerLawTransform(
                    name_mapping=([f"{name}_unit"], [name]),
                    xmin=spec.min,
                    xmax=spec.max,
                    alpha=spec.alpha,
                )
            )
        ]
    if isinstance(spec, (GaussianSpec, RayleighSpec)):
        raise ValueError(
            f"Prior type '{type(spec).__name__}' for parameter '{name}' has infinite "
            "support and cannot be automatically mapped to [0, 1] for NS-AW. "
            "Use a bounded prior (uniform, sine, cosine, power_law) instead."
        )
    raise ValueError(f"Unknown prior spec type for '{name}': {type(spec)}")
