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

import jax.numpy as jnp
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
from jimgw.cli._utils import (
    DETECTOR_SKY_PARAMS as _DETECTOR_SKY_PARAMS,
    EQUATORIAL_SKY_PARAMS as _EQUATORIAL_SKY_PARAMS,
    J_FRAME_SPIN_PARAMS as _J_FRAME_SPIN_PARAMS,
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
    GaussianTransform,
    NtoMTransform,
    PowerLawTransform,
    RayleighTransform,
    SineTransform,
    reverse_bijective_transform,
)

logger = logging.getLogger(__name__)

_TWO_PI = 2 * _PI


def infer_sample_transforms(
    prior_params: frozenset[str],
    trigger_time: float,
    ifos: list[GroundBased2G],
    sampling_cfg: SamplingConfig,
    *,
    unit_cube: bool = False,
    prior_cfg: PriorConfig,
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
        ifo_for_time = next(ifo for ifo in ifos if ifo.name == ifo_name)
        sample_transforms.append(
            GeocentricArrivalTimeToDetectorArrivalTimeTransform(
                trigger_time, ifo_for_time
            )
        )
        logger.debug(
            "Added GeocentricArrivalTimeToDetectorArrivalTimeTransform(detector=%s)",
            ifo_name,
        )
    elif has_geocentric_time:
        logger.debug(
            "time_frame='geocentric': sampling t_c directly, no time sample transform"
        )

    # --- Sky sample transform ---------------------------------------------
    has_equatorial_sky = _EQUATORIAL_SKY_PARAMS <= prior_params

    if has_equatorial_sky and sampling_cfg.sky_frame == "detector":
        sample_transforms.append(
            SkyFrameToDetectorFrameSkyPositionTransform(trigger_time, ifos)
        )
        logger.debug("Added SkyFrameToDetectorFrameSkyPositionTransform")

    # --- Unit-cube transforms (NS-AW) -------------------------------------
    if unit_cube:
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
    phase_marginalization: bool = False,
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
        phase_marginalization: When ``True``, ``phase_c`` is not a free
            parameter, so ``SpinAnglesToCartesianSpinTransform`` is built
            with ``fixed_phase=True`` (uses ``phase_c=0``).

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
            SpinAnglesToCartesianSpinTransform(
                freq_ref=waveform_f_ref, fixed_phase=phase_marginalization
            )
        )
        logger.debug(
            "Added SpinAnglesToCartesianSpinTransform(freq_ref=%.1f, fixed_phase=%s)",
            waveform_f_ref,
            phase_marginalization,
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
                ifo for ifo in ifos if ifo.name == sampling_cfg.time_frame
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

    Raises ``AssertionError`` for unknown spec types.
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
    if isinstance(spec, RayleighSpec):
        # RayleighPrior is built as UniformPrior(0,1) ∘ RayleighTransform (the CDF map
        # u → scale*sqrt(-2*log(u))).  The inverse CDF maps x → exp(-(x/scale)²/2) ∈ [0,1].
        return [
            reverse_bijective_transform(
                RayleighTransform(
                    name_mapping=([f"{name}_unit"], [name]),
                    sigma=spec.scale,
                )
            )
        ]
    if isinstance(spec, GaussianSpec):
        # GaussianPrior: use the inverse CDF (probit) to map u ∈ (0,1) → x ∈ ℝ.
        # GaussianTransform forward maps u → mu + sigma*ndtri(u); reverse maps x → ndtr((x-mu)/sigma) ∈ (0,1).
        return [
            reverse_bijective_transform(
                GaussianTransform(
                    name_mapping=([f"{name}_unit"], [name]),
                    mu=spec.loc,
                    sigma=spec.scale,
                )
            )
        ]
    raise TypeError(f"Unknown prior spec type for '{name}': {type(spec)}")


# ---------------------------------------------------------------------------
# Parameter-space conversion utility
# ---------------------------------------------------------------------------


def to_likelihood_space(
    params: dict[str, float],
    waveform_f_ref: float,
    trigger_time: float,
    ifos: list[GroundBased2G],
    time_frame: str,
) -> dict[str, float]:
    """Convert injection/reference parameters to likelihood space if needed.

    Handles: q→eta, J-frame spins→Cartesian, spherical spins→Cartesian,
    azimuth/zenith→ra/dec, t_det→t_c.

    The detector-frame conversions (azimuth/zenith and t_det) require
    ``trigger_time`` and ``ifos`` to be supplied.  These parameters are valid
    when the prior is parametrized in detector frame (e.g. ``azimuth``,
    ``zenith``, or ``t_det`` appear in ``[prior]``).

    ``time_frame`` mirrors ``[sampling].time_frame`` and selects which detector is
    used for the ``t_det`` ↔ ``t_c`` conversion.  Pass the same value that is
    used in the sampling config to guarantee consistency with the actual
    likelihood transforms.
    """
    p = {k: jnp.float64(v) for k, v in params.items()}

    # azimuth/zenith → ra/dec (must come before t_det conversion, which needs ra/dec)
    if _DETECTOR_SKY_PARAMS <= p.keys():
        t_sky = SkyFrameToDetectorFrameSkyPositionTransform(trigger_time, ifos)
        p = dict(t_sky.backward(p))

    # t_det → t_c (needs ra/dec as conditioning parameters)
    if "t_det" in p:
        if time_frame in ("detector", "geocentric"):
            ifo_for_time = ifos[0]
        else:
            ifo_for_time = next(ifo for ifo in ifos if ifo.name == time_frame)
        t_time = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time, ifo_for_time
        )
        p = dict(t_time.backward(p))

    # J-frame spins → Cartesian + iota
    if _J_FRAME_SPIN_PARAMS <= p.keys():
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
