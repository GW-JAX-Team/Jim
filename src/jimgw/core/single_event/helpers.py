"""Helpers for setting up a compact-binary analysis.

The ``get_default_*`` functions read the parameter names of a prior (and, for the NS AW
unit cube, its bounds) and return what Jim takes; ``to_likelihood_space`` converts a
parameter dictionary to what the waveform expects.  For anything these do not cover,
write the lists by hand.
"""

from collections.abc import Mapping, Sequence
from typing import Literal

import jax.numpy as jnp
from ripplegw.interfaces import Waveform

from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    GaussianPrior,
    PowerLawPrior,
    Prior,
    RayleighPrior,
    SinePrior,
    UniformPrior,
    find_specific_prior,
)
from jimgw.core.single_event.detector import GroundBased2G
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    SpinAnglesToCartesianSpinTransform,
)
from jimgw.core.transforms import (
    BijectiveTransform,
    BoundToBound,
    BoundToUnbound,
    CosineTransform,
    GaussianTransform,
    NtoMTransform,
    PowerLawTransform,
    RayleighTransform,
    SineTransform,
    reverse_bijective_transform,
)

_J_FRAME_SPIN_PARAMS = frozenset(
    {"theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2"}
)

# Angles with a known period; one wraps around when its prior covers one period.
_PERIODS = {
    "psi": float(jnp.pi),
    "ra": 2 * float(jnp.pi),
    "azimuth": 2 * float(jnp.pi),
    "phase_c": 2 * float(jnp.pi),
    "phi_jl": 2 * float(jnp.pi),
    "phi_12": 2 * float(jnp.pi),
    "s1_phi": 2 * float(jnp.pi),
    "s2_phi": 2 * float(jnp.pi),
}

# Widest delay between the geocentre and a detector (Earth radius / c is
# 0.0213 s), with a margin, in seconds.
_MAX_DETECTOR_DELAY = 0.022


def get_default_sample_transforms(
    prior: Prior,
    ifos: Sequence[GroundBased2G],
    *,
    detector_frame_time: bool = True,
    detector_frame_sky: bool = True,
    space: Literal["default", "unit_cube", "unbounded"] = "default",
) -> list[BijectiveTransform]:
    """Default sample transforms (prior space to sampling space) for ``prior``.

    Args:
        prior: The prior Jim will sample.
        ifos: Detectors with data attached.
        detector_frame_time: Sample ``t_c`` as the arrival time at ``ifos[0]``.
        detector_frame_sky: Sample ``ra``/``dec`` as ``azimuth``/``zenith`` relative to
            the baseline of ``ifos[0]`` and ``ifos[1]``.
        space: ``"default"`` keeps each parameter in its prior range.
            ``"unit_cube"`` maps every sampling parameter to [0, 1], as the NS AW
            sampler requires; it raises if a prior cannot be mapped exactly, e.g.
            ``t_c`` in the detector frame or a non-isotropic ``ra``/``dec``.
            ``"unbounded"`` maps every sampling parameter to the real line (its
            unit-cube map, then a logit), as the SMC sampler prefers, except the
            periodic angles, which the sampler maps itself.  A detector-frame time
            takes the ``t_c`` range widened by the largest detector delay.

    Returns:
        The transforms to pass to Jim as ``sample_transforms``.
    """
    names = set(prior.parameter_names)
    transforms: list[BijectiveTransform] = []

    time_transformed = "t_c" in names and detector_frame_time
    if time_transformed:
        transforms.append(GeocentricArrivalTimeToDetectorArrivalTimeTransform(ifos[0]))

    sky_transformed = {"ra", "dec"} <= names and detector_frame_sky
    if sky_transformed:
        transforms.append(SkyFrameToDetectorFrameSkyPositionTransform(ifos))

    if space == "default":
        return transforms

    if time_transformed and space == "unit_cube":
        raise ValueError(
            "space='unit_cube' needs exact bounds, but sampling 't_c' in the detector "
            "frame shifts it by a sky-dependent delay. Put 't_det' in the prior, or "
            "pass detector_frame_time=False to sample 't_c' directly."
        )

    # CombinePrior only groups independent priors; every other prior (including
    # SinePrior, whose base_prior is an internal helper) is one parameter to map.
    leaves: list[Prior] = []
    pending: list[Prior] = [prior]
    while pending:
        component = pending.pop(0)
        if isinstance(component, CombinePrior):
            pending = list(component.base_prior) + pending
        else:
            leaves.append(component)

    # ra/dec are consumed by the sky transform and t_c by the time transform;
    # azimuth/zenith and t_det get their own maps below.
    consumed = {"ra", "dec"} if sky_transformed else set()
    if time_transformed:
        consumed.add("t_c")
    if sky_transformed and space == "unit_cube":
        ra_prior = find_specific_prior(prior, "ra")
        isotropic_sky = (
            isinstance(ra_prior, UniformPrior)
            and bool(jnp.isclose(ra_prior.xmax - ra_prior.xmin, 2 * jnp.pi))
            and isinstance(find_specific_prior(prior, "dec"), CosinePrior)
        )
        if not isotropic_sky:
            raise ValueError(
                "space='unit_cube' with detector_frame_sky=True assumes an isotropic "
                "sky prior (ra uniform over one period, dec cosine). Put the prior on "
                "azimuth/zenith instead, or pass detector_frame_sky=False to map "
                "ra/dec directly."
            )

    unit_maps: list[list[BijectiveTransform]] = []
    for leaf in leaves:
        name = leaf.parameter_names[0]
        if name in consumed:
            continue
        wraps_around = (
            isinstance(leaf, UniformPrior)
            and name in _PERIODS
            and bool(jnp.isclose(leaf.xmax - leaf.xmin, _PERIODS[name]))
        )
        if space == "unbounded" and wraps_around:
            continue
        if isinstance(leaf, UniformPrior):
            unit_maps.append(
                [
                    BoundToBound(
                        name_mapping=([name], [f"{name}_unit"]),
                        original_lower_bound=leaf.xmin,
                        original_upper_bound=leaf.xmax,
                        target_lower_bound=0.0,
                        target_upper_bound=1.0,
                    )
                ]
            )
        elif isinstance(leaf, SinePrior):
            # density proportional to sin(x): cos(x) is uniform on [-1, 1]
            unit_maps.append(
                [
                    CosineTransform(name_mapping=([name], [f"cos_{name}"])),
                    BoundToBound(
                        name_mapping=([f"cos_{name}"], [f"cos_{name}_unit"]),
                        original_lower_bound=-1.0,
                        original_upper_bound=1.0,
                        target_lower_bound=0.0,
                        target_upper_bound=1.0,
                    ),
                ]
            )
        elif isinstance(leaf, CosinePrior):
            # density proportional to cos(x): sin(x) is uniform on [-1, 1]
            unit_maps.append(
                [
                    SineTransform(name_mapping=([name], [f"sin_{name}"])),
                    BoundToBound(
                        name_mapping=([f"sin_{name}"], [f"sin_{name}_unit"]),
                        original_lower_bound=-1.0,
                        original_upper_bound=1.0,
                        target_lower_bound=0.0,
                        target_upper_bound=1.0,
                    ),
                ]
            )
        elif isinstance(leaf, PowerLawPrior):
            # the inverse CDF, which PowerLawTransform maps [0, 1] -> [xmin, xmax]
            unit_maps.append(
                [
                    reverse_bijective_transform(
                        PowerLawTransform(
                            name_mapping=([f"{name}_unit"], [name]),
                            xmin=leaf.xmin,
                            xmax=leaf.xmax,
                            alpha=leaf.alpha,
                        )
                    )
                ]
            )
        elif isinstance(leaf, RayleighPrior):
            unit_maps.append(
                [
                    reverse_bijective_transform(
                        RayleighTransform(
                            name_mapping=([f"{name}_unit"], [name]),
                            sigma=leaf.sigma,
                        )
                    )
                ]
            )
        elif isinstance(leaf, GaussianPrior):
            unit_maps.append(
                [
                    reverse_bijective_transform(
                        GaussianTransform(
                            name_mapping=([f"{name}_unit"], [name]),
                            mu=leaf.mu,
                            sigma=leaf.sigma,
                        )
                    )
                ]
            )
        else:
            raise TypeError(
                f"No default {space} transform for '{name}' ({type(leaf).__name__}). "
                "Write the transforms for it by hand."
            )

    if sky_transformed:
        if space == "unit_cube":
            unit_maps.append(
                [
                    BoundToBound(
                        name_mapping=(["azimuth"], ["azimuth_unit"]),
                        original_lower_bound=0.0,
                        original_upper_bound=2 * float(jnp.pi),
                        target_lower_bound=0.0,
                        target_upper_bound=1.0,
                    )
                ]
            )
        unit_maps.append(
            [
                CosineTransform(name_mapping=(["zenith"], ["cos_zenith"])),
                BoundToBound(
                    name_mapping=(["cos_zenith"], ["cos_zenith_unit"]),
                    original_lower_bound=-1.0,
                    original_upper_bound=1.0,
                    target_lower_bound=0.0,
                    target_upper_bound=1.0,
                ),
            ]
        )

    if space == "unit_cube":
        return transforms + [t for unit_map in unit_maps for t in unit_map]

    # space == "unbounded": a logit after each unit-cube map.  A linear map onto
    # [0, 1] followed by a logit is a logit of the original range.
    for unit_map in unit_maps:
        last = unit_map[-1]
        (unit_name,) = last.name_mapping[1]
        unbounded_name = unit_name.removesuffix("_unit") + "_unbounded"
        if isinstance(last, BoundToBound):
            unit_map[-1] = BoundToUnbound(
                name_mapping=(last.name_mapping[0], [unbounded_name]),
                original_lower_bound=last.original_lower_bound,
                original_upper_bound=last.original_upper_bound,
            )
        else:
            unit_map.append(
                BoundToUnbound(
                    name_mapping=([unit_name], [unbounded_name]),
                    original_lower_bound=0.0,
                    original_upper_bound=1.0,
                )
            )
        transforms.extend(unit_map)

    if time_transformed:
        t_c_prior = find_specific_prior(prior, "t_c")
        if not isinstance(t_c_prior, UniformPrior):
            raise ValueError(
                "space='unbounded' with detector_frame_time=True needs a uniform 't_c' "
                "prior, whose range sets the range of 't_det'. Put 't_det' in the "
                "prior, or pass detector_frame_time=False to sample 't_c' directly."
            )
        transforms.append(
            BoundToUnbound(
                name_mapping=(["t_det"], ["t_det_unbounded"]),
                original_lower_bound=t_c_prior.xmin - _MAX_DETECTOR_DELAY,
                original_upper_bound=t_c_prior.xmax + _MAX_DETECTOR_DELAY,
            )
        )

    return transforms


def get_default_likelihood_transforms(
    prior: Prior,
    ifos: Sequence[GroundBased2G],
    waveform: Waveform,
    *,
    phase_marginalization: bool = False,
) -> list[NtoMTransform]:
    """Default likelihood transforms (prior space to waveform input) for ``prior``.

    Args:
        prior: The prior Jim will sample.
        ifos: Detectors with data attached.
        waveform: The waveform model; its ``f_ref`` fixes the spin-angle convention.
        phase_marginalization: ``phase_c`` is not a parameter, so the spin angles are
            converted at ``phase_c = 0``.

    Returns:
        The transforms to pass to Jim as ``likelihood_transforms``.
    """
    names = set(prior.parameter_names)
    transforms: list[NtoMTransform] = []

    if _J_FRAME_SPIN_PARAMS <= names:
        transforms.append(
            SpinAnglesToCartesianSpinTransform(
                freq_ref=waveform.f_ref,  # ty: ignore[unresolved-attribute]  # CBC waveforms only
                fixed_phase=phase_marginalization,
            )
        )

    for label in ("s1", "s2"):
        if {f"{label}_mag", f"{label}_theta", f"{label}_phi"} <= names:
            transforms.append(SphereSpinToCartesianSpinTransform(label))

    # Detector-frame sky or time in the prior: go back to ra/dec and t_c.
    if {"azimuth", "zenith"} <= names and not {"ra", "dec"} <= names:
        transforms.append(
            reverse_bijective_transform(
                SkyFrameToDetectorFrameSkyPositionTransform(ifos)
            )
        )

    if "t_det" in names and "t_c" not in names:
        transforms.append(
            reverse_bijective_transform(
                GeocentricArrivalTimeToDetectorArrivalTimeTransform(ifos[0])
            )
        )

    # Last: the spin-angle transform above still needs q.
    if "q" in names:
        transforms.append(MassRatioToSymmetricMassRatioTransform)

    return transforms


def get_default_periodic(
    prior: Prior,
    sample_transforms: Sequence[BijectiveTransform],
) -> dict[str, tuple[float, float]] | list[str]:
    """Sampling-space parameters that wrap around, in the form Jim's ``periodic`` takes.

    A parameter counts when it is an angle with a known period (``psi``: pi; ``ra``,
    ``azimuth``, ``phase_c``, ``phi_jl``, ``phi_12``, ``s1_phi``, ``s2_phi``: 2 pi) and
    its range covers one full period.

    Args:
        prior: The prior Jim will sample.
        sample_transforms: The sample transforms Jim will use.  If they map onto [0, 1]
            (``space="unit_cube"`` of ``get_default_sample_transforms``, for the NS AW
            sampler), the result is the list of names of the unit-cube parameters,
            whose bounds are implicit; otherwise it is a dict with bounds.

    Returns:
        ``{name: (lower, upper)}``, or a list of names for a unit-cube sampling space.
    """
    # The BoundToBound that maps a parameter onto [0, 1] knows its range.
    onto_unit_interval = [
        transform
        for transform in sample_transforms
        if isinstance(transform, BoundToBound)
        and bool(jnp.all(transform.target_lower_bound == 0.0))
        and bool(jnp.all(transform.target_upper_bound == 1.0))
    ]
    if onto_unit_interval:
        names: list[str] = []
        for transform in onto_unit_interval:
            for source, target, lower, upper in zip(
                transform.name_mapping[0],
                transform.name_mapping[1],
                transform.original_lower_bound,
                transform.original_upper_bound,
            ):
                if source in _PERIODS and bool(
                    jnp.isclose(upper - lower, _PERIODS[source])
                ):
                    names.append(target)
        return names

    sampling_names: tuple[str, ...] = tuple(prior.parameter_names)
    for transform in sample_transforms:
        sampling_names = transform.propagate_name(sampling_names)

    bounds: dict[str, tuple[float, float]] = {}
    for name, period in _PERIODS.items():
        if name not in sampling_names:
            continue
        component = find_specific_prior(prior, name)
        if component is None and name == "azimuth":
            # Made by the sky transform, in [0, 2 pi).
            bounds[name] = (0.0, period)
        elif isinstance(component, UniformPrior) and bool(
            jnp.isclose(component.xmax - component.xmin, period)
        ):
            bounds[name] = (float(component.xmin), float(component.xmax))
    return bounds


def to_likelihood_space(
    params: Mapping[str, float],
    ifos: Sequence[GroundBased2G],
    waveform: Waveform,
) -> dict[str, float]:
    """Convert parameters given in any supported parametrization to likelihood space.

    Handles azimuth/zenith to ra/dec, ``t_det`` to ``t_c`` (at ``ifos[0]``), J-frame
    spin angles and spherical spins to Cartesian spins, and ``q`` to ``eta``.  The
    detector-frame conversions read the trigger time from the data of ``ifos``.
    """
    p = {k: jnp.float64(v) for k, v in params.items()}

    # azimuth/zenith -> ra/dec (before the t_det conversion, which needs ra/dec)
    if {"azimuth", "zenith"} <= p.keys():
        t_sky = SkyFrameToDetectorFrameSkyPositionTransform(ifos)
        p = dict(t_sky.backward(p))

    if "t_det" in p:
        t_time = GeocentricArrivalTimeToDetectorArrivalTimeTransform(ifos[0])
        p = dict(t_time.backward(p))

    if _J_FRAME_SPIN_PARAMS <= p.keys():
        t = SpinAnglesToCartesianSpinTransform(
            freq_ref=waveform.f_ref  # ty: ignore[unresolved-attribute]  # CBC waveforms only
        )
        p = dict(t.forward(p))

    for label in ("s1", "s2"):
        if {f"{label}_mag", f"{label}_theta", f"{label}_phi"} <= p.keys():
            t = SphereSpinToCartesianSpinTransform(label)
            p = dict(t.forward(p))

    if "q" in p and "eta" not in p:
        p = dict(MassRatioToSymmetricMassRatioTransform.forward(p))

    return {k: float(v) for k, v in p.items()}
