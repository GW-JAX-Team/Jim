import logging
from typing import Optional

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
from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    GaussianPrior,
    PowerLawPrior,
    RayleighPrior,
    SinePrior,
    UniformPrior,
    UniformSpherePrior,
)

logger = logging.getLogger(__name__)


def build_prior(cfg: PriorConfig):
    """Build a CombinePrior from the dict-keyed prior config."""
    components = []
    for name, spec in cfg.root.items():
        if isinstance(spec, UniformSpec):
            components.append(UniformPrior(spec.min, spec.max, [name]))
        elif isinstance(spec, GaussianSpec):
            components.append(GaussianPrior(spec.loc, spec.scale, [name]))
        elif isinstance(spec, SineSpec):
            components.append(SinePrior([name]))
        elif isinstance(spec, CosineSpec):
            components.append(CosinePrior([name]))
        elif isinstance(spec, PowerLawSpec):
            components.append(PowerLawPrior(spec.min, spec.max, spec.alpha, [name]))
        elif isinstance(spec, RayleighSpec):
            components.append(RayleighPrior(spec.scale, [name]))
        elif isinstance(spec, UniformSphereSpec):
            components.append(UniformSpherePrior([name]))
        else:
            raise ValueError(f"Unknown prior spec type for '{name}': {type(spec)}")

    prior = CombinePrior(components)
    logger.info(
        "Built prior: %d parameter(s): %s",
        len(prior.parameter_names),
        prior.parameter_names,
    )
    return prior


def adapt_prior_for_ns_time(
    prior_cfg: PriorConfig,
    sampling_cfg: SamplingConfig,
) -> Optional[PriorConfig]:
    """For NS-AW: replace the time parameter so the unit-cube bounds are exact.

    Both ``t_c`` and ``t_det`` are treated as offsets from ``trigger_time``
    (e.g. ``min = -0.1``, ``max = 0.1``).  Because the ``t_c → t_det``
    transform shifts by a sky-position-dependent delay, the unit-cube bounds
    for the *output* parameter cannot be exact — the same bounds are used for
    both.

    NS-AW requires every sampling-space parameter to lie in [0, 1].
    Two cases require adaptation:

    1. **t_c in prior, time_frame != "geocentric"**: replace ``t_c`` with
       ``t_det`` using the same relative bounds:

           t_det ∈ [t_c.min, t_c.max]

    2. **t_det in prior, time_frame = "geocentric"**: replace ``t_det`` with
       ``t_c`` using the same relative bounds:

           t_c ∈ [t_det.min, t_det.max]

    Returns the modified ``PriorConfig``, or ``None`` if no adaptation is
    needed.
    """
    has_t_c = "t_c" in prior_cfg.root
    has_t_det = "t_det" in prior_cfg.root

    # Case 2: t_det in prior + geocentric sampling → adapt to t_c prior.
    if has_t_det and sampling_cfg.time_frame == "geocentric":
        t_det_spec = prior_cfg.root["t_det"]
        assert isinstance(t_det_spec, UniformSpec), (
            "NS-AW validation should ensure t_det is UniformSpec"
        )
        logger.warning(
            "NS-AW sampler: replacing t_det ~ Uniform(%.4f, %.4f) in [prior] with "
            "t_c ~ Uniform(%.4f, %.4f) (same relative-offset bounds). "
            "To sample t_det directly instead, remove [sampling] time_frame = 'geocentric'.",
            t_det_spec.min,
            t_det_spec.max,
            t_det_spec.min,
            t_det_spec.max,
        )
        new_root = {
            ("t_c" if k == "t_det" else k): v for k, v in prior_cfg.root.items()
        }
        return PriorConfig.model_validate(new_root)

    # No t_c in prior, or user already opted for geocentric t_c sampling → no change.
    if not has_t_c or sampling_cfg.time_frame == "geocentric":
        return None

    # Case 1: t_c in prior + detector time_frame → adapt to t_det prior.
    t_c_spec = prior_cfg.root["t_c"]
    assert isinstance(t_c_spec, UniformSpec), (
        "NS-AW validation should ensure t_c is UniformSpec"
    )
    logger.warning(
        "NS-AW sampler: replacing t_c ~ Uniform(%.4f, %.4f) in [prior] with "
        "t_det ~ Uniform(%.4f, %.4f) (same relative-offset bounds). "
        "To sample t_c directly instead, set [sampling] time_frame = 'geocentric'.",
        t_c_spec.min,
        t_c_spec.max,
        t_c_spec.min,
        t_c_spec.max,
    )

    # Rebuild the prior dict, preserving insertion order, substituting t_c → t_det.
    new_root = {("t_det" if k == "t_c" else k): v for k, v in prior_cfg.root.items()}
    return PriorConfig.model_validate(new_root)
