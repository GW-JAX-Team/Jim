"""Pydantic config models for the jim-run CLI pipeline.

All models are JAX-free so that ``jim-run --help`` starts in milliseconds.
Heavy imports (JAX, equinox, ripplegw) are deferred to the builder functions
called only after config validation.

Design intent: users specify *what* (prior bounds, waveform, sampler settings).
The CLI figures out *how* (transforms, parameter conversions, consistency checks).
"""

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

# SamplerConfig is safe to import here — samplers/config.py only uses numpy.
from jimgw.samplers.config import SamplerConfig


from jimgw.cli._utils import (
    CARTESIAN_SPIN_PARAMS as _CARTESIAN_SPIN_PARAMS,
    DETECTOR_SKY_PARAMS as _DETECTOR_SKY_PARAMS,
    EQUATORIAL_SKY_PARAMS as _EQUATORIAL_SKY_PARAMS,
    J_FRAME_SPIN_PARAMS as _J_FRAME_SPIN_PARAMS,
    SUPPORTED_DETECTORS as _SUPPORTED_DETECTORS,
)


# ---------------------------------------------------------------------------
# Data section
# ---------------------------------------------------------------------------


class _DataBase(BaseModel):
    model_config = {"extra": "forbid"}
    detectors: list[str]
    trigger_time: float

    @field_validator("detectors")
    @classmethod
    def _check_detectors(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("data.detectors must be a non-empty list")
        unknown = [d for d in v if d not in _SUPPORTED_DETECTORS]
        if unknown:
            raise ValueError(
                f"Unknown detector name(s): {unknown}. "
                f"Supported: {sorted(_SUPPORTED_DETECTORS)}"
            )
        if len(v) != len(set(v)):
            seen: set[str] = set()
            duplicates: list[str] = []
            for d in v:
                if d in seen:
                    duplicates.append(d)
                seen.add(d)
            raise ValueError(f"Duplicate detector name(s): {duplicates}")
        return v


class GWOSCDataConfig(_DataBase):
    """Fetch strain and PSD from GWOSC."""

    type: Literal["gwosc"] = "gwosc"
    duration: float
    post_trigger_duration: float = 2.0
    psd_duration: float


class InjectionDataConfig(_DataBase):
    """Synthetic injection into design-sensitivity noise."""

    type: Literal["injection"] = "injection"
    duration: float
    sampling_frequency: float
    injection_parameters: dict[str, float]
    zero_noise: bool = False


class FileDataConfig(_DataBase):
    """Load pre-saved strain and PSD from .npz files (useful for CI/offline use)."""

    type: Literal["file"] = "file"
    strain_files: dict[str, Path]  # detector_name -> .npz with 'strain', 'times'
    psd_files: dict[str, Path]  # detector_name -> .npz with 'psd', 'freqs'

    @model_validator(mode="after")
    def _check_all_detectors_have_files(self) -> "FileDataConfig":
        missing_strain = [d for d in self.detectors if d not in self.strain_files]
        missing_psd = [d for d in self.detectors if d not in self.psd_files]
        if missing_strain:
            raise ValueError(f"strain_files missing for: {missing_strain}")
        if missing_psd:
            raise ValueError(f"psd_files missing for: {missing_psd}")
        return self


DataConfig = Annotated[
    Union[GWOSCDataConfig, InjectionDataConfig, FileDataConfig],
    Discriminator("type"),
]


# ---------------------------------------------------------------------------
# Waveform section
# ---------------------------------------------------------------------------

Approximant = Literal[
    "IMRPhenomD",
    "IMRPhenomPv2",
    "TaylorF2",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomXAS",
    "IMRPhenomXAS_NRTidalv3",
    "IMRPhenomXHM",
    "IMRPhenomXPHM",
    "SineGaussian",
]


class WaveformConfig(BaseModel):
    model_config = {"extra": "forbid"}
    approximant: Approximant
    f_ref: float = 20.0


# ---------------------------------------------------------------------------
# Prior section
#
# Parameter names are the dict keys; each value is an inline table with a
# `type` discriminator plus type-specific bounds.
#
# Example TOML:
#   [prior]
#   M_c  = { type = "uniform",   min = 10.0, max = 80.0 }
#   iota = { type = "sine" }
#   d_L  = { type = "power_law", min = 1.0, max = 2000.0, alpha = 2.0 }
# ---------------------------------------------------------------------------


class UniformSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["uniform"] = "uniform"
    min: float
    max: float

    @model_validator(mode="after")
    def _check_bounds(self) -> "UniformSpec":
        if self.min >= self.max:
            raise ValueError(
                f"uniform prior requires min < max, got min={self.min}, max={self.max}"
            )
        return self


class GaussianSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["gaussian"] = "gaussian"
    loc: float
    scale: float

    @model_validator(mode="after")
    def _check_scale(self) -> "GaussianSpec":
        if self.scale <= 0:
            raise ValueError(
                f"gaussian prior requires scale > 0, got scale={self.scale}"
            )
        return self


class SineSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["sine"] = "sine"


class CosineSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["cosine"] = "cosine"


class PowerLawSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["power_law"] = "power_law"
    min: float
    max: float
    alpha: float

    @model_validator(mode="after")
    def _check_bounds(self) -> "PowerLawSpec":
        if self.min >= self.max:
            raise ValueError(
                f"power_law prior requires min < max, got min={self.min}, max={self.max}"
            )
        return self


class RayleighSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["rayleigh"] = "rayleigh"
    scale: float

    @model_validator(mode="after")
    def _check_scale(self) -> "RayleighSpec":
        if self.scale <= 0:
            raise ValueError(
                f"rayleigh prior requires scale > 0, got scale={self.scale}"
            )
        return self


class UniformSphereSpec(BaseModel):
    """Maps to UniformSpherePrior — generates three parameters: {name}_mag/theta/phi."""

    model_config = {"extra": "forbid"}
    type: Literal["uniform_sphere"] = "uniform_sphere"


PriorSpec = Annotated[
    Union[
        UniformSpec,
        GaussianSpec,
        SineSpec,
        CosineSpec,
        PowerLawSpec,
        RayleighSpec,
        UniformSphereSpec,
    ],
    Discriminator("type"),
]


class PriorConfig(RootModel[dict[str, PriorSpec]]):
    """Ordered dict of parameter_name → prior spec.

    Insertion order is preserved (Python 3.7+, TOML spec) and determines
    the parameter ordering passed to CombinePrior.
    """


# ---------------------------------------------------------------------------
# Sampling space section (optional)
# ---------------------------------------------------------------------------


class SamplingConfig(BaseModel):
    """Controls the coordinate system the sampler explores.

    Only relevant when the prior parametrization differs from the preferred
    sampling space. The CLI auto-infers transforms for every other case.
    """

    model_config = {"extra": "forbid"}

    time_frame: str = "detector"
    """Detector name to sample arrival time in (e.g. "H1"). Special values:
    - ``"detector"`` (default): use the first entry in data.detectors.
    - ``"geocentric"``: sample t_c directly without a time sample transform.
    Only used when ``t_c`` is in the prior."""

    sky_frame: Literal["detector", "geocentric"] = "detector"
    """Sampling space for sky position.
    - ``"detector"``: sample in azimuth/zenith (default, better mixing).
    - ``"geocentric"``: sample directly in ra/dec.
    Only used when ``ra``/``dec`` are in the prior."""


# ---------------------------------------------------------------------------
# Likelihood section
# ---------------------------------------------------------------------------
# CLI-level marg configs: structurally equivalent to the library versions but
# JAX-free. Converted to the real configs in the likelihood builder (Stage 6).


class CLIPhaseMargConfig(BaseModel):
    model_config = {"extra": "forbid"}


class CLITimeMargConfig(BaseModel):
    model_config = {"extra": "forbid"}
    tc_range: tuple[float, float] = (-0.1, 0.1)


class CLIDistanceMargConfig(BaseModel):
    """Distance marginalization config.

    ``distance_prior`` is a nested prior dict (same syntax as the top-level
    ``[prior]`` section) that is built into a ``Prior`` object by the builder.
    """

    model_config = {"extra": "forbid"}
    distance_prior: PriorConfig
    n_dist_points: int = 10000
    ref_dist: Optional[float] = None

    @model_validator(mode="after")
    def _check_single_distance_param(self) -> "CLIDistanceMargConfig":
        if len(self.distance_prior.root) != 1:
            raise ValueError(
                "distance_marginalization.distance_prior must contain exactly one parameter"
            )
        return self


class CLIOptimizerRefParams(BaseModel):
    """Find reference parameters automatically via CMA-ES (default)."""

    model_config = {"extra": "forbid"}
    type: Literal["optimizer"] = "optimizer"
    popsize: int = 500
    n_steps: int = 1000


class CLIProvidedRefParams(BaseModel):
    """Explicit likelihood-space reference parameters; skips CMA-ES."""

    model_config = {"extra": "forbid"}
    type: Literal["provided"] = "provided"
    values: dict[str, float]


class CLIInjectionRefParams(BaseModel):
    """Use injection parameters (converted to likelihood space) as reference.

    Only valid for injection runs (``data.type = "injection"``).
    """

    model_config = {"extra": "forbid"}
    type: Literal["injection"] = "injection"


HeterodynedRefParams = Annotated[
    Union[CLIOptimizerRefParams, CLIProvidedRefParams, CLIInjectionRefParams],
    Discriminator("type"),
]


class CLIHeterodynedConfig(BaseModel):
    """Enable the relative-binning (heterodyne) likelihood.

    When present, ``HeterodynedTransientLikelihoodFD`` is used instead of
    ``TransientLikelihoodFD``.  The ``reference_parameters`` sub-section
    selects how reference parameters are obtained:

    - ``type = "optimizer"`` (default): CMA-ES search using the prior.
    - ``type = "provided"``: explicit likelihood-space values (skips CMA-ES).
    - ``type = "injection"``: use ``data.injection_parameters`` (injection
      runs only).
    """

    model_config = {"extra": "forbid"}
    n_bins: int = 1000
    reference_parameters: HeterodynedRefParams = Field(
        default_factory=CLIOptimizerRefParams
    )


class LikelihoodConfig(BaseModel):
    model_config = {"extra": "forbid"}
    f_min: float
    f_max: float
    fixed_parameters: dict[str, float] = Field(default_factory=dict)
    phase_marginalization: bool = False
    time_marginalization: Optional[CLITimeMargConfig] = None
    distance_marginalization: Optional[CLIDistanceMargConfig] = None
    heterodyne: Optional[CLIHeterodynedConfig] = None

    @model_validator(mode="after")
    def _validate_marginalization_conflicts(self) -> "LikelihoodConfig":
        if self.heterodyne is not None:
            if self.time_marginalization is not None:
                raise ValueError(
                    "time_marginalization cannot be used with heterodyne likelihood"
                )
            if self.distance_marginalization is not None:
                raise ValueError(
                    "distance_marginalization cannot be used with heterodyne likelihood"
                )
        return self


# ---------------------------------------------------------------------------
# Output section
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    model_config = {"extra": "forbid"}
    dir: Path
    save_corner: bool = False
    n_samples: int = Field(
        default=0, description="Number of posterior samples to save. 0 = all."
    )
    overwrite: bool = False
    corner_parameters: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    model_config = {"extra": "forbid"}

    seed: int = 0
    data: DataConfig
    waveform: WaveformConfig
    prior: PriorConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    likelihood: LikelihoodConfig
    sampler: SamplerConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _validate_injection_frame_consistency(self) -> "PipelineConfig":
        if not isinstance(self.data, InjectionDataConfig):
            return self
        inj = set(self.data.injection_parameters)
        if "t_det" in inj and self.sampling.time_frame == "geocentric":
            raise ValueError(
                "injection_parameters uses 't_det' but [sampling].time_frame = "
                "'geocentric'. Use 't_c' in injection_parameters, or set "
                "time_frame to 'detector' or a specific detector name."
            )
        if inj & _DETECTOR_SKY_PARAMS and self.sampling.sky_frame != "detector":
            raise ValueError(
                "injection_parameters uses detector-frame sky position "
                "('azimuth'/'zenith') but [sampling].sky_frame != 'detector'. "
                "Use 'ra'/'dec' in injection_parameters, or set sky_frame = 'detector'."
            )
        if inj & _J_FRAME_SPIN_PARAMS and "phase_c" not in inj:
            raise ValueError(
                "injection_parameters uses J-frame spin angles but 'phase_c' is missing. "
                "SpinAnglesToCartesianSpinTransform requires 'phase_c' as a conditioning "
                "parameter. Add 'phase_c' to injection_parameters."
            )
        return self

    @model_validator(mode="after")
    def _validate_spin_parametrization(self) -> "PipelineConfig":
        prior_keys = frozenset(self.prior.root.keys())

        has_j_frame = bool(prior_keys & _J_FRAME_SPIN_PARAMS)
        has_sphere_spin = any(
            isinstance(self.prior.root.get(label), UniformSphereSpec)
            or all(f"{label}_{s}" in prior_keys for s in ("mag", "theta", "phi"))
            for label in ("s1", "s2")
        )
        has_cartesian_spin = bool(prior_keys & _CARTESIAN_SPIN_PARAMS)

        if sum([has_j_frame, has_sphere_spin, has_cartesian_spin]) > 1:
            raise ValueError(
                "Spin parametrizations are mutually exclusive. "
                "Found more than one of: J-frame angles, spherical per-spin, "
                f"Cartesian/aligned spins. Prior parameters: {sorted(prior_keys)}"
            )

        if has_j_frame:
            if "iota" in prior_keys:
                raise ValueError(
                    "J-frame spin angles produce 'iota' — 'iota' must not also appear in [prior]."
                )
            missing_j = _J_FRAME_SPIN_PARAMS - prior_keys
            if missing_j:
                raise ValueError(
                    "J-frame spin parametrization requires all 7 parameters; "
                    f"missing from [prior]: {sorted(missing_j)}"
                )
            missing_mass = {"M_c", "q"} - prior_keys
            if missing_mass:
                raise ValueError(
                    f"SpinAnglesToCartesianSpinTransform requires {missing_mass} in [prior] "
                    "as conditioning parameters."
                )

        return self

    @model_validator(mode="after")
    def _validate_sky_time_parametrization(self) -> "PipelineConfig":
        prior_keys = frozenset(self.prior.root.keys())

        equatorial_present = prior_keys & _EQUATORIAL_SKY_PARAMS
        if equatorial_present and not (_EQUATORIAL_SKY_PARAMS <= prior_keys):
            missing = _EQUATORIAL_SKY_PARAMS - prior_keys
            raise ValueError(
                f"[prior] must contain both 'ra' and 'dec' together; "
                f"missing: {sorted(missing)}"
            )
        detector_present = prior_keys & _DETECTOR_SKY_PARAMS
        if detector_present and not (_DETECTOR_SKY_PARAMS <= prior_keys):
            missing = _DETECTOR_SKY_PARAMS - prior_keys
            raise ValueError(
                f"[prior] must contain both 'azimuth' and 'zenith' together; "
                f"missing: {sorted(missing)}"
            )

        has_equatorial_sky = bool(equatorial_present)
        has_detector_sky = bool(detector_present)
        if has_equatorial_sky and has_detector_sky:
            raise ValueError(
                "Sky parametrizations are mutually exclusive: "
                "cannot have both ra/dec and azimuth/zenith in [prior]."
            )
        if has_detector_sky and self.sampling.sky_frame == "geocentric":
            raise ValueError(
                "azimuth/zenith are in [prior] but sky_frame='geocentric' requests "
                "equatorial-sky sampling. Either remove azimuth/zenith from [prior] "
                "and use ra/dec, or set sky_frame='detector'."
            )

        has_geocentric_time = "t_c" in prior_keys
        has_detector_time = "t_det" in prior_keys
        if has_geocentric_time and has_detector_time:
            raise ValueError(
                "Time parametrizations are mutually exclusive: "
                "cannot have both t_c and t_det in [prior]."
            )
        if has_detector_time and self.sampling.time_frame == "geocentric":
            raise ValueError(
                "t_det is in [prior] but time_frame='geocentric' requests geocentric-time "
                "sampling. Either remove t_det from [prior] and use t_c, or set "
                "time_frame to 'detector' or a specific detector name."
            )

        return self

    @model_validator(mode="after")
    def _validate_sampling_ifo_consistency(self) -> "PipelineConfig":
        if self.sampling.time_frame not in ("detector", "geocentric"):
            if self.sampling.time_frame not in self.data.detectors:
                raise ValueError(
                    f"[sampling] time_frame={self.sampling.time_frame!r} is not in "
                    f"data.detectors {self.data.detectors}"
                )

        prior_keys = frozenset(self.prior.root.keys())
        has_sky_params = bool(
            prior_keys & (_EQUATORIAL_SKY_PARAMS | _DETECTOR_SKY_PARAMS)
        )
        if (
            has_sky_params
            and self.sampling.sky_frame == "detector"
            and len(self.data.detectors) < 2
        ):
            raise ValueError(
                "Sky position sampling in detector frame requires at least 2 detectors; "
                f"got {self.data.detectors}"
            )

        return self

    @model_validator(mode="after")
    def _validate_ns_aw_constraints(self) -> "PipelineConfig":
        if self.sampler.type != "blackjax-ns-aw":
            return self

        prior_keys = frozenset(self.prior.root.keys())

        if "t_det" in prior_keys and not isinstance(
            self.prior.root["t_det"], UniformSpec
        ):
            raise ValueError(
                "NS-AW sampler: the 't_det' prior must be 'uniform' for automatic "
                "conversion to 't_c'. Either use a uniform t_det prior or replace "
                "'t_det' with 't_c' in [prior]."
            )

        if (
            "t_c" in prior_keys
            and self.sampling.time_frame != "geocentric"
            and not isinstance(self.prior.root["t_c"], UniformSpec)
        ):
            raise ValueError(
                "NS-AW sampler: the 't_c' prior must be 'uniform' for automatic "
                "conversion to 't_det'. Either use a uniform t_c prior, set "
                "[sampling] time_frame = 'geocentric' to sample t_c directly, or "
                "replace 't_c' with 't_det' in [prior]."
            )

        return self

    @model_validator(mode="after")
    def _validate_heterodyne_ref_data_type(self) -> "PipelineConfig":
        if (
            self.likelihood.heterodyne is not None
            and isinstance(
                self.likelihood.heterodyne.reference_parameters, CLIInjectionRefParams
            )
            and not isinstance(self.data, InjectionDataConfig)
        ):
            raise ValueError(
                "heterodyne.reference_parameters.type = 'injection' requires "
                "data.type = 'injection'"
            )
        return self
