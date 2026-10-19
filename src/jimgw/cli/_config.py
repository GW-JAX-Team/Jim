"""Pydantic config models for the jim-run CLI pipeline.

All models are JAX-free so that ``jim-run --help`` starts in milliseconds.
Heavy imports (JAX, equinox, ripplegw) are deferred to the builder functions
called only after config validation.

Design intent: users specify *what* (prior bounds, waveform, sampler settings).
The CLI figures out *how* (transforms, parameter conversions, consistency checks).
"""

import logging
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import numpy as np
from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from jimgw.cli._utils import (
    CARTESIAN_SPIN_PARAMS,
    DETECTOR_SKY_PARAMS,
    EQUATORIAL_SKY_PARAMS,
    J_FRAME_SPIN_PARAMS,
    PRESET_DETECTOR_NAMES,
)

# SamplerConfig is safe to import here — samplers/config.py only uses numpy.
from jimgw.samplers.config import SamplerConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data section
# ----------------------------------------------------------------------


class _DataBase(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    detectors: list[str]
    trigger_time: float = Field(allow_inf_nan=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @field_validator("detectors")
    @classmethod
    def _check_detectors(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("data.detectors must be a non-empty list")
        unknown = [d for d in v if d not in PRESET_DETECTOR_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown detector name(s): {unknown}. "
                f"Supported: {sorted(PRESET_DETECTOR_NAMES)}"
            )
        if len(v) != len(set(v)):
            duplicates = [d for d in set(v) if v.count(d) > 1]
            raise ValueError(f"Duplicate detector name(s): {duplicates}")
        return v


class GWOSCDataConfig(_DataBase):
    """Fetch strain and PSD from GWOSC."""

    type: Literal["gwosc"] = "gwosc"
    duration: float = Field(gt=0.0, allow_inf_nan=False)
    post_trigger_duration: float = Field(default=2.0, ge=0.0, allow_inf_nan=False)
    psd_duration: float = Field(gt=0.0, allow_inf_nan=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_trigger_inside_segment(self) -> "GWOSCDataConfig":
        unavailable = sorted(set(self.detectors) - {"H1", "L1", "V1"})
        if unavailable:
            raise ValueError(
                f"GWOSC strain is not available for detector(s) {unavailable}; "
                "choose H1, L1 or V1, or use injection/file data."
            )
        if self.post_trigger_duration > self.duration:
            raise ValueError("post_trigger_duration must not exceed duration")
        return self


# Formats read by ``PowerSpectrum.from_file``.
_PSD_FILE_SUFFIXES = (".npz", ".txt", ".dat", ".csv")


class _PSDSourceBase(_DataBase):
    """PSD or ASD files, shared by the data types that read one from disk.

    ``psd_files`` holds PSD values and ``asd_files`` holds ASD values (squared on
    load); they are mutually exclusive.  Each maps every detector to a file in any
    format ``PowerSpectrum.from_file`` reads, with ``ET`` and ``ET_2L`` named by
    their individual detectors (``ET1``-``ET3``, ``ET_Sar`` and ``ET_EMR``).
    """

    psd_files: Optional[dict[str, Path]] = None  # detector_name -> PSD file
    asd_files: Optional[dict[str, Path]] = None  # detector_name -> ASD file

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_psd_source(self) -> "_PSDSourceBase":
        if self.psd_files is not None and self.asd_files is not None:
            raise ValueError(
                "psd_files and asd_files are mutually exclusive; set at most one."
            )
        names = [name for d in self.detectors for name in PRESET_DETECTOR_NAMES[d]]
        for field, table in (
            ("psd_files", self.psd_files),
            ("asd_files", self.asd_files),
        ):
            if table is None:
                continue
            missing = [name for name in names if name not in table]
            if missing:
                raise ValueError(f"{field} missing for: {missing}")
            unknown = sorted(set(table) - set(names))
            if unknown:
                raise ValueError(f"{field} has unknown detector names: {unknown}")
            for detector, path in table.items():
                if path.suffix.lower() not in _PSD_FILE_SUFFIXES:
                    raise ValueError(
                        f"{field}[{detector!r}]: expected a "
                        f"{'/'.join(_PSD_FILE_SUFFIXES)} file, got {str(path)!r}"
                    )
        return self


class InjectionDataConfig(_PSDSourceBase):
    """Synthetic injection into Gaussian noise drawn from the detector PSD."""

    type: Literal["injection"] = "injection"
    duration: float = Field(gt=0.0, allow_inf_nan=False)
    sampling_frequency: float = Field(gt=0.0, allow_inf_nan=False)
    injection_parameters: dict[str, float]
    zero_noise: bool = False
    noise_seed: Optional[int] = Field(default=None, strict=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_noise_seed(self) -> "InjectionDataConfig":
        if self.zero_noise and self.noise_seed is not None:
            logger.warning(
                "noise_seed is ignored because zero_noise = true: no noise is drawn."
            )
        if not self.zero_noise and self.noise_seed is None:
            raise ValueError("noise_seed is required when zero_noise = false.")
        return self


class FileDataConfig(_PSDSourceBase):
    """Load pre-saved strain and PSD from local files (useful for CI/offline use).

    Supported strain formats: ``.npz``, ``.gwf`` / ``.gwf.gz``, ``.hdf5`` / ``.h5``,
    ``.csv``.  Exactly one of ``psd_files`` and ``asd_files`` is required.

    For frame (``.gwf``) and HDF5 files a channel name is required.  Provide
    ``strain_channels`` to map detector names to channel strings; if omitted,
    common LIGO/Virgo preset channel names are tried automatically for GWF files.

    With ``detectors = ["ET"]``, strain files, channels and PSD/ASD files use the
    individual detector names ``ET1``, ``ET2`` and ``ET3``; with ``["ET_2L"]``,
    ``ET_Sar`` and ``ET_EMR``.
    """

    type: Literal["file"] = "file"
    strain_files: dict[str, Path]  # detector_name -> strain file path
    strain_channels: dict[str, str] = Field(
        default_factory=dict
    )  # detector_name -> channel (e.g. "H1:GDS-CALIB_STRAIN")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_strain_and_psd_files(self) -> "FileDataConfig":
        strain_detectors = {
            name
            for detector in self.detectors
            for name in PRESET_DETECTOR_NAMES[detector]
        }
        missing_strain = sorted(strain_detectors - set(self.strain_files))
        if missing_strain:
            raise ValueError(f"strain_files missing for: {missing_strain}")
        for field, names in (
            ("strain_files", self.strain_files),
            ("strain_channels", self.strain_channels),
        ):
            unknown = sorted(set(names) - strain_detectors)
            if unknown:
                raise ValueError(f"{field} has unknown detector names: {unknown}")
        if self.psd_files is None and self.asd_files is None:
            raise ValueError("provide psd_files or asd_files (exactly one of the two)")
        return self


DataConfig = Annotated[
    Union[GWOSCDataConfig, InjectionDataConfig, FileDataConfig],
    Discriminator("type"),
]


# ----------------------------------------------------------------------
# Waveform section
# ----------------------------------------------------------------------

Approximant = Literal[
    "TaylorF2",
    "IMRPhenomD",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomHM",
    "IMRPhenomPv2",
    "IMRPhenomXAS",
    "IMRPhenomXAS_NRTidalv3",
    "IMRPhenomXHM",
    "IMRPhenomXP",
    "IMRPhenomXP_NRTidalv3",
    "IMRPhenomXPHM",
]


class WaveformConfig(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    approximant: Approximant
    f_ref: float = Field(default=20.0, gt=0.0)


# ----------------------------------------------------------------------
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
# ----------------------------------------------------------------------
class UniformSpec(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    type: Literal["uniform"] = "uniform"
    min: float
    max: float

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_bounds(self) -> "UniformSpec":
        if self.min >= self.max:
            raise ValueError(
                f"uniform prior requires min < max, got min={self.min}, max={self.max}"
            )
        return self


class GaussianSpec(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    type: Literal["gaussian"] = "gaussian"
    loc: float
    scale: float

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_scale(self) -> "GaussianSpec":
        if self.scale <= 0:
            raise ValueError(
                f"gaussian prior requires scale > 0, got scale={self.scale}"
            )
        return self


class SineSpec(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    type: Literal["sine"] = "sine"


class CosineSpec(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    type: Literal["cosine"] = "cosine"


class PowerLawSpec(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    type: Literal["power_law"] = "power_law"
    min: float
    max: float
    alpha: float

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_bounds(self) -> "PowerLawSpec":
        if self.min <= 0:
            raise ValueError("power_law prior requires min > 0")
        if self.min >= self.max:
            raise ValueError(
                f"power_law prior requires min < max, got min={self.min}, max={self.max}"
            )
        return self


class RayleighSpec(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    type: Literal["rayleigh"] = "rayleigh"
    scale: float

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_scale(self) -> "RayleighSpec":
        if self.scale <= 0:
            raise ValueError(
                f"rayleigh prior requires scale > 0, got scale={self.scale}"
            )
        return self


class UniformSphereSpec(BaseModel):
    """Maps to UniformSpherePrior — generates three parameters: {name}_mag/theta/phi."""

    model_config = {"extra": "forbid", "allow_inf_nan": False}
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


# ----------------------------------------------------------------------
# Sampling space section (optional)
# ----------------------------------------------------------------------


class SamplingConfig(BaseModel):
    """Controls the coordinate system the sampler explores.

    Only relevant when the prior parametrization differs from the preferred
    sampling space. The CLI auto-infers transforms for every other case.
    """

    model_config = {"extra": "forbid", "allow_inf_nan": False}

    detector_frame_time: bool = True
    """Sample the arrival time at the first detector in data.detectors (default)
    instead of the geocentric ``t_c``.  Only used when ``t_c`` is in the prior."""

    detector_frame_sky: bool = True
    """Sample the sky position as azimuth/zenith relative to the detector baseline
    (default, better mixing) instead of ra/dec.  Only used when ``ra``/``dec`` are
    in the prior."""


# ----------------------------------------------------------------------
# Likelihood section
# ----------------------------------------------------------------------

# CLI-level marg configs: structurally equivalent to the library versions but
# JAX-free. Converted to the real configs in the likelihood builder (Stage 6).


class CLIPhaseMargConfig(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}


class CLITimeMargConfig(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    tc_range: tuple[float, float] = (-0.1, 0.1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_time_range(self) -> "CLITimeMargConfig":
        if not self.tc_range[0] < self.tc_range[1]:
            raise ValueError("tc_range must contain finite, increasing bounds.")
        return self


class CLIDistanceMargConfig(BaseModel):
    """Distance marginalization config.

    ``distance_prior`` is a nested prior dict (same syntax as the top-level
    ``[prior]`` section) that is built into a ``Prior`` object by the builder.
    """

    model_config = {"extra": "forbid", "allow_inf_nan": False}
    distance_prior: PriorConfig
    n_dist_points: int = Field(default=10000, ge=2, strict=True)
    ref_dist: Optional[float] = Field(default=None, gt=0.0)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_single_distance_param(self) -> "CLIDistanceMargConfig":
        if set(self.distance_prior.root) != {"d_L"}:
            raise ValueError(
                "distance_marginalization.distance_prior must contain exactly one "
                "parameter, 'd_L'"
            )
        return self


class CLIOptimizerRefParams(BaseModel):
    """Find reference parameters automatically via CMA-ES (default)."""

    model_config = {"extra": "forbid", "allow_inf_nan": False}
    type: Literal["optimizer"] = "optimizer"
    popsize: int = Field(default=500, ge=1, strict=True)
    n_steps: int = Field(default=1000, ge=1, strict=True)
    target: Optional[float] = None


class CLIProvidedRefParams(BaseModel):
    """Explicit likelihood-space reference parameters; skips CMA-ES."""

    model_config = {"extra": "forbid", "allow_inf_nan": False}
    type: Literal["provided"] = "provided"
    values: dict[str, float]


class CLIInjectionRefParams(BaseModel):
    """Use injection parameters (converted to likelihood space) as reference.

    Only valid for injection runs (``data.type = "injection"``).
    """

    model_config = {"extra": "forbid", "allow_inf_nan": False}
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

    Binning is controlled by at most one of ``epsilon`` or ``n_bins``
    (mutually exclusive).  When neither is set, ``epsilon=0.5`` (rad per
    bin) is used as the default.
    """

    model_config = {"extra": "forbid", "allow_inf_nan": False}
    n_bins: Optional[int] = Field(default=None, ge=1, strict=True)
    epsilon: Optional[float] = Field(default=None, gt=0.0)
    reference_parameters: HeterodynedRefParams = Field(
        default_factory=CLIOptimizerRefParams
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_binning(self) -> "CLIHeterodynedConfig":
        if self.n_bins is not None and self.epsilon is not None:
            raise ValueError("n_bins and epsilon are mutually exclusive")
        return self


class CLIMultibandedConfig(BaseModel):
    """Enable the multi-banded likelihood.

    When present, ``MultibandedTransientLikelihoodFD`` is used instead of
    ``TransientLikelihoodFD``.

    ``reference_chirp_mass``, ``time_offset``, and ``delta_f_end`` are all
    optional: when omitted they are inferred automatically from the prior
    (``M_c`` minimum and ``t_c`` range respectively).  You only need to set
    them explicitly to override the inferred values.
    """

    model_config = {"extra": "forbid", "allow_inf_nan": False}
    reference_chirp_mass: Optional[float] = Field(default=None, gt=0.0)
    highest_mode: int = Field(default=2, ge=1, strict=True)
    accuracy_factor: float = Field(default=5.0, gt=0.0)
    time_offset: Optional[float] = Field(default=None, ge=0.0)
    delta_f_end: Optional[float] = Field(default=None, gt=0.0)
    max_banding_frequency: Optional[float] = Field(default=None, gt=0.0)
    min_banding_duration: float = Field(default=0.0, ge=0.0)


class LikelihoodConfig(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    f_min: float = Field(gt=0.0)
    f_max: float = Field(gt=0.0, allow_inf_nan=True)
    fixed_parameters: dict[str, float] = Field(default_factory=dict)
    phase_marginalization: bool = False
    time_marginalization: Optional[CLITimeMargConfig] = None
    distance_marginalization: Optional[CLIDistanceMargConfig] = None
    heterodyne: Optional[CLIHeterodynedConfig] = None
    multiband: Optional[CLIMultibandedConfig] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_marginalization_conflicts(self) -> "LikelihoodConfig":
        if not self.f_min < self.f_max:
            raise ValueError("f_min must be below f_max")
        for name, enabled in (
            ("phase_c", self.phase_marginalization),
            ("t_c", self.time_marginalization is not None),
            ("d_L", self.distance_marginalization is not None),
        ):
            if enabled and name in self.fixed_parameters:
                raise ValueError(
                    f"Cannot have {name} fixed while marginalizing over it"
                )
        if (
            self.time_marginalization is not None
            and self.distance_marginalization is not None
        ):
            raise ValueError(
                "Joint time + distance marginalization is not yet supported"
            )
        if self.heterodyne is not None and self.multiband is not None:
            raise ValueError("heterodyne and multiband cannot both be set")
        if self.heterodyne is not None:
            if self.time_marginalization is not None:
                raise ValueError(
                    "time_marginalization cannot be used with heterodyne likelihood"
                )
            if self.distance_marginalization is not None:
                raise ValueError(
                    "distance_marginalization cannot be used with heterodyne likelihood"
                )
        if self.multiband is not None:
            if self.time_marginalization is not None:
                raise ValueError(
                    "time_marginalization cannot be used with multiband likelihood"
                )
            if self.distance_marginalization is not None:
                raise ValueError(
                    "distance_marginalization cannot be used with multiband likelihood"
                )
            if self.phase_marginalization:
                raise ValueError(
                    "phase_marginalization cannot be used with multiband likelihood"
                )
        return self


# ----------------------------------------------------------------------
# Output section
# ----------------------------------------------------------------------


class OutputConfig(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    dir: Path
    save_corner: bool = False
    n_samples: int = Field(
        default=10000,
        ge=0,
        strict=True,
        description="Number of posterior samples to save. 0 = all.",
    )
    overwrite: bool = False
    corner_parameters: Optional[list[str]] = None


# ----------------------------------------------------------------------
# Top-level pipeline config
# ----------------------------------------------------------------------


class PipelineConfig(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}

    seed: int = Field(default=0, strict=True)
    data: DataConfig
    waveform: WaveformConfig
    prior: PriorConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    likelihood: LikelihoodConfig
    sampler: SamplerConfig
    output: OutputConfig

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_injection_frame_consistency(self) -> "PipelineConfig":
        if not isinstance(self.data, InjectionDataConfig):
            return self
        inj = set(self.data.injection_parameters)
        if "t_det" in inj and not self.sampling.detector_frame_time:
            raise ValueError(
                "injection_parameters uses 't_det' but [sampling].detector_frame_time = "
                "false. Use 't_c' in injection_parameters, or set "
                "detector_frame_time = true."
            )
        if inj & DETECTOR_SKY_PARAMS and not self.sampling.detector_frame_sky:
            raise ValueError(
                "injection_parameters uses detector-frame sky position "
                "('azimuth'/'zenith') but [sampling].detector_frame_sky = false. "
                "Use 'ra'/'dec' in injection_parameters, or set "
                "detector_frame_sky = true."
            )
        if inj & J_FRAME_SPIN_PARAMS and "phase_c" not in inj:
            raise ValueError(
                "injection_parameters uses J-frame spin angles but 'phase_c' is missing. "
                "SpinAnglesToCartesianSpinTransform requires 'phase_c' as a conditioning "
                "parameter. Add 'phase_c' to injection_parameters."
            )
        return self

    @model_validator(mode="after")
    def _validate_spin_parametrization(self) -> "PipelineConfig":
        prior_keys = frozenset(self.prior.root.keys())

        has_j_frame = bool(prior_keys & J_FRAME_SPIN_PARAMS)
        has_sphere_spin = any(
            isinstance(self.prior.root.get(label), UniformSphereSpec)
            or all(f"{label}_{s}" in prior_keys for s in ("mag", "theta", "phi"))
            for label in ("s1", "s2")
        )
        has_cartesian_spin = bool(prior_keys & CARTESIAN_SPIN_PARAMS)

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
            missing_j = J_FRAME_SPIN_PARAMS - prior_keys
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

        equatorial_present = prior_keys & EQUATORIAL_SKY_PARAMS
        if equatorial_present and not (EQUATORIAL_SKY_PARAMS <= prior_keys):
            missing = EQUATORIAL_SKY_PARAMS - prior_keys
            raise ValueError(
                f"[prior] must contain both 'ra' and 'dec' together; "
                f"missing: {sorted(missing)}"
            )
        detector_present = prior_keys & DETECTOR_SKY_PARAMS
        if detector_present and not (DETECTOR_SKY_PARAMS <= prior_keys):
            missing = DETECTOR_SKY_PARAMS - prior_keys
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
        if has_detector_sky and not self.sampling.detector_frame_sky:
            raise ValueError(
                "azimuth/zenith are in [prior] but detector_frame_sky = false requests "
                "equatorial-sky sampling. Either remove azimuth/zenith from [prior] "
                "and use ra/dec, or set detector_frame_sky = true."
            )

        has_geocentric_time = "t_c" in prior_keys
        has_detector_time = "t_det" in prior_keys
        if has_geocentric_time and has_detector_time:
            raise ValueError(
                "Time parametrizations are mutually exclusive: "
                "cannot have both t_c and t_det in [prior]."
            )
        if has_detector_time and not self.sampling.detector_frame_time:
            raise ValueError(
                "t_det is in [prior] but detector_frame_time = false requests "
                "geocentric-time sampling. Either remove t_det from [prior] and use "
                "t_c, or set detector_frame_time = true."
            )

        return self

    @model_validator(mode="after")
    def _validate_sampling_ifo_consistency(self) -> "PipelineConfig":
        prior_keys = frozenset(self.prior.root.keys())
        has_sky_params = bool(
            prior_keys & (EQUATORIAL_SKY_PARAMS | DETECTOR_SKY_PARAMS)
        )
        n_detectors = sum(len(PRESET_DETECTOR_NAMES[d]) for d in self.data.detectors)
        if has_sky_params and self.sampling.detector_frame_sky and n_detectors < 2:
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
                "NS AW sampler: the 't_det' prior must be 'uniform' for automatic "
                "conversion to 't_c'. Either use a uniform t_det prior or replace "
                "'t_det' with 't_c' in [prior]."
            )

        if (
            "t_c" in prior_keys
            and self.sampling.detector_frame_time
            and not isinstance(self.prior.root["t_c"], UniformSpec)
        ):
            raise ValueError(
                "NS AW sampler: the 't_c' prior must be 'uniform' for automatic "
                "conversion to 't_det'. Either use a uniform t_c prior, set "
                "[sampling] detector_frame_time = false to sample t_c directly, or "
                "replace 't_c' with 't_det' in [prior]."
            )

        if self.sampling.detector_frame_sky and EQUATORIAL_SKY_PARAMS <= prior_keys:
            ra, dec = self.prior.root["ra"], self.prior.root["dec"]
            isotropic = (
                isinstance(ra, UniformSpec)
                and np.isclose(ra.max - ra.min, 2 * np.pi)
                and isinstance(dec, CosineSpec)
            )
            if not isotropic:
                raise ValueError(
                    "NS AW sampler: with [sampling] detector_frame_sky = true, 'ra'/'dec' "
                    "in [prior] must be isotropic ('ra' uniform over one period, 'dec' "
                    "cosine). Either use such a prior, put the prior on 'azimuth'/'zenith' "
                    "instead, or set [sampling] detector_frame_sky = false to map "
                    "ra/dec directly."
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
