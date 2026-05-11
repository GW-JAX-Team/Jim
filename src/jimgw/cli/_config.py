"""Pydantic config models for the jim-run CLI pipeline.

All models are JAX-free so that ``jim-run --help`` starts in milliseconds.
Heavy imports (JAX, equinox, ripplegw) are deferred to the builder functions
called only after config validation.

Design intent: users specify *what* (prior bounds, waveform, sampler settings).
The CLI figures out *how* (transforms, parameter conversions, consistency checks).
"""

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Discriminator, Field, RootModel, model_validator

# SamplerConfig is safe to import here — samplers/config.py only uses numpy.
from jimgw.samplers.config import SamplerConfig


# ---------------------------------------------------------------------------
# Data section
# ---------------------------------------------------------------------------


class _DataBase(BaseModel):
    model_config = {"extra": "forbid"}
    ifos: list[str]
    trigger_time: float


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
    strain_files: dict[str, Path]  # ifo_name -> .npz with 'strain', 'times'
    psd_files: dict[str, Path]  # ifo_name -> .npz with 'psd', 'freqs'

    @model_validator(mode="after")
    def _check_all_ifos_have_files(self) -> "FileDataConfig":
        missing_strain = [ifo for ifo in self.ifos if ifo not in self.strain_files]
        missing_psd = [ifo for ifo in self.ifos if ifo not in self.psd_files]
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


class GaussianSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["gaussian"] = "gaussian"
    loc: float
    scale: float


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


class RayleighSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["rayleigh"] = "rayleigh"
    scale: float


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
    """IFO name to sample arrival time in (e.g. "H1"). Special values:
    - ``"detector"`` (default): use the first entry in data.ifos.
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
