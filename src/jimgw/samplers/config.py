"""Pydantic configuration models for Jim's samplers.

Each sampler has its own ``*Config`` class discriminated by a ``type`` literal;
`SamplerConfig` is the discriminated-union annotation a caller passes to
``Jim(..., sampler_config=...)``.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Discriminator, Field, field_validator, model_validator


class BaseSamplerConfig(BaseModel):
    """Fields shared by all sampler configs."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    verbose: bool = False


# ---------------------------------------------------------------------------
# flowMC sub-configs
# ---------------------------------------------------------------------------


class ParallelTemperingConfig(BaseModel):
    """Parallel-tempering settings for the flowMC backend.

    Construct directly or pass a plain ``dict`` to ``FlowMCConfig.parallel_tempering``.
    Use ``True`` to enable with all defaults, ``False`` / ``None`` to disable.
    """

    model_config = {"extra": "forbid"}

    n_temperatures: int = 5
    max_temperature: float = 10.0
    n_tempered_steps: int = 5


class MALAConfig(BaseModel):
    """MALA local-kernel settings for the flowMC backend.

    ``step_size`` may be a scalar ``float`` (applied uniformly to all
    dimensions) or a 1-D ``np.ndarray`` of length ``n_dims`` for
    per-dimension step sizes.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    step_size: float | np.ndarray = 2e-3


class HMCConfig(BaseModel):
    """HMC local-kernel settings for the flowMC backend.

    ``step_size`` may be a scalar ``float`` (applied uniformly) or a 1-D
    ``np.ndarray`` of length ``n_dims`` for per-dimension sizes.
    ``condition_matrix`` may also be a scalar or 1-D array and is used
    as the mass matrix / preconditioning for HMC leapfrog steps.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    step_size: float = 2e-3
    condition_matrix: float | np.ndarray = 1.0
    n_leapfrog_steps: int = 10


class GRWConfig(BaseModel):
    """Gaussian random-walk local-kernel settings for the flowMC backend.

    ``step_size`` may be a scalar ``float`` (applied uniformly to all
    dimensions) or a 1-D ``np.ndarray`` of length ``n_dims`` for
    per-dimension step sizes.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    step_size: float | np.ndarray = 2e-3


class FlowMCConfig(BaseSamplerConfig):
    """Configuration for [`FlowMCSampler`][jimgw.samplers.flowmc.FlowMCSampler].

    The ``local_kernel`` field selects the MCMC kernel used for local proposals:

    * ``"MALA"`` — Metropolis-Adjusted Langevin; default.
    * ``"HMC"`` — Hamiltonian Monte Carlo.
    * ``"GRW"`` — Gaussian random walk.

    Parallel tempering is **off by default**.  To enable, pass a
    [`ParallelTemperingConfig`][jimgw.samplers.config.ParallelTemperingConfig],
    a ``dict`` of its fields, or simply ``True`` (uses all defaults).
    ``False`` disables it.

    !!! note
        Only the sub-config matching the active ``local_kernel`` is used.
        Non-default values in inactive sub-configs emit a `UserWarning`.

    !!! note
        Periodic parameters are **not** configured here.  Pass a ``periodic``
        argument to [`Jim`][jimgw.core.jim.Jim] instead; Jim resolves
        parameter names to dimension indices and passes them to the sampler.
    """

    type: Literal["flowmc"] = "flowmc"

    n_chains: int = 1000
    n_local_steps: int = 100
    n_global_steps: int = 1000
    n_training_loops: int = 20
    n_production_loops: int = 10
    n_epochs: int = 20

    local_kernel: Literal["MALA", "HMC", "GRW"] = "MALA"
    parallel_tempering: Optional[ParallelTemperingConfig] = None
    mala: MALAConfig = Field(default_factory=MALAConfig)
    hmc: HMCConfig = Field(default_factory=HMCConfig)
    grw: GRWConfig = Field(default_factory=GRWConfig)

    rq_spline_hidden_units: list[int] = Field(default_factory=lambda: [128, 128])
    rq_spline_n_bins: int = 10
    rq_spline_n_layers: int = 8
    n_NFproposal_batch_size: int = 1000

    learning_rate: float = 1e-3
    batch_size: int = 10000
    n_max_examples: int = 30000
    history_window: int = 100

    chain_batch_size: int = 0
    local_thinning: int = 1
    global_thinning: int = 1

    early_stopping: bool = True
    early_stopping_tolerance: float = 0.1
    early_stopping_patience: int = 3
    early_stopping_min_acceptance: float = 0.1

    @field_validator("parallel_tempering", mode="before")
    @classmethod
    def _coerce_parallel_tempering(cls, v: object) -> Optional[ParallelTemperingConfig]:
        if v is None or v is False:
            return None
        if v is True:
            return ParallelTemperingConfig()
        if isinstance(v, dict):
            return ParallelTemperingConfig(**v)
        if isinstance(v, ParallelTemperingConfig):
            return v
        raise ValueError(
            "parallel_tempering must be None, False, True, a dict of ParallelTemperingConfig "
            f"fields, or a ParallelTemperingConfig instance; got {type(v).__name__}."
        )

    @model_validator(mode="after")
    def _warn_if_irrelevant_kernel_set(self) -> FlowMCConfig:
        active = self.local_kernel
        for name in ("MALA", "HMC", "GRW"):
            if name == active:
                continue
            sub_config = getattr(self, name.lower())
            if sub_config.model_fields_set:
                warnings.warn(
                    f"FlowMCConfig: `{name.lower()}` sub-config has non-default "
                    f"values but `local_kernel='{active}'` — the `{name.lower()}` "
                    f"settings will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
        return self


class BlackJAXNSAWConfig(BaseSamplerConfig):
    """Configuration for the BlackJAX acceptance-walk nested sampler.

    !!! note
        This sampler requires the sampling space to be the unit hypercube
        ``[0, 1]^n_dims``.  When using Jim, this means all
        ``sample_transforms`` must map the prior support onto the unit cube.

    !!! note
        Periodic parameters are **not** configured here.  Pass a
        ``periodic`` argument to [`Jim`][jimgw.core.jim.Jim] instead.
        For NS-AW, bounds are implicit as ``[0, 1]``; just list the
        parameter names.
    """

    type: Literal["blackjax-ns-aw"] = "blackjax-ns-aw"

    n_live: int = 1000
    n_delete_frac: float = 0.5
    n_target: int = 60
    max_mcmc: int = 5000
    max_proposals: int = 1000
    termination_dlogz: float = 0.1

    @field_validator("n_delete_frac")
    @classmethod
    def _n_delete_frac_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("n_delete_frac must be strictly between 0 and 1")
        return v

    @model_validator(mode="after")
    def _n_live_n_delete_consistency(self) -> "BlackJAXNSAWConfig":
        if self.n_live < 2:
            raise ValueError(f"n_live must be >= 2 (got {self.n_live}).")
        n_delete = int(self.n_live * self.n_delete_frac)
        if n_delete < 1:
            raise ValueError(
                f"n_live * n_delete_frac = {self.n_live * self.n_delete_frac} "
                f"yields n_delete = {n_delete}; require n_delete >= 1. "
                "Increase n_live or n_delete_frac."
            )
        return self


class BlackJAXNSSConfig(BaseSamplerConfig):
    """Configuration for the BlackJAX nested slice sampler.

    !!! note
        Periodic parameters are **not** configured here.  Pass a ``periodic``
        argument to [`Jim`][jimgw.core.jim.Jim] instead.
    """

    type: Literal["blackjax-nss"] = "blackjax-nss"

    n_live: int = 1000
    n_delete_frac: float = 0.5
    num_inner_steps_per_dim: int = 10
    termination_dlogz: float = 0.1

    @field_validator("n_delete_frac")
    @classmethod
    def _n_delete_frac_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("n_delete_frac must be strictly between 0 and 1")
        return v

    @model_validator(mode="after")
    def _n_live_n_delete_consistency(self) -> "BlackJAXNSSConfig":
        if self.n_live < 2:
            raise ValueError(f"n_live must be >= 2 (got {self.n_live}).")
        n_delete = int(self.n_live * self.n_delete_frac)
        if n_delete < 1:
            raise ValueError(
                f"n_live * n_delete_frac = {self.n_live * self.n_delete_frac} "
                f"yields n_delete = {n_delete}; require n_delete >= 1. "
                "Increase n_live or n_delete_frac."
            )
        return self


class BlackJAXSMCConfig(BaseSamplerConfig):
    """Configuration for the BlackJAX SMC sampler.

    !!! note
        Periodic parameters are **not** configured here.  Pass a ``periodic``
        argument to [`Jim`][jimgw.core.jim.Jim] instead.
    """

    type: Literal["blackjax-smc"] = "blackjax-smc"

    n_particles: int = 2000
    n_mcmc_steps_per_dim: int = 100
    target_ess: Optional[int] = None
    target_ess_fraction: Optional[float] = None
    initial_cov_scale: float = 0.5
    target_acceptance_rate: float = 0.234
    scale_adaptation_gain: float = 3.0

    persistent_sampling: bool = True
    temperature_ladder: Optional[list[float]] = None

    @field_validator("temperature_ladder")
    @classmethod
    def _validate_temperature_ladder(
        cls, v: Optional[list[float]]
    ) -> Optional[list[float]]:
        if v is None:
            return v
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError(
                "temperature_ladder must be a 1-D sequence of length >= 2."
            )
        if not np.all(np.diff(arr) > 0):
            raise ValueError("temperature_ladder must be strictly increasing.")
        if arr[0] != 0.0 or arr[-1] != 1.0:
            raise ValueError("temperature_ladder must start at 0.0 and end at 1.0.")
        return v

    @model_validator(mode="after")
    def _validate_ess_args(self) -> BlackJAXSMCConfig:
        both_set = self.target_ess is not None and self.target_ess_fraction is not None
        if both_set:
            raise ValueError(
                "BlackJAXSMCConfig: set exactly one of `target_ess` or "
                "`target_ess_fraction`, not both."
            )

        # Apply default if neither was set.
        if self.target_ess is None and self.target_ess_fraction is None:
            object.__setattr__(self, "target_ess_fraction", 0.9)

        # Validate target_ess.
        if self.target_ess is not None:
            if self.target_ess <= 0:
                raise ValueError(f"target_ess must be > 0, got {self.target_ess}.")
            if not self.persistent_sampling and self.target_ess > self.n_particles:
                raise ValueError(
                    f"target_ess ({self.target_ess}) > n_particles "
                    f"({self.n_particles}) is not valid when persistent_sampling=False; "
                    "the ESS cannot exceed the number of particles. "
                    "Set persistent_sampling=True or lower target_ess."
                )

        # Validate target_ess_fraction.
        if self.target_ess_fraction is not None:
            if self.target_ess_fraction <= 0:
                raise ValueError(
                    f"target_ess_fraction must be > 0, got {self.target_ess_fraction}."
                )
            if not self.persistent_sampling and self.target_ess_fraction > 1.0:
                raise ValueError(
                    f"target_ess_fraction ({self.target_ess_fraction}) > 1.0 is not valid "
                    "when persistent_sampling=False; the ESS fraction cannot exceed 1. "
                    "Set persistent_sampling=True or use a fraction in (0, 1]."
                )

        # Warn if a fixed temperature ladder is given alongside an ESS target.
        if self.temperature_ladder is not None and (
            "target_ess" in self.model_fields_set
            or "target_ess_fraction" in self.model_fields_set
        ):
            warnings.warn(
                "BlackJAXSMCConfig: ESS target has no effect when "
                "`temperature_ladder` is provided (fixed-ladder mode).",
                UserWarning,
                stacklevel=2,
            )
        return self

    def _resolve_target_ess_fraction(self) -> float:
        """Return the ESS target as a fraction of n_particles."""
        if self.target_ess_fraction is not None:
            return self.target_ess_fraction
        assert self.target_ess is not None
        return self.target_ess / self.n_particles


SamplerConfig = Annotated[
    Union[FlowMCConfig, BlackJAXNSAWConfig, BlackJAXNSSConfig, BlackJAXSMCConfig],
    Discriminator("type"),
]
"""Discriminated union of every concrete sampler config."""
