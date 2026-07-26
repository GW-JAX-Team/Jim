"""Pydantic configuration models for Jim's samplers.

Each sampler has its own ``*Config`` class discriminated by a ``type`` literal;
`SamplerConfig` is the discriminated-union annotation a caller passes to
``Jim(..., sampler_config=...)``.
"""

import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, Optional, Self, TypeVar, Union

import jax
import numpy as np
from pydantic import BaseModel, Discriminator, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_SamplerType = TypeVar("_SamplerType", bound=str)


class BaseSamplerConfig(BaseModel, Generic[_SamplerType]):
    """Fields shared by all sampler configs."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}
    type: _SamplerType


# ---------------------------------------------------------------------------
# Checkpoint mixin — included by the three BlackJAX configs that support it
# ---------------------------------------------------------------------------


class _CheckpointMixin(BaseModel):
    """Checkpoint/resume fields for samplers that support them.

    Args:
        checkpoint_dir: Directory where ``checkpoint.pkl`` is written.
            ``None`` (default) disables checkpointing.  The directory is
            created automatically if it does not exist.  The checkpoint
            filename is always ``checkpoint.pkl``.
        checkpoint_interval: Minimum wall-clock seconds between checkpoint
            writes.  Default ``0`` (checkpointing disabled).  Set to a
            positive value to enable; ``checkpoint_dir`` must also be set.
    """

    # model_config is inherited from BaseSamplerConfig; not redeclared here.

    checkpoint_dir: Optional[Path] = None
    checkpoint_interval: float = Field(default=0.0, ge=0.0)

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def _coerce_checkpoint_dir(cls, v: object) -> Optional[Path]:
        if v is None:
            return None
        return Path(str(v))

    @model_validator(mode="after")
    def _check_checkpoint_consistency(self) -> Self:
        if self.checkpoint_interval > 0 and self.checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir must be set when checkpoint_interval > 0. "
                "Provide a directory path or set checkpoint_interval=0 to disable checkpointing."
            )
        return self

    def write_checkpoint(self, data: dict, label: str) -> float:
        """Atomically write *data* to ``checkpoint_dir/checkpoint.pkl``.

        The write is done via a temporary ``.pkl.tmp`` file that is renamed
        into place so a crash mid-write never leaves a corrupt checkpoint.

        Args:
            data: Serialisable dict to pickle.
            label: Human-readable label included in the debug log message.

        Returns:
            Wall-clock time of the write (``time.perf_counter()``), suitable
            for resetting the caller's ``_last_ckpt_t`` timer.
        """
        assert self.checkpoint_dir is not None
        ckpt_path = self.checkpoint_dir / "checkpoint.pkl"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as _f:
            pickle.dump(data, _f)
        tmp.replace(ckpt_path)
        t = time.perf_counter()
        logger.debug(
            "%s: checkpoint saved at n_iter=%s",
            label,
            data.get("n_iter", "?"),
        )
        return t

    def configure_jax_cache(self) -> None:
        """Enable JAX's persistent XLA compilation cache under ``checkpoint_dir/jax_cache``.

        Sets ``jax_compilation_cache_dir`` to ``{checkpoint_dir}/jax_cache`` so
        that compiled functions are stored to disk and reused across processes
        (e.g., after a crash-and-resume).  Also sets
        ``jax_persistent_cache_min_compile_time_secs`` to ``0.0`` so that all
        compilations are cached regardless of their duration.

        No-op if ``checkpoint_dir`` is ``None``.  Safe to call multiple times.
        """
        if self.checkpoint_dir is None:
            return

        cache_dir = self.checkpoint_dir / "jax_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


class _ShardingMixin(BaseModel):
    """Single-host device sharding — included by NSS and SwiG (not NS AW)."""

    n_devices: int = Field(default=1, ge=1)


# ---------------------------------------------------------------------------
# Live-set mixin — shared by the nested samplers (NS AW, NSS, SwiG) that
# maintain a live-particle set and delete/replace a fraction of it each step
# ---------------------------------------------------------------------------


class _LiveSetConfigMixin(BaseModel):
    """Shared ``n_live``/``n_delete_frac`` fields and validation for nested samplers."""

    n_live: int
    n_delete_frac: float

    @field_validator("n_delete_frac")
    @classmethod
    def _n_delete_frac_range(cls, value: float) -> float:
        if not (0.0 < value < 1.0):
            raise ValueError("n_delete_frac must be strictly between 0 and 1")
        return value

    @model_validator(mode="after")
    def _n_live_n_delete_consistency(self) -> Self:
        if self.n_live < 2:
            raise ValueError(f"n_live must be >= 2 (got {self.n_live}).")
        n_delete = int(self.n_live * self.n_delete_frac)
        if n_delete < 1:
            raise ValueError(
                f"n_live * n_delete_frac = {self.n_live * self.n_delete_frac} "
                f"yields n_delete = {n_delete}; require n_delete >= 1. "
                "Increase n_live or n_delete_frac."
            )
        # n_devices is only present on samplers that also mix in _ShardingMixin
        # (NSS, SwiG); absent (default 1) elsewhere, so this is a no-op for NS AW.
        n_devices = getattr(self, "n_devices", 1)
        if n_devices > 1 and self.n_live % n_devices:
            raise ValueError("n_live must be divisible by n_devices when sharding")
        if n_devices > 1 and n_delete % n_devices:
            raise ValueError("n_delete must be divisible by n_devices when sharding")
        return self


# ---------------------------------------------------------------------------
# flowMC sub-configs
# ---------------------------------------------------------------------------


class ParallelTemperingConfig(BaseModel):
    """Parallel-tempering settings for the flowMC backend.

    Construct directly or pass a plain ``dict`` to ``FlowMCConfig.parallel_tempering``.
    Use ``True`` to enable with all defaults, ``False`` / ``None`` to disable.
    """

    model_config = {"extra": "forbid"}

    n_temperatures: int = Field(default=5, ge=1)
    max_temperature: float = Field(default=10.0, gt=0.0)
    n_tempered_steps: int = Field(default=5, ge=1)


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
    n_leapfrog_steps: int = Field(default=10, ge=1)


class GRWConfig(BaseModel):
    """Gaussian random-walk local-kernel settings for the flowMC backend.

    ``step_size`` may be a scalar ``float`` (applied uniformly to all
    dimensions) or a 1-D ``np.ndarray`` of length ``n_dims`` for
    per-dimension step sizes.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    step_size: float | np.ndarray = 2e-3


class FlowMCConfig(BaseSamplerConfig[Literal["flowmc"]], _CheckpointMixin):
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

    n_chains: int = Field(default=1000, ge=1)
    n_local_steps: int = Field(default=100, ge=1)
    n_global_steps: int = Field(default=1000, ge=1)
    n_training_loops: int = Field(default=20, ge=1)
    n_production_loops: int = Field(default=10, ge=1)
    n_epochs: int = Field(default=20, ge=1)

    local_kernel: Literal["MALA", "HMC", "GRW"] = "MALA"
    parallel_tempering: Optional[ParallelTemperingConfig] = None
    # dict[str, Any] accepted here; Pydantic coerces it to the typed config via field_validator.
    mala: MALAConfig | dict[str, Any] = Field(default_factory=MALAConfig)
    hmc: HMCConfig | dict[str, Any] = Field(default_factory=HMCConfig)
    grw: GRWConfig | dict[str, Any] = Field(default_factory=GRWConfig)

    rq_spline_hidden_units: list[int] = Field(default_factory=lambda: [128, 128])
    rq_spline_n_bins: int = Field(default=10, ge=1)
    rq_spline_n_layers: int = Field(default=8, ge=1)
    n_NFproposal_batch_size: int = Field(default=1000, ge=1)

    learning_rate: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=10000, ge=1)
    n_max_examples: int = Field(default=30000, ge=1)
    history_window: int = Field(default=100, ge=1)

    # chain_batch_size=0 means "full batch" (flowMC's off sentinel), not a lower bound of 1.
    chain_batch_size: int = Field(default=0, ge=0)
    local_thinning: int = Field(default=1, ge=1)
    global_thinning: int = Field(default=100, ge=1)

    early_stopping: bool = True
    early_stopping_tolerance: float = Field(default=0.1, ge=0.0)
    early_stopping_patience: int = Field(default=3, ge=1)
    early_stopping_min_acceptance: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("parallel_tempering", mode="before")
    @classmethod
    def _coerce_parallel_tempering(cls, v: object) -> Optional[ParallelTemperingConfig]:
        if v is None or v is False:
            return None
        if v is True:
            return ParallelTemperingConfig()
        if isinstance(v, dict):
            return ParallelTemperingConfig.model_validate(v)
        if isinstance(v, ParallelTemperingConfig):
            return v
        raise ValueError(
            "parallel_tempering must be None, False, True, a dict of ParallelTemperingConfig "
            f"fields, or a ParallelTemperingConfig instance; got {type(v).__name__}."
        )

    @model_validator(mode="after")
    def _warn_if_irrelevant_kernel_set(self) -> Self:
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


class BlackJAXNSAWConfig(
    BaseSamplerConfig[Literal["blackjax-ns-aw"]], _CheckpointMixin, _LiveSetConfigMixin
):
    """Configuration for the BlackJAX acceptance-walk nested sampler.

    !!! note
        This sampler requires the sampling space to be the unit hypercube
        ``[0, 1]^n_dims``.  When using Jim, this means all
        ``sample_transforms`` must map the prior support onto the unit cube.

    !!! note
        Periodic parameters are **not** configured here.  Pass a
        ``periodic`` argument to [`Jim`][jimgw.core.jim.Jim] instead.
        For NS AW, bounds are implicit as ``[0, 1]``; just list the
        parameter names.
    """

    type: Literal["blackjax-ns-aw"] = "blackjax-ns-aw"

    n_live: int = 1000
    n_delete_frac: float = 0.5
    n_target: int = Field(default=60, ge=1)
    max_mcmc: int = Field(default=5000, ge=1)
    max_proposals: int = Field(default=1000, ge=1)
    termination_dlogz: float = Field(default=0.1, gt=0.0)


class BlackJAXNSSConfig(
    BaseSamplerConfig[Literal["blackjax-nss"]],
    _CheckpointMixin,
    _LiveSetConfigMixin,
    _ShardingMixin,
):
    """Configuration for the BlackJAX nested slice sampler.

    !!! note
        Periodic parameters are **not** configured here.  Pass a ``periodic``
        argument to [`Jim`][jimgw.core.jim.Jim] instead.
    """

    type: Literal["blackjax-nss"] = "blackjax-nss"

    n_live: int = 2000
    n_delete_frac: float = 0.5
    num_inner_steps_per_dim: int = Field(default=20, ge=1)
    termination_dlogz: float = Field(default=0.1, gt=0.0)


class BlackJAXSwiGConfig(
    BaseSamplerConfig[Literal["blackjax-swig"]],
    _CheckpointMixin,
    _LiveSetConfigMixin,
    _ShardingMixin,
):
    """Configuration for Nested Slice within Gibbs (SwiG) sampling.

    ``blocks`` are expressed in sampling-space parameter names.
    A supported likelihood validates that they form an exact partition and
    identifies which blocks refresh its cache.
    """

    type: Literal["blackjax-swig"] = "blackjax-swig"

    blocks: list[list[str]]
    n_live: int = 500
    n_delete_frac: float = 0.125
    num_gibbs_sweeps: int = Field(default=2, ge=1)
    num_inner_steps_per_dim: int = Field(default=1, ge=1)
    max_steps: int = Field(default=10, ge=1)
    max_shrinkage: int = Field(default=100, ge=1)
    termination_dlogz: float = Field(default=0.1, gt=0.0)

    @field_validator("blocks")
    @classmethod
    def _validate_blocks(cls, blocks: list[list[str]]) -> list[list[str]]:
        if not blocks:
            raise ValueError("blocks must contain at least one parameter block")
        if any(not block for block in blocks):
            raise ValueError("blocks cannot contain empty parameter blocks")
        flat = [name for block in blocks for name in block]
        duplicates = sorted({name for name in flat if flat.count(name) > 1})
        if duplicates:
            raise ValueError(f"parameters appear in multiple blocks: {duplicates}")
        return blocks


class BlackJAXSMCConfig(BaseSamplerConfig[Literal["blackjax-smc"]], _CheckpointMixin):
    """Configuration for the BlackJAX SMC sampler.

    Parameters
    ----------
    batch_size : int, optional
        Number of particles to process per sequential batch during the MCMC
        update step. When ``batch_size > 0``, the sampler uses ``jax.lax.map``
        instead of ``jax.vmap``, which reduces peak GPU memory at the cost
        of sequential execution. ``0`` (default) uses the original full
        ``jax.vmap`` behaviour.

    !!! note
        Periodic parameters are **not** configured here.  Pass a ``periodic``
        argument to [`Jim`][jimgw.core.jim.Jim] instead.
    """

    type: Literal["blackjax-smc"] = "blackjax-smc"

    n_particles: int = Field(default=5000, ge=1)
    n_mcmc_steps_per_dim: int = Field(default=100, ge=1)
    target_ess: Optional[int] = None
    target_ess_fraction: Optional[float] = None
    # 0 = full vmap; >0 = lax.map batch size to reduce peak memory
    batch_size: int = Field(default=0, ge=0)
    initial_cov_scale: float = Field(default=0.5, gt=0.0)
    target_acceptance_rate: float = Field(default=0.234, gt=0.0, lt=1.0)
    scale_adaptation_gain: float = Field(default=3.0, gt=0.0)

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
    def _validate_ess_args(self) -> Self:
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
    Union[
        FlowMCConfig,
        BlackJAXNSAWConfig,
        BlackJAXNSSConfig,
        BlackJAXSwiGConfig,
        BlackJAXSMCConfig,
    ],
    Discriminator("type"),
]
"""Discriminated union of every concrete sampler config."""
