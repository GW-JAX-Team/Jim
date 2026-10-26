"""Pydantic configuration models for likelihood analytic marginalizations."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from jimgw.core.prior import Prior


class PhaseMargConfig(BaseModel):
    """Configuration sentinel for phase marginalization (no parameters)."""

    model_config = {"extra": "forbid"}


class TimeMargConfig(BaseModel):
    """Configuration for time marginalization."""

    model_config = {"extra": "forbid", "allow_inf_nan": False}
    tc_range: tuple[float, float] = (-0.1, 0.1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_time_range(self) -> "TimeMargConfig":
        if not self.tc_range[0] < self.tc_range[1]:
            raise ValueError("tc_range must contain finite, increasing bounds.")
        return self


class DistanceMargConfig(BaseModel):
    """Configuration for distance marginalization."""

    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": True,
        "allow_inf_nan": False,
    }
    distance_prior: Prior  # required — no default
    n_dist_points: int = Field(default=10000, ge=2, strict=True)
    ref_dist: Optional[float] = Field(default=None, gt=0.0)
