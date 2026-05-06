"""Pydantic configuration models for likelihood analytic marginalizations."""

from typing import Optional

from pydantic import BaseModel

from jimgw.core.prior import Prior


class PhaseMargConfig(BaseModel):
    """Configuration sentinel for phase marginalization (no parameters)."""

    model_config = {"extra": "forbid"}


class TimeMargConfig(BaseModel):
    """Configuration for time marginalization."""

    model_config = {"extra": "forbid"}
    tc_range: tuple[float, float] = (-0.1, 0.1)


class DistanceMargConfig(BaseModel):
    """Configuration for distance marginalization."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}
    distance_prior: Prior  # required — no default
    n_dist_points: int = 10000
    ref_dist: Optional[float] = None
