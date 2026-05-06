"""Shared helpers for sampler integration tests.

Provides a lightweight 2-D Gaussian-likelihood Jim instance used by all
four sampler-specific integration tests.  The prior is a uniform [0,1]^2
unit cube; the likelihood peaks at (0.5, 0.5) with σ=0.1 so the posterior
mean is analytically known and easily verifiable.
"""

from __future__ import annotations

from jimgw.core.base import LikelihoodBase
from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.samplers.config import SamplerConfig


class _GaussianLikelihood(LikelihoodBase):
    """Isotropic Gaussian peaked at (0.5, 0.5) with σ=0.1."""

    sigma: float = 0.1

    def evaluate(self, params: dict, data: dict) -> float:  # type: ignore[override]
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / self.sigma**2


def make_gaussian_jim(sampler_config: SamplerConfig) -> Jim:
    """Return a Jim instance wired to the 2-D Gaussian likelihood."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    return Jim(_GaussianLikelihood(), prior, sampler_config)
