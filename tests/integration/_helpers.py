"""Shared helpers for sampler integration tests.

Provides a lightweight 2-D Gaussian-likelihood Jim instance used by all
sampler-specific integration tests.  The prior is a uniform [0,1]^2
unit cube; the likelihood peaks at (0.5, 0.5) with σ=0.1 so the posterior
mean is analytically known and easily verifiable.

``make_gaussian_swig_jim`` provides a `SingleEventLikelihood`-derived variant
of the same Gaussian for the cache-aware SwiG sampler, which requires a
waveform-cache-capable likelihood: "x" stands in for the sole
waveform-affecting parameter, so a `["y"]` block exercises the cache-reuse
path.
"""

from __future__ import annotations

from jimgw.core.base import LikelihoodBase
from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.likelihood import SingleEventLikelihood
from jimgw.samplers.config import BlackJAXSwiGConfig, SamplerConfig

_SIGMA = 0.1


def _gaussian_log_likelihood(x, y):
    return -0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / _SIGMA**2


class _GaussianLikelihood(LikelihoodBase):
    """Isotropic Gaussian peaked at (0.5, 0.5) with σ=0.1."""

    sigma: float = _SIGMA

    def evaluate(self, params: dict) -> float:  # type: ignore[override]
        return _gaussian_log_likelihood(params["x"], params["y"])


def make_gaussian_jim(sampler_config: SamplerConfig) -> Jim:
    """Return a Jim instance wired to the 2-D Gaussian likelihood."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    return Jim(_GaussianLikelihood(), prior, sampler_config)


class _WaveformStub:
    parameter_names = ("x",)


class _GaussianWaveformCacheLikelihood(SingleEventLikelihood):
    """Isotropic Gaussian peaked at (0.5, 0.5) with σ=0.1, "x" waveform-only."""

    def __init__(self) -> None:
        self.waveform = _WaveformStub()
        self.fixed_parameters: dict = {}
        self.trigger_time = 0.0
        self.gmst = 0.0
        self.time_marginalization = False
        self.phase_marginalization = False
        self.distance_marginalization = False

    def _evaluate(self, params: dict) -> float:
        return _gaussian_log_likelihood(params["x"], params["y"])

    def generate_waveform(self, params: dict) -> dict:
        return {"x": params["x"]}

    def _evaluate_from_waveform(self, params: dict, waveform_cache: dict) -> float:
        return _gaussian_log_likelihood(waveform_cache["x"], params["y"])


def make_gaussian_swig_jim(sampler_config: BlackJAXSwiGConfig) -> Jim:
    """Return a Jim instance wired to the cache-aware 2-D Gaussian likelihood."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    return Jim(_GaussianWaveformCacheLikelihood(), prior, sampler_config)
