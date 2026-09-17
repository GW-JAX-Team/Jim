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

``make_banana_jim`` provides a curved ("banana") 2-D posterior whose local
correlation structure varies across the distribution, so no single fixed
covariance matrix fits it everywhere (unlike a linearly-correlated Gaussian);
used to exercise SMC's ``precondition=True`` path.

``make_circular_jim`` provides a genuinely periodic ("phase") 2-D posterior;
used to exercise SMC's ``precondition=True`` path combined with ``periodic``.
"""

import jax.numpy as jnp

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

    def _generate_waveform(self, params: dict) -> dict:
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


# "Banana" (curved/nonlinear-correlation) posterior; see module docstring.

_BANANA_SIGMA_X = 0.12
_BANANA_SIGMA_Y = 0.05
_BANANA_CURVATURE = 2.0


def _banana_log_likelihood(x, y):
    mu_y = 0.5 + _BANANA_CURVATURE * (x - 0.5) ** 2
    return -0.5 * (
        (x - 0.5) ** 2 / _BANANA_SIGMA_X**2 + (y - mu_y) ** 2 / _BANANA_SIGMA_Y**2
    )


class _BananaLikelihood(LikelihoodBase):
    """Curved ("banana") 2-D posterior: x ~ N(0.5, sigma_x^2), y centered on a
    parabola in x rather than a fixed linear correlation."""

    def evaluate(self, params: dict) -> float:  # type: ignore[override]
        return _banana_log_likelihood(params["x"], params["y"])


def make_banana_jim(sampler_config: SamplerConfig) -> Jim:
    """Return a Jim instance wired to the curved ("banana") 2-D likelihood."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    return Jim(_BananaLikelihood(), prior, sampler_config)


# Genuinely periodic ("circular") posterior; see module docstring.

_CIRCULAR_KAPPA = 8.0
_CIRCULAR_MU_ANGLE = 0.8  # close enough to the seam at 0 == 2*pi that mass straddles it, exercising the wrap
_CIRCULAR_MU_Y = 0.5
_CIRCULAR_SIGMA_Y = 0.1


def _circular_log_likelihood(phase, y):
    return (
        _CIRCULAR_KAPPA * jnp.cos(phase - _CIRCULAR_MU_ANGLE)
        - 0.5 * ((y - _CIRCULAR_MU_Y) / _CIRCULAR_SIGMA_Y) ** 2
    )


class _CircularLikelihood(LikelihoodBase):
    """Genuinely periodic ("phase") 2-D posterior: phase ~ von-Mises-like on
    [0, 2*pi), y ~ N(0.5, sigma_y^2)."""

    def evaluate(self, params: dict) -> float:  # type: ignore[override]
        return _circular_log_likelihood(params["phase"], params["y"])


def make_circular_jim(sampler_config: SamplerConfig) -> Jim:
    """Return a Jim instance wired to the periodic ("circular") 2-D likelihood,
    with "phase" declared periodic on [0, 2*pi)."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    return Jim(
        _CircularLikelihood(),
        prior,
        sampler_config,
        periodic={"phase": (0.0, 2 * jnp.pi)},
    )
