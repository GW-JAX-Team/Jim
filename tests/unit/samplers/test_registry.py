"""Tests for the sampler registry and build_sampler factory."""

import sys

import pytest

from jimgw.core.prior import CombinePrior, UniformPrior  # type: ignore[attr-defined]
from jimgw.samplers import build_sampler
from jimgw.samplers.config import (
    BlackJAXNSAWConfig,
    FlowMCConfig,
)
from jimgw.samplers.flowmc import FlowMCSampler  # type: ignore[import]


def _make_prior():
    return CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
        ]
    )


def _make_callables(prior):
    """Build minimal callables from a uniform prior."""
    names = prior.parameter_names

    def log_prior_fn(arr):
        named = dict(zip(names, arr, strict=True))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        return 0.0

    def log_posterior_fn(arr):
        return log_prior_fn(arr)

    return log_prior_fn, log_likelihood_fn, log_posterior_fn


def test_build_sampler_returns_flowmc():
    prior = _make_prior()
    lp, ll, lpost = _make_callables(prior)
    cfg = FlowMCConfig(
        n_chains=10,
        n_local_steps=2,
        n_global_steps=2,
        global_thinning=1,
        n_training_loops=1,
        n_production_loops=1,
        n_epochs=1,
        parallel_tempering=None,
    )
    sampler = build_sampler(
        cfg,
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
    )
    assert isinstance(sampler, FlowMCSampler)


def test_build_sampler_unknown_type_raises():
    from jimgw.samplers.config import BaseSamplerConfig

    class _FakeConfig(BaseSamplerConfig):
        type: str = "not-a-real-type"  # type: ignore[assignment]

    prior = _make_prior()
    lp, ll, lpost = _make_callables(prior)
    fake_config = _FakeConfig()
    with pytest.raises(KeyError, match="not-a-real-type"):
        build_sampler(
            fake_config,  # type: ignore[arg-type]
            n_dims=1,
            log_prior_fn=lp,
            log_likelihood_fn=ll,
            log_posterior_fn=lpost,
        )


def test_build_sampler_blackjax_raises_import_error_when_missing(monkeypatch):
    """When blackjax is not installed, requesting a BlackJAX sampler should raise ImportError."""
    monkeypatch.setitem(sys.modules, "blackjax", None)  # type: ignore[arg-type]

    prior = _make_prior()
    lp, ll, lpost = _make_callables(prior)
    cfg = BlackJAXNSAWConfig()
    with pytest.raises((ImportError, KeyError)):
        build_sampler(
            cfg,
            n_dims=1,
            log_prior_fn=lp,
            log_likelihood_fn=ll,
            log_posterior_fn=lpost,
        )


def test_registry_has_all_four_types():
    from jimgw.samplers import _REGISTRY

    assert "flowmc" in _REGISTRY
    assert "blackjax-ns-aw" in _REGISTRY
    assert "blackjax-nss" in _REGISTRY
    assert "blackjax-smc" in _REGISTRY
