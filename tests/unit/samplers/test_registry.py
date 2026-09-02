"""Tests for the sampler registry and build_sampler factory."""

import pytest

from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.samplers import build_sampler
from jimgw.samplers.blackjax.ns_aw import BlackJAXNSAWSampler
from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler
from jimgw.samplers.blackjax.smc import BlackJAXSMCSampler
from jimgw.samplers.blackjax.swig import BlackJAXSwiGSampler
from jimgw.samplers.config import (
    BlackJAXNSAWConfig,
    BlackJAXNSSConfig,
    BlackJAXSMCConfig,
    BlackJAXSwiGConfig,
    FlowMCConfig,
)
from jimgw.samplers.flowmc import FlowMCSampler


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


def test_build_sampler_returns_blackjax_ns_aw():
    prior = _make_prior()
    lp, ll, lpost = _make_callables(prior)
    sampler = build_sampler(
        BlackJAXNSAWConfig(),
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
    )
    assert isinstance(sampler, BlackJAXNSAWSampler)


def test_build_sampler_returns_blackjax_nss():
    prior = _make_prior()
    lp, ll, lpost = _make_callables(prior)
    sampler = build_sampler(
        BlackJAXNSSConfig(),
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
    )
    assert isinstance(sampler, BlackJAXNSSSampler)


def test_build_sampler_returns_blackjax_smc():
    prior = _make_prior()
    lp, ll, lpost = _make_callables(prior)
    sampler = build_sampler(
        BlackJAXSMCConfig(),
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
    )
    assert isinstance(sampler, BlackJAXSMCSampler)


def test_build_sampler_forwards_backend_specific_arguments(monkeypatch):
    from jimgw.samplers import _REGISTRY

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
    received_kwargs = {}

    def builder(**kwargs):
        received_kwargs.update(kwargs)
        return FlowMCSampler(**kwargs)

    monkeypatch.setitem(_REGISTRY, "flowmc", lambda: builder)
    sampler = build_sampler(
        cfg,
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
    )
    assert isinstance(sampler, FlowMCSampler)
    assert received_kwargs["config"] is cfg


def test_build_sampler_unknown_type_raises():
    from jimgw.samplers.config import BaseSamplerConfig

    class _FakeConfig(BaseSamplerConfig[str]):
        type: str = "not-a-real-type"

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


def test_build_sampler_requires_cache_callbacks():
    prior = _make_prior()
    lp, ll, lpost = _make_callables(prior)
    with pytest.raises(ValueError, match="requires cache callbacks"):
        build_sampler(
            BlackJAXSwiGConfig(blocks=[["x"]]),
            n_dims=1,
            log_prior_fn=lp,
            log_likelihood_fn=ll,
            log_posterior_fn=lpost,
        )


def test_build_sampler_returns_cache_aware_sampler():
    prior = _make_prior()
    lp, ll, lpost = _make_callables(prior)

    def build_cache(position):
        return position

    def log_likelihood_from_cache_fn(position, cache):
        del cache
        return ll(position)

    sampler = build_sampler(
        BlackJAXSwiGConfig(blocks=[["x"]]),
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        rebuild_required_by_block={(0,): True},
        build_cache=build_cache,
        log_likelihood_from_cache_fn=log_likelihood_from_cache_fn,
    )
    assert isinstance(sampler, BlackJAXSwiGSampler)


def test_build_sampler_rejects_backend_specific_arguments_for_flowmc():
    prior = _make_prior()
    lp, ll, lpost = _make_callables(prior)

    with pytest.raises(TypeError, match="unexpected keyword argument 'build_cache'"):
        build_sampler(
            FlowMCConfig(),
            n_dims=1,
            log_prior_fn=lp,
            log_likelihood_fn=ll,
            log_posterior_fn=lpost,
            build_cache=lambda position: position,
        )


def test_registry_has_black_box_sampler_types():
    from jimgw.samplers import _REGISTRY

    assert "flowmc" in _REGISTRY
    assert "blackjax-ns-aw" in _REGISTRY
    assert "blackjax-nss" in _REGISTRY
    assert "blackjax-smc" in _REGISTRY
    assert "blackjax-swig" in _REGISTRY
