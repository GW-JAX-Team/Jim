"""Tests for CLI prior construction and time-frame adaptation."""

import pytest

from jimgw.cli._config import GaussianSpec, PriorConfig, SamplingConfig, UniformSpec
from jimgw.cli._prior import adapt_prior_for_ns_time


@pytest.mark.parametrize(
    "source, time_frame",
    [("t_c", "H1"), ("t_det", "geocentric")],
)
def test_time_adaptation_requires_uniform_prior(source, time_frame):
    prior = PriorConfig({source: GaussianSpec(loc=0.0, scale=0.1)})
    with pytest.raises(ValueError, match=f"uniform {source} prior"):
        adapt_prior_for_ns_time(prior, SamplingConfig(time_frame=time_frame))


@pytest.mark.parametrize(
    "source, destination, time_frame",
    [("t_c", "t_det", "H1"), ("t_det", "t_c", "geocentric")],
)
def test_time_adaptation_preserves_uniform_bounds(source, destination, time_frame):
    spec = UniformSpec(min=-0.1, max=0.2)
    prior = PriorConfig({source: spec})
    result = adapt_prior_for_ns_time(prior, SamplingConfig(time_frame=time_frame))
    assert result is not None
    assert list(result.root) == [destination]
    assert result.root[destination] == spec
    assert list(prior.root) == [source]
