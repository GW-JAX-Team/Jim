"""Tests for CLI prior construction and time-frame adaptation."""

import pytest

from jimgw.cli._config import GaussianSpec, PriorConfig, SamplingConfig, UniformSpec
from jimgw.cli._prior import adapt_prior_for_ns_time


@pytest.mark.parametrize(
    "source, detector_frame_time",
    [("t_c", True), ("t_det", False)],
)
def test_time_adaptation_requires_uniform_prior(source, detector_frame_time):
    prior = PriorConfig({source: GaussianSpec(loc=0.0, scale=0.1)})
    sampling = SamplingConfig(detector_frame_time=detector_frame_time)
    with pytest.raises(ValueError, match=f"uniform {source} prior"):
        adapt_prior_for_ns_time(prior, sampling)


@pytest.mark.parametrize(
    "source, destination, detector_frame_time",
    [("t_c", "t_det", True), ("t_det", "t_c", False)],
)
def test_time_adaptation_preserves_uniform_bounds(
    source, destination, detector_frame_time
):
    spec = UniformSpec(min=-0.1, max=0.2)
    prior = PriorConfig({source: spec})
    sampling = SamplingConfig(detector_frame_time=detector_frame_time)
    result = adapt_prior_for_ns_time(prior, sampling)
    assert result is not None
    assert list(result.root) == [destination]
    assert result.root[destination] == spec
    assert list(prior.root) == [source]
