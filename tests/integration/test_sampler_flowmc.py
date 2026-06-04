"""Integration test: flowMC sampler end-to-end with a 2-D Gaussian."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from jimgw.samplers.config import FlowMCConfig

from tests.integration._helpers import make_gaussian_jim


@pytest.fixture(scope="module")
def flowmc_jim():
    cfg = FlowMCConfig(
        n_chains=50,
        n_local_steps=5,
        n_global_steps=5,
        global_thinning=1,
        n_training_loops=2,
        n_production_loops=2,
        n_epochs=3,
    )
    jim = make_gaussian_jim(cfg)
    jim.sample()
    return jim


def test_flowmc_get_samples_shape(flowmc_jim):
    samples = flowmc_jim.get_samples()
    assert set(samples.keys()) == {"x", "y", "log_likelihood"}
    n = samples["x"].shape[0]
    assert n > 0
    assert samples["y"].shape == (n,)


def test_flowmc_posterior_mean_near_half(flowmc_jim):
    samples = flowmc_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.1
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.1


def test_flowmc_output_has_log_likelihood(flowmc_jim):
    result = flowmc_jim.sampler.get_samples()
    assert "log_likelihood" in result
    assert result["log_likelihood"].shape == (result["samples"].shape[0],)
