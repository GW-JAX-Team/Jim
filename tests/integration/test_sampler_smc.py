"""Integration test: BlackJAX SMC sampler end-to-end with a 2-D Gaussian.

Tests the two adaptive modes (persistent_sampling=True and False).
The fixed-ladder modes are exercised by the unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration

blackjax = pytest.importorskip("blackjax")

from jimgw.samplers.config import BlackJAXSMCConfig  # noqa: E402

from tests.integration._helpers import make_gaussian_jim  # noqa: E402

_SMC_MODES = [
    pytest.param(True, id="adaptive-persistent"),
    pytest.param(False, id="adaptive-tempered"),
]


@pytest.fixture(scope="module", params=_SMC_MODES)
def smc_jim(request):
    cfg = BlackJAXSMCConfig(
        n_particles=100,
        n_mcmc_steps_per_dim=5,
        target_ess=50,
        persistent_sampling=request.param,
    )
    jim = make_gaussian_jim(cfg)
    jim.sample()
    return jim


def test_smc_get_samples_shape(smc_jim):
    samples = smc_jim.get_samples()
    assert set(samples.keys()) == {"x", "y", "log_likelihood"}
    n = samples["x"].shape[0]
    assert n > 0
    assert samples["y"].shape == (n,)
    assert samples["log_likelihood"].shape == (n,)


def test_smc_posterior_mean_near_half(smc_jim):
    samples = smc_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.1
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.1


def test_smc_output_has_log_likelihood(smc_jim):
    result = smc_jim.sampler.get_samples()
    assert "log_likelihood" in result
    ll = result["log_likelihood"]
    assert isinstance(ll, np.ndarray), "log_likelihood must be a numpy ndarray"
    assert ll.ndim >= 1, "log_likelihood must have at least one dimension"
    assert ll.size > 0, "log_likelihood must be non-empty"
    assert np.all(np.isfinite(ll)), "log_likelihood must contain only finite values"
