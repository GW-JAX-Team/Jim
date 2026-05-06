"""Integration test: BlackJAX nested slice sampler end-to-end with a 2-D Gaussian."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.integration

blackjax = pytest.importorskip("blackjax")

from jimgw.samplers.config import BlackJAXNSSConfig  # noqa: E402

from tests.integration._helpers import make_gaussian_jim  # noqa: E402


@pytest.fixture(scope="module")
def nss_jim():
    cfg = BlackJAXNSSConfig(n_live=50, termination_dlogz=0.5)
    jim = make_gaussian_jim(cfg)
    jim.sample()
    return jim


def test_nss_get_samples_shape(nss_jim):
    samples = nss_jim.get_samples()
    assert set(samples.keys()) == {"x", "y", "log_likelihood"}
    n = samples["x"].shape[0]
    assert n > 0
    assert samples["y"].shape == (n,)


def test_nss_posterior_mean_near_half(nss_jim):
    samples = nss_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.1
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.1


def test_nss_log_evidence_finite(nss_jim):
    diag = nss_jim.sampler.get_diagnostics()
    log_z = diag["log_Z"]
    assert math.isfinite(log_z)
