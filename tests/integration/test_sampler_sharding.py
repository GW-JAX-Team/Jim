"""Integration test: sharded BlackJAX NSS/SwiG sampling through the full Jim pipeline.

Requires 4 local JAX devices — run with
``XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest ...``. Unlike
``tests/unit/samplers/blackjax/test_sharding.py`` (which drives
``BlackJAXNSSSampler``/``BlackJAXSwiGSampler`` directly), these tests exercise
sharding through ``Jim`` end-to-end: config parsing, prior/likelihood wiring,
and the checkpoint-free sampling loop are all included.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

_HAS_FOUR_DEVICES = jax.local_device_count() >= 4

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_FOUR_DEVICES,
        reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
    ),
]

blackjax = pytest.importorskip("blackjax")

from jimgw.samplers.config import BlackJAXNSSConfig, BlackJAXSwiGConfig
from tests.integration._helpers import (
    make_gaussian_jim,
    make_gaussian_swig_jim,
)


@pytest.fixture(scope="module")
def sharded_nss_jim():
    cfg = BlackJAXNSSConfig(
        n_live=48, n_delete_frac=0.25, termination_dlogz=0.5, n_devices=4
    )
    jim = make_gaussian_jim(cfg)
    jim.sample()
    return jim


def test_sharded_nss_posterior_mean_near_half(sharded_nss_jim):
    samples = sharded_nss_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.15
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.15


def test_sharded_nss_log_evidence_finite(sharded_nss_jim):
    diag = sharded_nss_jim.sampler.get_diagnostics()
    assert math.isfinite(diag["log_Z"])


@pytest.fixture(scope="module")
def sharded_swig_jim():
    cfg = BlackJAXSwiGConfig(
        blocks=[["x"], ["y"]],
        n_live=48,
        n_delete_frac=0.25,
        termination_dlogz=0.5,
        n_devices=4,
    )
    jim = make_gaussian_swig_jim(cfg)
    jim.sample()
    return jim


def test_sharded_swig_posterior_mean_near_half(sharded_swig_jim):
    samples = sharded_swig_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.15
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.15


def test_sharded_swig_log_evidence_finite(sharded_swig_jim):
    diag = sharded_swig_jim.sampler.get_diagnostics()
    assert np.isfinite(diag["log_Z"])
