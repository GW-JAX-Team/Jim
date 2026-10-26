"""Integration test: BlackJAX SMC end to end with a 2-D Gaussian.

Each of the four modes runs the default kernel (t-pCN) with a small flow, on
the real line through logit sample transforms.  The persistent modes also
reproduce the known evidence.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.integration

blackjax = pytest.importorskip("blackjax")

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.transforms import BoundToUnbound
from jimgw.samplers.config import BlackJAXSMCConfig, SMCFlowConfig
from tests.integration._helpers import _SIGMA, _GaussianLikelihood

# Evidence of the Gaussian likelihood under the uniform prior on [0, 1]^2; the
# Gaussian mass outside the square is below 1e-6.
_LOG_Z = float(2 * jnp.log(jnp.sqrt(2 * jnp.pi) * _SIGMA))

_SMC_MODES = [
    pytest.param({}, id="adaptive-persistent"),
    pytest.param({"persistent_sampling": False}, id="adaptive-tempered"),
    pytest.param(
        {"temperature_ladder": [0.0, 0.01, 0.1, 0.3, 1.0]}, id="fixed-persistent"
    ),
    pytest.param(
        {
            "temperature_ladder": [0.0, 0.01, 0.1, 0.3, 1.0],
            "persistent_sampling": False,
        },
        id="fixed-tempered",
    ),
]


@pytest.fixture(scope="module", params=_SMC_MODES)
def smc_jim(request):
    cfg = BlackJAXSMCConfig(
        n_particles=1000,
        flow=SMCFlowConfig(n_layers=2, hidden_size=[16], n_epochs=5, batch_size=250),
        **request.param,
    )
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    jim = Jim(
        _GaussianLikelihood(),
        prior,
        cfg,
        sample_transforms=[
            BoundToUnbound((["x"], ["x_unbounded"]), 0.0, 1.0),
            BoundToUnbound((["y"], ["y_unbounded"]), 0.0, 1.0),
        ],
    )
    jim.sample()
    return jim


def test_smc_get_samples_shape(smc_jim):
    samples = smc_jim.get_samples()
    assert set(samples.keys()) == {"x", "y", "log_likelihood"}
    n = samples["x"].shape[0]
    assert n > 0
    assert samples["y"].shape == (n,)
    assert samples["log_likelihood"].shape == (n,)
    assert np.all(np.isfinite(samples["log_likelihood"]))


def test_smc_posterior_mean_near_half(smc_jim):
    samples = smc_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.02
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.02


def test_smc_persistent_evidence_matches_the_known_value(smc_jim):
    diagnostics = smc_jim.get_diagnostics()
    if "log_Z" not in diagnostics:
        pytest.skip("tempered SMC modes report no evidence")
    tolerance = 4 * max(diagnostics["log_Z_error"], 0.02)
    assert diagnostics["log_Z"] == pytest.approx(_LOG_Z, abs=tolerance)
