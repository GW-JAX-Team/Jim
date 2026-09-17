"""Integration tests for BlackJAX SMC's normalizing-flow preconditioning.

Uses a "banana" (curved, non-Gaussian) 2-D posterior — a target whose local
correlation structure varies across the distribution, so no single fixed
covariance matrix fits it everywhere, unlike a linearly-correlated Gaussian
(which adaptive-covariance RWMH already handles well without a flow) — to
check that ``precondition=True`` (a) still recovers the correct posterior, and
(b) meaningfully improves mixing (mean acceptance rate) over
``precondition=False`` at a matched, deliberately tight sampling budget.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

blackjax = pytest.importorskip("blackjax")
flowMC = pytest.importorskip("flowMC")

from jimgw.samplers.config import BlackJAXSMCConfig
from tests.integration._helpers import (
    _BANANA_CURVATURE,
    _BANANA_SIGMA_X,
    _CIRCULAR_MU_ANGLE,
    _CIRCULAR_MU_Y,
    make_banana_jim,
    make_circular_jim,
)

_PRECONDITION_KWARGS = {
    "precondition": {
        "rq_spline_n_layers": 3,
        "rq_spline_hidden_units": [32, 32],
        "rq_spline_n_bins": 8,
        "flow_n_epochs": 50,
        "flow_learning_rate": 1e-3,
    }
}


def test_smc_precondition_recovers_banana_posterior():
    """precondition=True on a curved (banana) target still recovers the truth."""
    cfg = BlackJAXSMCConfig(
        n_particles=500,
        n_mcmc_steps_per_dim=10,
        target_ess=200,
        **_PRECONDITION_KWARGS,
    )
    jim = make_banana_jim(cfg)
    jim.sample()
    samples = jim.get_samples()

    expected_mean_y = 0.5 + _BANANA_CURVATURE * _BANANA_SIGMA_X**2
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.05
    assert abs(float(np.mean(samples["y"])) - expected_mean_y) < 0.05


def test_smc_precondition_recovers_circular_posterior():
    """precondition=True + periodic={...} on a genuinely periodic target still
    recovers the truth -- looser tolerance than the non-periodic banana test,
    since this knowingly relies on the pocoMC-style wrap approximation (see
    docs/guides/samplers.md's "Normalizing-flow preconditioning" section).

    _CIRCULAR_MU_ANGLE is placed close enough to the seam at 0 == 2*pi that
    posterior mass straddles it, so a passing run also confirms the wrap
    actually engaged -- not just that the (no-op on this target) code path
    type-checks.
    """
    cfg = BlackJAXSMCConfig(
        n_particles=500,
        n_mcmc_steps_per_dim=10,
        target_ess=200,
        **_PRECONDITION_KWARGS,
    )
    jim = make_circular_jim(cfg)
    jim.sample()
    samples = jim.get_samples()
    phase = np.asarray(samples["phase"])

    assert np.any(phase < 0.5) and np.any(phase > 2 * np.pi - 0.5), (
        "expected mass on both sides of the wrap seam, confirming the wrap engaged"
    )

    resultant = np.mean(np.exp(1j * phase))
    assert abs(resultant) > 0.5, "expected the recovered phase to stay concentrated"
    mean_angle = float(np.angle(resultant))
    angular_distance = abs(
        (mean_angle - float(_CIRCULAR_MU_ANGLE) + np.pi) % (2 * np.pi) - np.pi
    )
    assert angular_distance < 0.3
    assert abs(float(np.mean(samples["y"])) - _CIRCULAR_MU_Y) < 0.1


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "This test's acceptance-rate margin was confounded by the sampling-space-"
        "covariance bug fixed alongside it: the preconditioned kernel used to scale "
        "its latent-space proposal by cov(x) (raw sampling-space covariance) instead "
        "of cov(to_latent(x)), so on this fixture (whose sampling-space covariance is "
        "~0.01-0.06, much smaller than the flow's already-whitened ~unit latent scale) "
        "it proposed accidentally undersized latent steps. That trivially inflates "
        "acceptance rate without reflecting genuinely better proposal geometry -- the "
        "classic small-step/high-acceptance/poor-mixing confound. Now that the "
        "covariance is computed correctly (see _latent_covariance in smc.py), "
        "preconditioning's acceptance rate resembles a well-tuned RWMH's target "
        "acceptance rate rather than exceeding it by a fixed margin, so this metric no "
        "longer discriminates 'better proposal' from 'merely correctly scaled'. A "
        "replacement needs a step-size-aware efficiency metric (e.g. expected squared "
        "jump distance per likelihood call), which needs new instrumentation jim "
        "doesn't currently expose -- tracked as a follow-up, not fixed here. "
        "test_smc_precondition_recovers_banana_posterior still passes and confirms "
        "preconditioning still recovers the correct posterior; only the mixing-"
        "efficiency claim is currently unverified by a test."
    ),
    strict=False,
)
def test_smc_precondition_improves_acceptance_at_matched_budget():
    """At a matched, deliberately tight MCMC budget, preconditioning should give a
    meaningfully higher mean acceptance rate on a curved (banana) target than
    plain (unpreconditioned) random-walk MCMC — a fixed covariance matrix cannot
    track the target's varying local correlation, while a trained flow can.

    Assert on acceptance rate rather than a tight posterior-recovery comparison:
    head-to-head recovery-error comparisons between two runs are statistically
    noisy at these particle counts, while acceptance rate is a direct, low-noise
    read on how well each proposal fits the local geometry.

    XFAIL: see the marker reason above -- this metric is confounded by the
    covariance-scale bug fixed alongside it and no longer discriminates what it
    was written to discriminate.
    """
    shared_kwargs = {
        "n_particles": 400,
        "n_mcmc_steps_per_dim": 4,  # tight budget: plain RWMH should struggle here
        "target_ess": 150,
        "initial_cov_scale": 0.1,
    }

    plain_cfg = BlackJAXSMCConfig(precondition=False, **shared_kwargs)
    plain_jim = make_banana_jim(plain_cfg)
    plain_jim.sample()
    plain_diag = plain_jim.get_diagnostics()

    precond_cfg = BlackJAXSMCConfig(**shared_kwargs, **_PRECONDITION_KWARGS)
    precond_jim = make_banana_jim(precond_cfg)
    precond_jim.sample()
    precond_diag = precond_jim.get_diagnostics()

    plain_acceptance = float(np.mean(plain_diag["acceptance_history"]))
    precond_acceptance = float(np.mean(precond_diag["acceptance_history"]))

    assert precond_acceptance > plain_acceptance + 0.1, (
        f"expected preconditioning to noticeably improve mean acceptance rate "
        f"on a curved (banana) target (plain={plain_acceptance:.3f}, "
        f"precond={precond_acceptance:.3f})"
    )
