"""Unit tests for flow-preconditioning primitives used by the BlackJAX SMC samplers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")
flowMC = pytest.importorskip("flowMC")

from blackjax.mcmc import random_walk

from jimgw.samplers.blackjax.smc.precondition import (
    build_flow,
    build_optimizer,
    combine_flow,
    flatten_flow,
    from_latent,
    partition_flow,
    to_latent,
    train_flow,
    training_particles_from_weighted,
)
from jimgw.samplers.config import PreconditionConfig
from jimgw.samplers.periodic import to_position_wrapper

jax.config.update("jax_enable_x64", True)

_N_DIMS = 2


def _small_config(**overrides) -> PreconditionConfig:
    """Small flow config for fast tests.

    ``flow_learning_rate`` is deliberately kept at ``PreconditionConfig``'s own
    default (1e-3): a more aggressive rate leaves the trained RQSpline's inverse
    root-solve measurably less precise, which affects how tight a round-trip
    test's tolerance can be (though not MCMC correctness; see
    ``build_preconditioned_mcmc_step``).
    """
    defaults = {
        "rq_spline_n_layers": 2,
        "rq_spline_hidden_units": [16, 16],
        "rq_spline_n_bins": 4,
        "flow_n_epochs": 100,
        "flow_learning_rate": 1e-3,
    }
    defaults.update(overrides)
    return PreconditionConfig(**defaults)


def _build_and_train_flow(key, particles, config):
    flow_key, train_key = jax.random.split(key)
    flow = build_flow(_N_DIMS, config, flow_key)
    optimizer = build_optimizer(flow, config)
    _, flow, optimizer, loss_history = train_flow(
        train_key, flow, optimizer, particles, config
    )
    return flow, optimizer, loss_history


# Sampling-space <-> latent-space mapping correctness


def test_to_latent_from_latent_round_trip():
    key = jax.random.key(0)
    flow_key, x_key = jax.random.split(key)
    config = _small_config()
    flow = build_flow(_N_DIMS, config, flow_key)
    # Non-trivial (non-identity) whitening stats, so this isn't a vacuous mean=0/cov=I check.
    particles = jax.random.multivariate_normal(
        x_key, jnp.array([1.0, -2.0]), jnp.array([[2.0, 0.5], [0.5, 1.0]]), (256,)
    )
    flow, _, _ = _build_and_train_flow(key, particles, config)

    xs = jax.random.multivariate_normal(
        jax.random.key(1),
        jnp.array([1.0, -2.0]),
        jnp.array([[2.0, 0.5], [0.5, 1.0]]),
        (64,),
    )

    def round_trip(x):
        latent, _ = to_latent(flow, x)
        x_rec, _ = from_latent(flow, latent)
        return x_rec

    x_rec = jax.vmap(round_trip)(xs)
    # A trained RQSpline's inverse root-solve is only approximate (unlike a fresh flow's ~1e-15).
    np.testing.assert_allclose(np.asarray(x_rec), np.asarray(xs), atol=0.03)


def test_log_det_sign_matches_flow_log_prob():
    """The single most important test: get the change-of-variables sign right.

    `to_latent`'s log_det is defined as log|d latent/dx|; by construction this must
    satisfy flow.log_prob(x) == base_dist.log_prob(latent) + log_det (the standard
    forward change-of-variables identity used to build ``logdensity_latent`` in
    ``build_preconditioned_mcmc_step``). A sign error here would silently bias
    every posterior produced with ``precondition=True``.
    """
    key = jax.random.key(2)
    _flow_key, x_key = jax.random.split(key)
    config = _small_config()
    particles = jax.random.multivariate_normal(
        x_key, jnp.array([0.5, 0.5]), jnp.array([[1.0, 0.3], [0.3, 1.0]]), (256,)
    )
    flow, _, _ = _build_and_train_flow(key, particles, config)

    xs = particles[:32]

    def check(x):
        latent, log_det = to_latent(flow, x)
        return flow.log_prob(x), flow.base_dist.log_prob(latent) + log_det

    log_prob_direct, log_prob_via_latent = jax.vmap(check)(xs)
    np.testing.assert_allclose(
        np.asarray(log_prob_via_latent),
        np.asarray(log_prob_direct),
        atol=1e-4,
        rtol=1e-4,
    )


def test_to_latent_log_det_matches_numerical_jacobian():
    """Cross-check the analytic log-det against a jax.jacfwd numerical Jacobian."""
    key = jax.random.key(3)
    _flow_key, x_key = jax.random.split(key)
    config = _small_config()
    particles = jax.random.multivariate_normal(
        x_key, jnp.zeros(_N_DIMS), jnp.eye(_N_DIMS), (256,)
    )
    flow, _, _ = _build_and_train_flow(key, particles, config)

    x0 = jnp.array([0.3, -0.2])

    def latent_of_x(x):
        latent, _ = to_latent(flow, x)
        return latent

    jac = jax.jacfwd(latent_of_x)(x0)
    _, analytic_log_det = to_latent(flow, x0)
    numerical_log_det = jnp.linalg.slogdet(jac)[1]
    assert float(analytic_log_det) == pytest.approx(float(numerical_log_det), abs=1e-4)


# Partition / flatten / combine round-trip


def test_partition_combine_round_trip_reproduces_flow():
    key = jax.random.key(4)
    config = _small_config()
    flow = build_flow(_N_DIMS, config, key)

    flat_params, unravel_fn, static = partition_flow(flow)
    rebuilt = combine_flow(flat_params, unravel_fn, static)

    x = jnp.array([0.1, -0.4])
    latent_orig, log_det_orig = to_latent(flow, x)
    latent_rebuilt, log_det_rebuilt = to_latent(rebuilt, x)
    np.testing.assert_allclose(np.asarray(latent_orig), np.asarray(latent_rebuilt))
    np.testing.assert_allclose(np.asarray(log_det_orig), np.asarray(log_det_rebuilt))


def test_flatten_flow_matches_partition_after_training():
    key = jax.random.key(5)
    _flow_key, data_key = jax.random.split(key)
    config = _small_config()
    particles = jax.random.multivariate_normal(
        data_key, jnp.zeros(_N_DIMS), jnp.eye(_N_DIMS), (256,)
    )
    flow, _, _ = _build_and_train_flow(key, particles, config)

    flat_params, unravel_fn, static = partition_flow(flow)
    flat_again = flatten_flow(flow)
    np.testing.assert_allclose(np.asarray(flat_again), np.asarray(flat_params))

    rebuilt = combine_flow(flat_again, unravel_fn, static)
    x = jnp.array([0.2, 0.2])
    np.testing.assert_allclose(
        np.asarray(rebuilt.log_prob(x)), np.asarray(flow.log_prob(x)), atol=1e-6
    )


# Training


def test_train_flow_reduces_loss_and_uses_full_batch():
    key = jax.random.key(6)
    flow_key, data_key = jax.random.split(key)
    config = _small_config(flow_train_batch_size=0)
    particles = jax.random.multivariate_normal(
        data_key, jnp.array([1.0, 1.0]), jnp.array([[1.0, 0.7], [0.7, 1.0]]), (256,)
    )
    flow = build_flow(_N_DIMS, config, flow_key)
    optimizer = build_optimizer(flow, config)

    _, _, _, loss_history = train_flow(key, flow, optimizer, particles, config)

    loss_history = np.asarray(loss_history)
    assert loss_history.shape == (config.flow_n_epochs,)
    # Full-batch means every epoch's recorded loss is a true epoch loss, enabling this check.
    assert float(loss_history[-1]) < float(loss_history[0])


def test_training_particles_from_weighted_resamples_to_equal_weight():
    key = jax.random.key(7)
    particles = jnp.arange(10.0).reshape(10, 1)
    weights = jnp.zeros(10).at[3].set(1.0)  # all mass on particle index 3
    resampled = training_particles_from_weighted(key, particles, weights, 10)
    assert resampled.shape == (10, 1)
    np.testing.assert_allclose(np.asarray(resampled), np.full((10, 1), 3.0))


# Kernel-level invariance test (highest-value correctness check)


def test_preconditioned_kernel_preserves_correlated_gaussian_stationarity():
    """Run only the preconditioned mutation kernel (no tempering/resampling) against
    a known strongly-correlated Gaussian, starting particles already at
    stationarity, and check the empirical mean/covariance stay close to the truth.

    This exercises the full to_latent/from_latent/logdensity_latent/kernel pipeline
    together and would catch a sign error that the isolated mapping tests could miss.
    """
    from jimgw.samplers.blackjax.smc.precondition import build_preconditioned_mcmc_step

    mu = jnp.array([1.0, -2.0])
    sigma = jnp.array([[2.0, 1.5], [1.5, 3.0]])
    precision = jnp.linalg.inv(sigma)

    def logdensity_fn(x):
        diff = x - mu
        return -0.5 * diff @ precision @ diff

    key = jax.random.key(8)
    flow_key, data_key, init_key, run_key = jax.random.split(key, 4)

    config = _small_config(flow_n_epochs=200)
    training_particles = jax.random.multivariate_normal(data_key, mu, sigma, (1000,))
    flow, _, _ = _build_and_train_flow(flow_key, training_particles, config)

    flat_params, unravel_fn, static = partition_flow(flow)
    step_fn = build_preconditioned_mcmc_step(
        unravel_fn, static, to_position_wrapper(None, _N_DIMS)
    )

    n_particles = 800
    # A generous step count so a residual per-step bias would clearly blow through the tolerances below.
    n_steps = 150
    init_positions = jax.random.multivariate_normal(init_key, mu, sigma, (n_particles,))

    latent_cov = jnp.eye(_N_DIMS) * (2.38**2 / _N_DIMS)

    def run_chain(key, position):
        state = random_walk.init(position, logdensity_fn)

        def body(state, key):
            new_state, info = step_fn(
                key, state, logdensity_fn, latent_cov, flat_params
            )
            return new_state, info

        keys = jax.random.split(key, n_steps)
        final_state, _ = jax.lax.scan(body, state, keys)
        return final_state.position

    keys = jax.random.split(run_key, n_particles)
    final_positions = jax.vmap(run_chain)(keys, init_positions)
    final_positions = np.asarray(final_positions)

    empirical_mean = final_positions.mean(axis=0)
    empirical_cov = np.cov(final_positions, rowvar=False)

    # Tolerances are close to the finite-N MC sampling error (atol=0.2 ~ 3.5x SEM).
    np.testing.assert_allclose(empirical_mean, np.asarray(mu), atol=0.2)
    np.testing.assert_allclose(empirical_cov, np.asarray(sigma), atol=0.6, rtol=0.3)


@pytest.mark.parametrize(
    "periodic_index",
    [None, {0: (0.0, 1.0)}],
    ids=["no_periodic", "periodic_dim0"],
)
def test_preconditioned_kernel_rejection_is_exact_noop(periodic_index):
    """A rejected MH step must return the sampling-space position/logdensity bitwise
    unchanged from the input -- not merely close, and regardless of whether
    periodic wrapping is active.

    `from_latent(to_latent(x)) != x` exactly for a *trained* flow (its inverse
    root-solve is only approximate; see test_to_latent_from_latent_round_trip
    for the measured magnitude). On rejection `new_latent_state.position ==
    latent0` exactly, so naively reconstituting x from that (unchanged) latent
    position would silently displace a rejected particle by the flow's
    round-trip error on *every single kernel call* -- a systematic drift, not
    noise, compounding over the many (num_mcmc_steps) repeated calls per SMC
    iteration and invisible in the acceptance rate. This regression-tests the
    explicit jnp.where(info.is_accepted, ...) no-op guard in
    build_preconditioned_mcmc_step, including with a real (not no-op)
    position_wrapper, since rejection must stay a no-op either way.
    """
    from jimgw.samplers.blackjax.smc.precondition import build_preconditioned_mcmc_step

    mu = jnp.array([1.0, -2.0])
    sigma = jnp.array([[2.0, 1.5], [1.5, 3.0]])
    precision = jnp.linalg.inv(sigma)

    def logdensity_fn(x):
        diff = x - mu
        return -0.5 * diff @ precision @ diff

    key = jax.random.key(11)
    flow_key, data_key, init_key, run_key = jax.random.split(key, 4)
    config = _small_config(flow_n_epochs=200)
    training_particles = jax.random.multivariate_normal(data_key, mu, sigma, (1000,))
    flow, _, _ = _build_and_train_flow(flow_key, training_particles, config)
    flat_params, unravel_fn, static = partition_flow(flow)
    step_fn = build_preconditioned_mcmc_step(
        unravel_fn, static, to_position_wrapper(periodic_index, _N_DIMS)
    )

    n_particles = 300
    n_steps = 25
    init_positions = jax.random.multivariate_normal(init_key, mu, sigma, (n_particles,))
    # A moderately large proposal covariance (latent space is ~unit-scale) so a substantial fraction of steps are rejected.
    cov = jnp.eye(_N_DIMS) * (5.0**2)

    def run_chain(key, position):
        state = random_walk.init(position, logdensity_fn)

        def body(state, key):
            new_state, info = step_fn(key, state, logdensity_fn, cov, flat_params)
            return new_state, (
                state.position,
                new_state.position,
                new_state.logdensity,
                info.is_accepted,
            )

        keys = jax.random.split(key, n_steps)
        _, trajectory = jax.lax.scan(body, state, keys)
        return trajectory

    keys = jax.random.split(run_key, n_particles)
    before, after, after_logdensity, accepted = jax.vmap(run_chain)(
        keys, init_positions
    )
    before = np.asarray(before)
    after = np.asarray(after)
    after_logdensity = np.asarray(after_logdensity)
    accepted = np.asarray(accepted)

    n_rejected = int((~accepted).sum())
    n_accepted = int(accepted.sum())
    assert n_rejected > 0 and n_accepted > 0, (
        "expected a mix of accept/reject to exercise both branches; "
        f"got accepted={n_accepted}, rejected={n_rejected} -- adjust cov"
    )
    np.testing.assert_array_equal(after[~accepted], before[~accepted])
    # Sanity check the flip side too, so a step function that always no-ops wouldn't pass.
    assert not np.allclose(after[accepted], before[accepted])

    # The carried `logdensity` must be the true sampling-space log-density on both branches.
    direct_logdensity = jax.vmap(jax.vmap(logdensity_fn))(after)
    np.testing.assert_allclose(
        after_logdensity, np.asarray(direct_logdensity), atol=1e-8, rtol=1e-8
    )


def test_preconditioned_kernel_periodic_wrap_rescues_out_of_range_proposals():
    """The wrap mechanism must actually engage, not just be present as dead code.

    Uses a hard-cutoff target (like UniformPrior's -inf outside its bounds) on a
    periodic dimension, so a proposal that maps outside [0, 1) is rejected without
    wrapping but can be accepted with wrapping -- the same rescue pocoMC's own
    preconditioned kernel relies on. Compares acceptance rate with vs. without a
    real position_wrapper, same keys/flow, to directly demonstrate the wrap fires
    and changes the outcome (not just that it type-checks).
    """
    from jimgw.samplers.blackjax.smc.precondition import build_preconditioned_mcmc_step

    def logdensity_fn(x):
        in_bounds = jnp.logical_and(x[0] >= 0.0, x[0] < 1.0)
        return jnp.where(in_bounds, -0.5 * x[1] ** 2, -jnp.inf)

    key = jax.random.key(12)
    flow_key, data_key, run_key = jax.random.split(key, 3)
    config = _small_config(flow_n_epochs=200)
    training_particles = jnp.stack(
        [
            jax.random.uniform(data_key, (1000,)),
            jax.random.normal(jax.random.fold_in(data_key, 1), (1000,)),
        ],
        axis=-1,
    )
    flow, _, _ = _build_and_train_flow(flow_key, training_particles, config)
    flat_params, unravel_fn, static = partition_flow(flow)

    n_particles = 500
    n_steps = 15
    init_positions = training_particles[:n_particles]
    # Large relative to the flow's ~unit-scale latent space, so many proposals map outside [0, 1).
    cov = jnp.eye(_N_DIMS) * (3.0**2)

    def run_chain(step_fn, key, position):
        state = random_walk.init(position, logdensity_fn)

        def body(state, key):
            new_state, info = step_fn(key, state, logdensity_fn, cov, flat_params)
            return new_state, (new_state.position, info.is_accepted)

        keys = jax.random.split(key, n_steps)
        _, (positions, accepted) = jax.lax.scan(body, state, keys)
        return positions, accepted

    keys = jax.random.split(run_key, n_particles)

    wrapped_step = build_preconditioned_mcmc_step(
        unravel_fn, static, to_position_wrapper({0: (0.0, 1.0)}, _N_DIMS)
    )
    wrapped_positions, wrapped_accepted = jax.vmap(
        lambda k, p: run_chain(wrapped_step, k, p)
    )(keys, init_positions)

    unwrapped_step = build_preconditioned_mcmc_step(
        unravel_fn, static, to_position_wrapper(None, _N_DIMS)
    )
    _, unwrapped_accepted = jax.vmap(lambda k, p: run_chain(unwrapped_step, k, p))(
        keys, init_positions
    )

    wrapped_rate = float(np.asarray(wrapped_accepted).mean())
    unwrapped_rate = float(np.asarray(unwrapped_accepted).mean())
    assert wrapped_rate > unwrapped_rate + 0.05, (
        f"expected wrapping to rescue out-of-range proposals (wrapped={wrapped_rate:.3f}, "
        f"unwrapped={unwrapped_rate:.3f})"
    )

    # The canonical-range invariant: every reported position stays in [0, 1) on dim 0.
    wrapped_dim0 = np.asarray(wrapped_positions)[..., 0]
    assert np.all(wrapped_dim0 >= 0.0) and np.all(wrapped_dim0 < 1.0)


def test_preconditioned_kernel_preserves_circular_target_stationarity():
    """Smoke test only: a genuinely periodic (cosine-based) target's mass, straddling
    the wrap seam, stays put over many steps of the periodic-aware kernel.

    This does NOT discriminate the known small bias from naive wrapping (see
    build_preconditioned_mcmc_step's docstring) -- tolerances here are loose,
    matching the existing correlated-Gaussian stationarity test's role as a wiring
    check, not a proof of exact invariance.

    The target's peak is placed away from the wrap seam (0 == 2*pi), not at it:
    concentrating it exactly at the seam forces even a well-trained flow to model
    one continuous lobe as two disconnected halves at opposite ends of its input
    range -- a flow-capacity artifact, not a periodic-wrap bug, confirmed by
    trying it (a large, spurious-looking drift that disappears once the peak
    moves away from the seam). "Straddling the seam" only needs to happen in the
    tails, which this placement still allows over many steps.
    """
    from jimgw.samplers.blackjax.smc.precondition import build_preconditioned_mcmc_step

    two_pi = 2 * jnp.pi
    kappa = 8.0  # concentration; mass spread ~ 1/sqrt(kappa) radians
    mu_angle = jnp.pi
    mu1, sigma1 = 0.5, 0.3

    def logdensity_fn(x):
        return kappa * jnp.cos(x[0] - mu_angle) - 0.5 * ((x[1] - mu1) / sigma1) ** 2

    key = jax.random.key(13)
    flow_key, data_key, init_key, run_key = jax.random.split(key, 4)
    config = _small_config(flow_n_epochs=200)
    # Exact von Mises samples so the flow trains on logdensity_fn's true distribution.
    rng = np.random.default_rng(42)
    angle = jnp.asarray(np.mod(rng.vonmises(float(mu_angle), kappa, size=1000), two_pi))
    dim1 = mu1 + sigma1 * jax.random.normal(jax.random.fold_in(data_key, 1), (1000,))
    training_particles = jnp.stack([angle, dim1], axis=-1)
    flow, _, _ = _build_and_train_flow(flow_key, training_particles, config)
    flat_params, unravel_fn, static = partition_flow(flow)
    step_fn = build_preconditioned_mcmc_step(
        unravel_fn, static, to_position_wrapper({0: (0.0, two_pi)}, _N_DIMS)
    )

    n_particles = 800
    n_steps = 100
    init_angle = jnp.asarray(
        np.mod(rng.vonmises(float(mu_angle), kappa, size=n_particles), two_pi)
    )
    init_dim1 = mu1 + sigma1 * jax.random.normal(
        jax.random.fold_in(init_key, 1), (n_particles,)
    )
    init_positions = jnp.stack([init_angle, init_dim1], axis=-1)
    latent_cov = jnp.eye(_N_DIMS) * (0.15**2)

    def run_chain(key, position):
        state = random_walk.init(position, logdensity_fn)

        def body(state, key):
            new_state, info = step_fn(
                key, state, logdensity_fn, latent_cov, flat_params
            )
            return new_state, info

        keys = jax.random.split(key, n_steps)
        final_state, _ = jax.lax.scan(body, state, keys)
        return final_state.position

    keys = jax.random.split(run_key, n_particles)
    final_positions = np.asarray(jax.vmap(run_chain)(keys, init_positions))

    # Circular mean via the resultant vector; loose tolerances (see docstring).
    resultant = np.mean(np.exp(1j * final_positions[:, 0]))
    assert abs(resultant) > 0.5, "expected the circular marginal to stay concentrated"
    mean_angle = float(np.angle(resultant)) % (2 * np.pi)
    angular_distance = min(
        abs(mean_angle - float(mu_angle)), two_pi - abs(mean_angle - float(mu_angle))
    )
    assert angular_distance < 0.3
    assert abs(float(final_positions[:, 1].mean()) - mu1) < 0.15


def test_flow_params_channel_does_not_force_recompile():
    """Feeding a retrained flow's weights through blackjax's shared-parameter
    channel (as each SMC mode class's ``_run`` does between SMC iterations) must not
    force JIT retracing — only the parameter *value* should change, not the
    pytree structure the jitted step function was traced against.
    """
    from blackjax import adaptive_tempered_smc, inner_kernel_tuning, rmh
    from blackjax.smc.base import extend_params
    from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
    from blackjax.smc.resampling import systematic

    from jimgw.samplers.blackjax.smc.precondition import (
        build_preconditioned_mcmc_step,
        training_particles_from_weighted,
    )

    config = _small_config(flow_n_epochs=5)
    key = jax.random.key(9)
    flow_key, part_key = jax.random.split(key)
    flow = build_flow(_N_DIMS, config, flow_key)
    optimizer = build_optimizer(flow, config)
    flat_params, unravel_fn, static = partition_flow(flow)
    mcmc_step = build_preconditioned_mcmc_step(
        unravel_fn, static, to_position_wrapper(None, _N_DIMS)
    )

    def log_prior_fn(x):
        return jnp.sum(jax.scipy.stats.uniform.logpdf(x, 0.0, 1.0))

    def log_likelihood_fn(x):
        return -0.5 * jnp.sum((x - 0.5) ** 2) / 0.01

    initial_particles = jax.random.uniform(part_key, (100, _N_DIMS))
    initial_cov = jnp.eye(_N_DIMS) * 0.01

    def mcmc_parameter_update_fn(_key, state, _info):
        return extend_params({"cov": jnp.atleast_2d(jnp.cov(state.particles.T))})

    smc_algorithm = inner_kernel_tuning(
        smc_algorithm=adaptive_tempered_smc,
        logprior_fn=log_prior_fn,
        loglikelihood_fn=log_likelihood_fn,
        mcmc_step_fn=mcmc_step,
        mcmc_init_fn=rmh.init,
        resampling_fn=systematic,
        mcmc_parameter_update_fn=mcmc_parameter_update_fn,
        initial_parameter_value=extend_params(
            {"cov": initial_cov, "flow_params": flat_params}
        ),
        num_mcmc_steps=5,
        target_ess=0.5,
        batch_size=0,
    )

    state = smc_algorithm.init(
        initial_particles,
        extend_params({"cov": initial_cov, "flow_params": flat_params}),
    )
    run_step = jax.jit(smc_algorithm.step)
    rng_key = jax.random.key(10)
    cache_sizes = []

    # The very first call retraces once (a pre-existing ap/at characteristic, not something preconditioning introduces); what matters is the cache size plateaus afterward instead of growing with retraining.
    for i in range(8):
        rng_key, step_key = jax.random.split(rng_key)
        state, _info = run_step(step_key, state)
        cache_sizes.append(run_step._cache_size())
        if state.sampler_state.tempering_param >= 1.0:
            break
        sampler_state = state.sampler_state
        current_covariance = state.parameter_override["cov"][0]
        rng_key, resample_key = jax.random.split(rng_key)
        training_particles = training_particles_from_weighted(
            resample_key,
            sampler_state.particles,
            sampler_state.weights,
            sampler_state.particles.shape[0],
        )
        rng_key, flow, optimizer, _ = train_flow(
            rng_key, flow, optimizer, training_particles, config
        )
        flat_params = flatten_flow(flow)
        state = StateWithParameterOverride(
            sampler_state,
            extend_params({"cov": current_covariance, "flow_params": flat_params}),
        )

    assert len(cache_sizes) >= 3, (
        "test converged in too few iterations to exercise repeated retraining; "
        f"cache_sizes={cache_sizes}"
    )
    assert cache_sizes[-1] <= 2, f"cache size grew unbounded: {cache_sizes}"
    assert cache_sizes[-1] == cache_sizes[1], (
        f"cache size still changing after the first iteration: {cache_sizes} — "
        "retraining the flow is forcing a retrace on (at least) one later iteration"
    )
