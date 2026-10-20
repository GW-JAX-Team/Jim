import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.samplers.periodic import (
    to_prior_space_proposal,
    to_recentring_map,
    to_unit_cube_stepper,
)

# Index-based periodic specs matching PARAMS_3D = ("alpha", "phase_c", "psi")
# phase_c → index 1, psi → index 2
PERIODIC_3D = {1: (0.0, 2 * jnp.pi), 2: (0.0, jnp.pi)}


# ----------------------------------------------------------------------
# Index and bound validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("index", [1.0, "1"])
@pytest.mark.parametrize("adapter", [to_unit_cube_stepper, to_recentring_map])
def test_periodic_adapters_reject_noninteger_indices(adapter, index):
    spec = [index] if adapter is to_unit_cube_stepper else {index: (0.0, 1.0)}
    with pytest.raises(TypeError, match="must be integers"):
        adapter(spec, 3)


@pytest.mark.parametrize("index", [-1, 3])
@pytest.mark.parametrize("adapter", [to_unit_cube_stepper, to_recentring_map])
def test_periodic_adapters_reject_out_of_range_indices(adapter, index):
    spec = [index] if adapter is to_unit_cube_stepper else {index: (0.0, 1.0)}
    with pytest.raises(ValueError, match="out of bounds"):
        adapter(spec, 3)


def test_periodic_adapters_accept_numpy_integer_dimensions_and_indices():
    stepper = to_unit_cube_stepper([np.int64(1)], np.int32(3))
    _, forward, inverse = to_recentring_map({np.int64(1): (0.0, 1.0)}, np.int32(3))
    position = jnp.array([0.2, 0.9, 0.3])
    direction = jnp.array([0.1, 0.4, 0.1])
    expected = jnp.array([0.3, 0.3, 0.4])
    assert jnp.allclose(stepper(position, direction, 1.0), expected)
    cuts = jnp.array([0.0, 0.5, 0.0])
    assert jnp.allclose(inverse(forward(position, cuts), cuts)[0], position)


@pytest.mark.parametrize("bound", [jnp.nan, jnp.inf, -jnp.inf])
@pytest.mark.parametrize("side", [0, 1])
def test_periodic_bounds_must_be_finite(bound, side):
    bounds = [0.0, 1.0]
    bounds[side] = bound
    with pytest.raises(ValueError, match="must be finite"):
        to_recentring_map({1: tuple(bounds)}, 3)


def test_periodic_interval_must_be_finite():
    with pytest.raises(ValueError, match="interval.*finite"):
        to_recentring_map({1: (-1e308, 1e308)}, 3)


# ----------------------------------------------------------------------
# to_unit_cube_stepper — flat arrays: [alpha, phase_c, psi]
# ----------------------------------------------------------------------


def _pos(alpha=0.3, phase_c=0.6, psi=0.9):
    return jnp.array([alpha, phase_c, psi])


def _dir(alpha=0.1, phase_c=0.7, psi=-0.5):
    return jnp.array([alpha, phase_c, psi])


def test_unit_cube_stepper_none_periodic():
    stepper = to_unit_cube_stepper(None, 3)
    pos = _pos()
    direction = _dir()
    result = stepper(pos, direction, 1.0)
    expected = pos + direction
    assert jnp.allclose(result, expected)


def test_unit_cube_stepper_wraps_periodic():
    stepper = to_unit_cube_stepper([1], 3)  # index 1 = phase_c
    pos = jnp.array([0.3, 0.9, 0.1])
    direction = jnp.array([0.0, 0.5, 0.0])
    # phase_c (index 1): 0.9 + 0.5 = 1.4 → mod 1.0 = 0.4
    result = stepper(pos, direction, 1.0)
    assert float(result[1]) == pytest.approx(0.4, abs=1e-6)
    # alpha and psi are not periodic
    assert float(result[0]) == pytest.approx(0.3, abs=1e-6)
    assert float(result[2]) == pytest.approx(0.1, abs=1e-6)


def test_unit_cube_stepper_no_wrapping_needed():
    stepper = to_unit_cube_stepper([1], 3)  # index 1 = phase_c
    pos = jnp.array([0.0, 0.1, 0.0])
    direction = jnp.array([0.0, 0.2, 0.0])
    result = stepper(pos, direction, 1.0)
    assert float(result[1]) == pytest.approx(0.3, abs=1e-6)


def test_unit_cube_stepper_empty_periodic():
    stepper = to_unit_cube_stepper([], 3)
    pos = _pos()
    direction = _dir()
    result = stepper(pos, direction, 1.0)
    assert jnp.allclose(result, pos + direction)


# ----------------------------------------------------------------------
# to_prior_space_proposal — flat arrays
# ----------------------------------------------------------------------


class _MockState:
    """Minimal stand-in for blackjax StateWithLogLikelihood."""

    def __init__(self, position):
        self.position = position
        self.loglikelihood = 0.5


def _mock_init_fn(x, loglikelihood_birth=None):
    return _MockState(position=x)


def test_prior_space_proposal_none_periodic():
    direction = jnp.array([0.1, 0.7, -0.5])
    proposal_factory = to_prior_space_proposal(
        None,
        3,
        lambda _rng_key, _position, _cov: direction,
    )
    pos = jnp.array([0.3, 6.0, 0.9])
    gen = proposal_factory(_mock_init_fn, 0.0, jnp.eye(3))
    slice_fn = gen(None, pos, None)
    new_state, _accepted = slice_fn(1.0)
    expected = pos + direction
    assert jnp.allclose(new_state.position, expected)


def test_prior_space_proposal_wraps_phase_c():
    """Periodic dims are wrapped after a nonzero proposal move."""
    two_pi = 2 * jnp.pi
    direction = jnp.array([0.0, 1.0, 0.0])
    proposal_factory = to_prior_space_proposal(
        {1: (0.0, two_pi)},
        3,
        lambda _rng_key, _position, _cov: direction,
    )
    pos = jnp.array([0.0, 6.0, 0.0])
    gen = proposal_factory(_mock_init_fn, 0.0, jnp.eye(3))
    slice_fn = gen(None, pos, None)
    new_state, accepted = slice_fn(1.0)
    expected = float(jnp.mod(7.0, two_pi))
    assert bool(accepted) is True
    assert float(new_state.position[1]) == pytest.approx(expected, abs=1e-6)


def test_prior_space_proposal_invalid_bounds_raises():
    with pytest.raises(ValueError, match="hi > lo"):
        to_prior_space_proposal({1: (1.0, 0.0)}, 3, lambda *_args: _dir())


# ----------------------------------------------------------------------
# to_recentring_map: BlackJAX SMC
# ----------------------------------------------------------------------


def test_recentring_cut_is_in_the_middle_of_the_widest_empty_arc():
    cuts_fn, _, _ = to_recentring_map(PERIODIC_3D, 3)
    phase_c = jnp.array([1.0, 1.3, 1.8, 2.0])  # empty arc from 2.0 to 1.0 + 2 pi
    particles = jnp.stack([jnp.zeros(4), phase_c, jnp.full(4, 0.5)], axis=1)
    cuts = cuts_fn(particles)
    assert float(cuts[0]) == 0.0
    assert float(cuts[1]) == pytest.approx(1.5 + jnp.pi)
    # psi: every particle at 0.5, so the empty arc is the whole period.
    assert float(cuts[2]) == pytest.approx(0.5 + 0.5 * jnp.pi)


def test_recentring_round_trip_and_jacobian():
    cuts_fn, forward, inverse = to_recentring_map(PERIODIC_3D, 3)
    key_a, key_p, key_q = jax.random.split(jax.random.key(0), 3)
    particles = jnp.stack(
        [
            jax.random.normal(key_a, (200,)),
            jax.random.uniform(key_p, (200,), maxval=2 * jnp.pi),
            jax.random.uniform(key_q, (200,), maxval=jnp.pi),
        ],
        axis=1,
    )
    cuts = cuts_fn(particles)
    u = jax.vmap(forward, in_axes=(0, None))(particles, cuts)
    x, log_det = jax.vmap(inverse, in_axes=(0, None))(u, cuts)

    assert jnp.allclose(u[:, 0], particles[:, 0])
    assert float(jnp.max(jnp.abs(x - particles))) < 1e-12
    jacobian = jax.vmap(jax.jacfwd(lambda v: inverse(v, cuts)[0]))(u)
    _, autodiff_log_det = jnp.linalg.slogdet(jacobian)
    assert jnp.allclose(log_det, autodiff_log_det, atol=1e-10)


def test_recentring_keeps_two_modes_half_a_period_apart_whole():
    # One mode sits on the seam at 0 = 2 pi, the other at pi.
    cuts_fn, forward, _ = to_recentring_map({0: (0.0, 2 * jnp.pi)}, 1)
    key_mode, key_noise = jax.random.split(jax.random.key(1))
    in_seam_mode = jax.random.bernoulli(key_mode, 0.6, (2000,))
    centre = jnp.where(in_seam_mode, 0.0, jnp.pi)
    theta = jnp.mod(centre + 0.3 * jax.random.normal(key_noise, (2000,)), 2 * jnp.pi)
    particles = theta[:, None]
    u = jax.vmap(forward, in_axes=(0, None))(particles, cuts_fn(particles))[:, 0]

    seam_mode, other_mode = u[in_seam_mode], u[~in_seam_mode]
    separated = (seam_mode.max() < other_mode.min()) | (
        other_mode.max() < seam_mode.min()
    )
    assert bool(separated)


def test_recentring_without_periodic_dimensions_is_the_identity():
    cuts_fn, forward, inverse = to_recentring_map(None, 2)
    particles = jnp.array([[0.3, -1.0], [2.0, 4.0]])
    cuts = cuts_fn(particles)
    x, log_det = inverse(forward(particles[0], cuts), cuts)
    assert jnp.array_equal(cuts, jnp.zeros(2))
    assert jnp.array_equal(x, particles[0])
    assert float(log_det) == 0.0


@pytest.mark.parametrize("bounds", [(2.0, 1.0), (1.0, 1.0)])
def test_recentring_map_invalid_bounds_raises(bounds):
    with pytest.raises(ValueError, match="hi > lo"):
        to_recentring_map({1: bounds}, 3)
