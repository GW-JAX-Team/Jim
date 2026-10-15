import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.samplers.periodic import (
    to_displacement_wrapper,
    to_prior_space_proposal,
    to_unit_cube_stepper,
)

# Index-based periodic specs matching PARAMS_3D = ("alpha", "phase_c", "psi")
# phase_c → index 1, psi → index 2
PERIODIC_3D = {1: (0.0, 2 * jnp.pi), 2: (0.0, jnp.pi)}


# ----------------------------------------------------------------------
# Index and bound validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("index", [1.0, "1"])
@pytest.mark.parametrize("adapter", [to_unit_cube_stepper, to_displacement_wrapper])
def test_periodic_adapters_reject_noninteger_indices(adapter, index):
    spec = [index] if adapter is to_unit_cube_stepper else {index: (0.0, 1.0)}
    with pytest.raises(TypeError, match="must be integers"):
        adapter(spec, 3)


@pytest.mark.parametrize("index", [-1, 3])
@pytest.mark.parametrize("adapter", [to_unit_cube_stepper, to_displacement_wrapper])
def test_periodic_adapters_reject_out_of_range_indices(adapter, index):
    spec = [index] if adapter is to_unit_cube_stepper else {index: (0.0, 1.0)}
    with pytest.raises(ValueError, match="out of bounds"):
        adapter(spec, 3)


def test_periodic_adapters_accept_numpy_integer_dimensions_and_indices():
    stepper = to_unit_cube_stepper([np.int64(1)], np.int32(3))
    wrapper = to_displacement_wrapper({np.int64(1): (0.0, 1.0)}, np.int32(3))
    position = jnp.array([0.2, 0.9, 0.3])
    direction = jnp.array([0.1, 0.4, 0.1])
    expected = jnp.array([0.3, 0.3, 0.4])
    assert jnp.allclose(stepper(position, direction, 1.0), expected)
    assert jnp.allclose(position + wrapper(direction, position), expected)


@pytest.mark.parametrize("bound", [jnp.nan, jnp.inf, -jnp.inf])
@pytest.mark.parametrize("side", [0, 1])
def test_periodic_bounds_must_be_finite(bound, side):
    bounds = [0.0, 1.0]
    bounds[side] = bound
    with pytest.raises(ValueError, match="must be finite"):
        to_displacement_wrapper({1: tuple(bounds)}, 3)


def test_periodic_interval_must_be_finite():
    with pytest.raises(ValueError, match="interval.*finite"):
        to_displacement_wrapper({1: (-1e308, 1e308)}, 3)


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
# to_displacement_wrapper — flat arrays
# ----------------------------------------------------------------------


def test_displacement_wrapper_none_periodic():
    wrapper = to_displacement_wrapper(None, 3)
    proposed = jnp.array([0.1, 0.5, -0.2])
    current = jnp.array([0.3, 2.0, 1.0])
    result = wrapper(proposed, current)
    assert jnp.allclose(result, proposed)


def test_displacement_wrapper_wraps_periodic():
    two_pi = 2 * jnp.pi
    wrapper = to_displacement_wrapper({1: (0.0, two_pi)}, 3)  # index 1 = phase_c
    # current phase_c = 5.8, displacement = 1.0 → new = 6.8 → wrapped = mod(6.8, 2π) ≈ 0.517
    current_val = 5.8
    disp_val = 1.0
    current = jnp.array([0.0, current_val, 0.0])
    proposed = jnp.array([0.0, disp_val, 0.0])
    result = wrapper(proposed, current)
    wrapped_pos = float(jnp.mod(current_val + disp_val, two_pi))
    expected_disp = wrapped_pos - current_val
    assert float(result[1]) == pytest.approx(expected_disp, abs=1e-6)
    # Non-periodic params unchanged
    assert float(result[0]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("bounds", [(2.0, 1.0), (1.0, 1.0)])
def test_displacement_wrapper_invalid_bounds_raises(bounds):
    with pytest.raises(ValueError, match="hi > lo"):
        to_displacement_wrapper({1: bounds}, 3)
