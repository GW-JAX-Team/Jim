import math

import jax.numpy as jnp
import pytest

from jimgw.samplers.periodic import (
    to_displacement_wrapper,
    to_prior_space_stepper,
    to_unit_cube_stepper,
)


# Index-based periodic specs matching PARAMS_3D = ("alpha", "phase_c", "psi")
# phase_c → index 1, psi → index 2
PERIODIC_3D = {1: (0.0, 2 * math.pi), 2: (0.0, math.pi)}


# ---------------------------------------------------------------------------
# to_unit_cube_stepper — flat arrays: [alpha, phase_c, psi]
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# to_prior_space_stepper — flat arrays
# ---------------------------------------------------------------------------


def test_prior_space_stepper_none_periodic():
    stepper = to_prior_space_stepper(None, 3)
    pos = _pos()
    direction = _dir()
    result, accepted = stepper(pos, direction, 1.0)
    assert accepted is True
    assert jnp.allclose(result, pos + direction)


def test_prior_space_stepper_wraps_phase_c():
    two_pi = 2 * math.pi
    stepper = to_prior_space_stepper({1: (0.0, two_pi)}, 3)  # index 1 = phase_c
    pos = jnp.array([0.0, 6.0, 0.0])
    direction = jnp.array([0.0, 1.0, 0.0])
    result, accepted = stepper(pos, direction, 1.0)
    # 6.0 + 1.0 = 7.0 → mod 2π ≈ 0.717
    expected = float(jnp.mod(7.0, two_pi))
    assert float(result[1]) == pytest.approx(expected, abs=1e-6)
    assert accepted is True


def test_prior_space_stepper_returns_accepted_bool():
    stepper = to_prior_space_stepper({1: (0.0, 2 * math.pi)}, 3)
    _, accepted = stepper(_pos(), _dir(), 0.1)
    assert accepted is True


def test_prior_space_stepper_invalid_bounds_raises():
    with pytest.raises(ValueError, match="hi > lo"):
        to_prior_space_stepper({1: (1.0, 0.0)}, 3)


# ---------------------------------------------------------------------------
# to_displacement_wrapper — flat arrays
# ---------------------------------------------------------------------------


def test_displacement_wrapper_none_periodic():
    wrapper = to_displacement_wrapper(None, 3)
    proposed = jnp.array([0.1, 0.5, -0.2])
    current = jnp.array([0.3, 2.0, 1.0])
    result = wrapper(proposed, current)
    assert jnp.allclose(result, proposed)


def test_displacement_wrapper_wraps_periodic():
    two_pi = 2 * math.pi
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


def test_displacement_wrapper_invalid_bounds_raises():
    with pytest.raises(ValueError, match="hi > lo"):
        to_displacement_wrapper({1: (2.0, 1.0)}, 3)
