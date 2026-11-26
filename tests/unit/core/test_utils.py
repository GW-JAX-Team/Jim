"""Tests for the shared core utilities."""

import jax
import jax.numpy as jnp
import pytest

from jimgw.core.utils import (
    carte_to_spherical_angles,
    round_up_to_power_of_two,
    safe_arctan2,
    safe_polar_angle,
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [(-3, 1), (0, 1), (1, 1), (2, 2), (3, 4), (8, 8), (9, 16)],
)
def test_round_up_to_power_of_two(n, expected):
    result = round_up_to_power_of_two(n)
    assert result == expected
    assert type(result) is int


class TestSphericalAngles:
    @pytest.mark.parametrize(
        ("point", "theta"),
        [
            ((1e-10, 0.0, 1.0), 1e-10),
            ((0.0, 1e-10, -1.0), jnp.pi - 1e-10),
            ((1e200, 1e200, 1e200), jnp.arctan(jnp.sqrt(2.0))),
            ((3.0, 4.0, 0.0), jnp.pi / 2),
        ],
    )
    def test_the_polar_angle_is_accurate_near_the_poles_and_at_large_radii(
        self, point, theta
    ):
        assert jnp.isclose(safe_polar_angle(*point), theta, rtol=1e-15, atol=0)

    @pytest.mark.parametrize(
        ("point", "angles"),
        [
            ((0.0, 0.0, 1.0), (0.0, 0.5)),
            ((0.0, 0.0, -1.0), (jnp.pi, 0.5)),
            ((0.0, 0.0, 0.0), (jnp.pi / 2, 0.5)),
            ((1e-16, 0.0, 1.0), (0.0, 0.5)),
        ],
    )
    def test_points_on_the_axis_get_the_default_azimuth(self, point, angles):
        theta, phi = carte_to_spherical_angles(*point, default_value=0.5)
        assert theta == angles[0]
        assert phi == angles[1]

    @pytest.mark.parametrize(
        "point", [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1e-10, 0.0, 1.0)]
    )
    def test_gradients_are_finite_on_and_near_the_axis(self, point):
        def angles(v):
            return jnp.stack(carte_to_spherical_angles(*v))

        v = jnp.array(point)
        assert jnp.isfinite(jax.jacfwd(angles)(v)).all()
        assert jnp.isfinite(jax.jacrev(angles)(v)).all()
        assert jnp.isfinite(jax.grad(lambda w: safe_arctan2(w[1], w[0]))(v)).all()
