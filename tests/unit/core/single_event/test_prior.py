"""Unit tests for single-event priors."""

import jax
import jax.numpy as jnp

from jimgw.core.single_event.prior import UniformComponentChirpMassPrior
from tests.utils import assert_all_finite, assert_all_in_range


class TestUniformComponentChirpMassPrior:
    """``UniformComponentChirpMassPrior`` is ``PowerLawPrior`` specialised to
    ``alpha = 1`` on ``M_c`` -- the power-law machinery itself is covered in
    ``tests/unit/core/test_prior.py::test_power_law``.
    """

    def test_uniform_component_chirp_mass(self):
        xmin, xmax = 1.0, 5.0
        p = UniformComponentChirpMassPrior(xmin, xmax)

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["M_c"])
        assert_all_in_range(samples["M_c"], xmin, xmax)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support (alpha = 1 closed form; use valid
        # range (0, 1] for base, excluding 0 as it maps to the boundary where
        # log_prob = -inf)
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.001, 1.0, 1000)[None])
        y = jax.vmap(p.transform)(x)
        expected = jnp.log(y["M_c"]) + jnp.log(2.0) - jnp.log(xmax**2 - xmin**2)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), expected)

        # Check log_prob is -inf outside the support
        x_outside = p.add_name(jnp.array([xmin - 0.01, xmax + 1.0])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))
