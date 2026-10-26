import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest
from scipy import stats

from jimgw.core.base import LikelihoodBase
from jimgw.core.jim import Jim
from jimgw.core.prior import (
    BoundedMixin,
    CombinePrior,
    CosinePrior,
    GaussianPrior,
    LogisticDistribution,
    PowerLawPrior,
    Prior,
    RayleighPrior,
    SequentialTransformPrior,
    SinePrior,
    StandardNormalDistribution,
    UniformDistribution,
    UniformPrior,
    UniformSpherePrior,
    find_specific_prior,
)
from jimgw.samplers.config import (
    BlackJAXNSSConfig,
    BlackJAXSMCConfig,
)
from tests.utils import assert_all_finite, assert_all_in_range


class TestPriorLookup:
    def test_finds_component_prior_in_nested_combine_prior(self):
        mc_prior = UniformPrior(10.0, 30.0, ["M_c"])
        tc_prior = UniformPrior(-0.1, 0.1, ["t_c"])
        prior = CombinePrior([CombinePrior([mc_prior]), tc_prior])

        assert find_specific_prior(prior, "M_c") is mc_prior
        assert find_specific_prior(prior, "t_c") is tc_prior

    def test_finds_gaussian_component_and_returns_none_when_absent(self):
        mc_prior = GaussianPrior(20.0, 1.0, ["M_c"])
        prior = CombinePrior([mc_prior])

        assert find_specific_prior(prior, "M_c") is mc_prior
        assert find_specific_prior(prior, "t_c") is None


class TestPriorConstructorValidation:
    @pytest.mark.parametrize("parameter_names", [[], ["x", "y"]])
    @pytest.mark.parametrize(
        "prior_type, settings",
        [
            (LogisticDistribution, {}),
            (StandardNormalDistribution, {}),
            (UniformDistribution, {}),
            (UniformPrior, {"xmin": 0.0, "xmax": 1.0}),
            (GaussianPrior, {"mu": 0.0, "sigma": 1.0}),
            (SinePrior, {}),
            (CosinePrior, {}),
            (UniformSpherePrior, {}),
            (RayleighPrior, {"sigma": 1.0}),
            (PowerLawPrior, {"xmin": 1.0, "xmax": 2.0, "alpha": 2.0}),
        ],
    )
    def test_univariate_priors_require_exactly_one_parameter(
        self, prior_type, settings, parameter_names
    ):
        with pytest.raises(ValueError, match="1D|name of the vector"):
            prior_type(parameter_names=parameter_names, **settings)

    @pytest.mark.parametrize("base_prior", [[], [UniformPrior(0.0, 1.0, ["x"])] * 2])
    def test_sequential_prior_requires_exactly_one_base_prior(self, base_prior):
        with pytest.raises(ValueError, match="one base prior"):
            SequentialTransformPrior(base_prior, [])

    def test_validation_remains_enabled_with_python_optimization(self):
        program = """
from jimgw.core.prior import UniformPrior, GaussianPrior, SequentialTransformPrior
from jimgw.cli._config import GaussianSpec, PriorConfig, SamplingConfig
from jimgw.cli._prior import adapt_prior_for_ns_time

cases = [
    lambda: UniformPrior(0.0, 1.0, []),
    lambda: UniformPrior(1.0, 0.0, ['x']),
    lambda: GaussianPrior(0.0, -1.0, ['x']),
    lambda: SequentialTransformPrior([], []),
    lambda: adapt_prior_for_ns_time(
        PriorConfig({'t_c': GaussianSpec(loc=0.0, scale=0.1)}),
        SamplingConfig(),
    ),
]
for index, make in enumerate(cases):
    try:
        make()
    except ValueError:
        continue
    raise RuntimeError(f'Invalid input {index} was accepted under python -O')
"""
        result = subprocess.run(
            [sys.executable, "-O", "-c", program],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
    @pytest.mark.parametrize(
        "prior_type, settings, field",
        [
            (UniformPrior, {"xmin": -1.0, "xmax": 2.0}, "xmin"),
            (UniformPrior, {"xmin": -1.0, "xmax": 2.0}, "xmax"),
            (GaussianPrior, {"mu": 0.0, "sigma": 1.0}, "mu"),
            (GaussianPrior, {"mu": 0.0, "sigma": 1.0}, "sigma"),
            (RayleighPrior, {"sigma": 1.0}, "sigma"),
            (UniformSpherePrior, {"max_mag": 1.0}, "max_mag"),
            (PowerLawPrior, {"xmin": 1.0, "xmax": 2.0, "alpha": 2.0}, "xmin"),
            (PowerLawPrior, {"xmin": 1.0, "xmax": 2.0, "alpha": 2.0}, "xmax"),
            (PowerLawPrior, {"xmin": 1.0, "xmax": 2.0, "alpha": 2.0}, "alpha"),
        ],
    )
    def test_numeric_prior_settings_must_be_finite(
        self, prior_type, settings, field, bad
    ):
        with pytest.raises(ValueError, match="finite"):
            prior_type(parameter_names=["x"], **{**settings, field: bad})

    @pytest.mark.parametrize(
        "prior_type, settings",
        [
            (UniformPrior, {"xmin": 1.0, "xmax": 1.0}),
            (UniformPrior, {"xmin": 2.0, "xmax": 1.0}),
            (GaussianPrior, {"mu": 0.0, "sigma": 0.0}),
            (GaussianPrior, {"mu": 0.0, "sigma": -1.0}),
            (RayleighPrior, {"sigma": 0.0}),
            (RayleighPrior, {"sigma": -1.0}),
            (UniformSpherePrior, {"max_mag": 0.0}),
            (UniformSpherePrior, {"max_mag": -1.0}),
            (PowerLawPrior, {"xmin": 0.0, "xmax": 2.0, "alpha": 2.0}),
            (PowerLawPrior, {"xmin": -1.0, "xmax": 2.0, "alpha": 2.0}),
            (PowerLawPrior, {"xmin": 2.0, "xmax": 1.0, "alpha": 2.0}),
        ],
    )
    def test_invalid_domains_raise_value_error(self, prior_type, settings):
        with pytest.raises(ValueError):
            prior_type(parameter_names=["x"], **settings)

    def test_uniform_interval_width_must_not_overflow(self):
        with pytest.raises(ValueError, match="interval width must be finite"):
            UniformPrior(-1e308, 1e308, ["x"])

    def test_rayleigh_retains_its_valid_unbounded_support(self):
        prior = RayleighPrior(1.0, ["x"])
        assert prior.get_bounds() == (0.0, jnp.inf)
        assert jnp.isfinite(prior.log_prob({"x": jnp.asarray(1.0)}))


class TestUnivariatePrior:
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("direction", [-1.0, 1.0])
    def test_logistic_at_largest_finite_values(self, dtype, direction):
        prior = LogisticDistribution(["x"])
        value = jnp.asarray(direction * jnp.finfo(dtype).max, dtype=dtype)
        density, gradient = jax.jit(
            jax.value_and_grad(lambda x: prior.log_prob({"x": x}))
        )(value)
        assert jnp.isfinite(density)
        assert density == -jnp.abs(value)
        assert gradient == -direction

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("value", [-20.0, -2.0, -0.01, -0.0, 0.0, 0.01, 2.0, 20.0])
    def test_logistic_density_gradient_and_curvature(self, dtype, value):
        prior = LogisticDistribution(["x"])
        log_prob = lambda x: prior.log_prob({"x": x})
        x = jnp.asarray(value, dtype=dtype)
        density, gradient = jax.jit(jax.value_and_grad(log_prob))(x)
        curvature = jax.jit(jax.grad(jax.grad(log_prob)))(x)
        expected_gradient = -jnp.tanh(x / 2)
        expected_curvature = -0.5 * (1 - expected_gradient**2)
        tolerance = 2e-6 if dtype == jnp.float32 else 1e-12
        assert jnp.allclose(
            density, stats.logistic.logpdf(value), rtol=tolerance, atol=tolerance
        )
        assert jnp.allclose(gradient, expected_gradient, rtol=tolerance, atol=tolerance)
        assert jnp.allclose(
            curvature, expected_curvature, rtol=tolerance, atol=tolerance
        )

    @pytest.mark.parametrize("value", [-1e300, -1000.0, 1000.0, 1e300])
    def test_logistic_tail_density_and_gradient_remain_finite(self, value):
        prior = LogisticDistribution(["x"])
        log_prob = lambda x: prior.log_prob({"x": x})
        density, gradient = jax.jit(jax.value_and_grad(log_prob))(jnp.asarray(value))
        assert jnp.isfinite(density)
        assert jnp.isclose(density, -abs(value))
        assert jnp.isclose(gradient, -jnp.sign(value))

    def test_logistic_distribution(self):
        """Test the LogisticDistribution prior."""
        p = LogisticDistribution(["x"])

        # Draw samples and check they are finite
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["x"])

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob matches scipy.stats.logistic
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(p.add_name(x[None])), stats.logistic.logpdf(x)
        )

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(p.add_name(x[None]))
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(p.add_name(x[None])))

    def test_standard_normal_distribution(self):
        """Test the StandardNormalDistribution prior."""
        p = StandardNormalDistribution(["x"])

        # Draw samples and check they are finite
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["x"])

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob matches scipy.stats.norm
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(p.add_name(x[None])), stats.norm.logpdf(x)
        )

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(p.add_name(x[None]))
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(p.add_name(x[None])))

    def test_uniform_distribution(self):
        """Test the UniformDistribution prior."""
        p = UniformDistribution(["x"])
        xmin, xmax = p.xmin, p.xmax  # 0.0, 1.0

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["x"])
        assert_all_in_range(samples["x"], xmin, xmax)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support
        x = p.add_name(jnp.linspace(xmin, xmax, 1000)[None])
        assert jnp.allclose(
            jax.vmap(p.log_prob)(x), -jnp.log(xmax - xmin) * jnp.ones_like(x["x"])
        )

        # Check log_prob is -inf outside the support
        x_outside = p.add_name(jnp.array([xmin - 1.0, xmax + 1.0])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(x)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(x))

    def test_uniform(self):
        """Test the UniformPrior prior."""
        xmin, xmax = -10.0, 10.0
        p = UniformPrior(xmin, xmax, ["x"])

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["x"])
        assert_all_in_range(samples["x"], xmin, xmax)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support (use valid range [0, 1] for base)
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.0, 1.0, 1000)[None])
        y = jax.vmap(p.transform)(x)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), -jnp.log(xmax - xmin))

        # Check log_prob is -inf outside the support
        x_outside = p.add_name(jnp.array([xmin - 1.0, xmax + 1.0])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_sine(self):
        """Test the SinePrior prior."""
        p = SinePrior(["x"])

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["x"])
        assert_all_in_range(samples["x"], 0.0, jnp.pi)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support (use valid range (0, 1) exclusive
        # to avoid boundaries where sin(x) = 0 gives log_prob = -inf)
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.001, 0.999, 1000)[None])
        y = jax.vmap(p.base_prior[0].transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.sin(y["x"]) / 2.0))

        # Check log_prob is -inf outside the support
        x_outside = p.add_name(jnp.array([-0.5, jnp.pi + 0.5])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_cosine(self):
        """Test the CosinePrior prior."""
        p = CosinePrior(["x"])

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["x"])
        assert_all_in_range(samples["x"], -jnp.pi / 2.0, jnp.pi / 2.0)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support (use valid range (0, 1) exclusive
        # to avoid boundaries where cos(x) = 0 gives log_prob = -inf)
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.001, 0.999, 1000)[None])
        y = jax.vmap(p.base_prior[0].base_prior[0].transform)(x)
        y = jax.vmap(p.base_prior[0].transform)(y)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.cos(y["x"]) / 2.0))

        # Check log_prob is -inf outside the support
        x_outside = p.add_name(
            jnp.array([-jnp.pi / 2.0 - 0.5, jnp.pi / 2.0 + 0.5])[None]
        )
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_power_law(self):
        """Test the PowerLawPrior prior for various exponents."""
        xmin, xmax = 0.1, 100.0
        for alpha in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]:
            alpha = float(alpha)
            p = PowerLawPrior(xmin, xmax, alpha, ["x"])

            # Draw samples and check they are finite and in range
            samples = p.sample(jax.random.key(0), 10000)
            assert_all_finite(samples["x"])
            assert_all_in_range(samples["x"], xmin, xmax)

            # Check log_prob is finite for samples
            log_prob = jax.vmap(p.log_prob)(samples)
            assert_all_finite(log_prob)

            # Check log_prob is correct in the support (use valid range (0, 1] for base,
            # excluding 0 as it maps to the boundary where log_prob = -inf)
            x = p.trace_prior_parent([])[0].add_name(
                jnp.linspace(0.001, 1.0, 1000)[None]
            )
            y = jax.vmap(p.transform)(x)
            if alpha < -1.0:
                expected = (
                    alpha * jnp.log(y["x"])
                    + jnp.log(-alpha - 1)
                    - jnp.log(xmin ** (alpha + 1) - xmax ** (alpha + 1))
                )
            elif alpha > -1.0:
                expected = (
                    alpha * jnp.log(y["x"])
                    + jnp.log(alpha + 1)
                    - jnp.log(xmax ** (alpha + 1) - xmin ** (alpha + 1))
                )
            else:
                expected = -jnp.log(y["x"]) - jnp.log(jnp.log(xmax) - jnp.log(xmin))
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

    def test_gaussian(self):
        """Test the GaussianPrior prior."""
        mu, sigma = 2.0, 3.0
        p = GaussianPrior(mu, sigma, ["x"])

        # Draw samples and check they are finite
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["x"])

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.transform)(x)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(y), stats.norm.logpdf(y["x"], loc=mu, scale=sigma)
        )

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_Rayleigh(self):
        """Test the RayleighPrior prior."""
        sigma = 2.0
        p = RayleighPrior(sigma, ["x"])

        # Draw samples and check they are finite and positive
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["x"])
        assert jnp.all(samples["x"] > 0.0)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support (use valid range (0, 1) exclusive
        # to avoid boundaries: 0 maps to inf and 1 maps to the boundary where log_prob = -inf)
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.001, 0.999, 1000)[None])
        y = jax.vmap(p.base_prior[0].transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(y), stats.rayleigh.logpdf(y["x"], scale=sigma)
        )

        # Check log_prob is -inf for negative values (outside support [0, inf))
        x_outside = p.add_name(jnp.array([-1.0, -10.0])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))


class TestMultivariatePrior:
    def test_uniform_sphere(self):
        """Test the UniformSpherePrior prior."""
        p = UniformSpherePrior(["x"])

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.key(0), 10000)
        assert_all_finite(samples["x_mag"])
        assert_all_finite(samples["x_theta"])
        assert_all_finite(samples["x_phi"])
        assert_all_in_range(samples["x_mag"], 0.0, 1.0)
        assert_all_in_range(samples["x_theta"], 0.0, jnp.pi)
        assert_all_in_range(samples["x_phi"], 0.0, 2 * jnp.pi)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support (theta away from 0/pi where
        # sin(theta) -> 0 gives log_prob = -inf). The joint density factorizes over
        # the three independent component priors.
        mag = jnp.linspace(0.05, 0.95, 1000)
        theta = jnp.linspace(0.1, jnp.pi - 0.1, 1000)
        phi = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 1000)
        grid = p.add_name(jnp.stack([mag, theta, phi]))
        expected = (
            -jnp.log(1.0)  # x_mag ~ UniformPrior(0, max_mag=1.0): -log(xmax - xmin)
            + jnp.log(jnp.sin(theta) / 2.0)  # x_theta ~ SinePrior
            + (-jnp.log(2 * jnp.pi))  # x_phi ~ UniformPrior(0, 2*pi)
        )
        assert jnp.allclose(jax.vmap(p.log_prob)(grid), expected)

        # Check log_prob is -inf outside the support (one coordinate past its bound)
        x_outside = p.add_name(
            jnp.stack(
                [
                    jnp.array([-0.5, 1.5, 0.5, 0.5]),  # x_mag < 0, x_mag > 1
                    jnp.array([0.5, 0.5, -0.5, jnp.pi + 0.5]),  # x_theta < 0, > pi
                    jnp.array([1.0, 1.0, 1.0, 1.0]),
                ]
            )
        )
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)
        assert not jnp.any(jnp.isnan(logp_outside))

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(grid)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(grid))


class TestOther:
    def test_bounded_mixin(self):
        """Test the BoundedMixin mixin."""

        class TestBoundedPrior(BoundedMixin, SequentialTransformPrior):
            xmin: float = -2.0
            xmax: float = 3.0

            def __init__(self, parameter_names):
                super().__init__(
                    [StandardNormalDistribution([f"{parameter_names[0]}_base"])],
                    [],
                )

        p = TestBoundedPrior(["x"])

        # Check log_prob is finite inside bounds
        x_inside = p.add_name(jnp.array([-1.5, 0.0, 2.5])[None])
        logp_inside = jax.vmap(p.log_prob)(x_inside)
        assert_all_finite(logp_inside)

        # Check log_prob is -inf outside bounds
        x_outside = p.add_name(jnp.array([-2.5, 3.5])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check JIT works correctly
        x = p.add_name(jnp.linspace(-3.0, 4.0, 100)[None])
        logp = jax.vmap(p.log_prob)(x)
        jitted_logp = jax.jit(jax.vmap(p.log_prob))(x)
        assert jnp.allclose(logp, jitted_logp)

        assert p.is_normalized is False

    def test_custom_prior_is_normalized_defaults_to_false(self):
        """A custom Prior subclass without is_normalized override returns False."""

        class MyPrior(Prior):
            def log_prob(self, z):
                return jnp.array(0.0)

            def sample(self, rng_key, n_samples):
                return {"x": jnp.zeros(n_samples)}

        assert MyPrior(("x",)).is_normalized is False

    def test_unnormalized_prior_raises_for_evidence_samplers(self):
        """Jim rejects unnormalized priors for evidence-computing samplers."""

        class MyPrior(Prior):
            def log_prob(self, z):
                return jnp.array(0.0)

            def sample(self, rng_key, n_samples):
                return {"x": jnp.zeros(n_samples)}

        class MockLikelihood(LikelihoodBase):
            def evaluate(self, params):
                return jnp.array(0.0)

        prior = MyPrior(("x",))
        lh = MockLikelihood()

        for sampler_config in (
            BlackJAXNSSConfig(),
            BlackJAXSMCConfig(),
        ):
            with pytest.raises(ValueError, match="normalized prior"):
                Jim(
                    likelihood=lh,
                    prior=prior,
                    sampler_config=sampler_config,
                )
