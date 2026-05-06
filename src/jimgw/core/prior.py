from dataclasses import field
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logit
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped
from abc import abstractmethod
import equinox as eqx

from jimgw.core.transforms import (
    BijectiveTransform,
    ScaleTransform,
    OffsetTransform,
    CosineTransform,
    PowerLawTransform,
    RayleighTransform,
    reverse_bijective_transform,
)


class Prior(eqx.Module):
    """Base class for prior distributions.

    This class should not be used directly. It provides a common interface and
    bookkeeping for parameter names and transforms.
    """

    parameter_names: tuple[str, ...]

    @property
    def n_dims(self) -> int:
        """Number of parameters in this prior."""
        return len(self.parameter_names)

    @property
    def is_normalized(self) -> bool:
        """Return True if this prior is a proper probability distribution (integrates to 1).

        Defaults to False for safety. All built-in Jim priors override this to True.
        Custom priors must explicitly set ``is_normalized = True`` (by overriding this
        property) only after verifying that ``∫ exp(log_prob(x)) dx == 1``.

        Samplers that compute Bayesian evidence (NSS, SMC) require a normalized prior.
        Jim will raise at construction time if this returns False for those backends.
        """
        return False

    def __init__(self, parameter_names: list[str]):
        """
        Args:
            parameter_names: List of parameter names for this prior.
        """
        self.parameter_names = tuple(parameter_names)

    def add_name(self, x: Float[Array, "n_dims"]) -> dict[str, Float]:
        """Convert a flat parameter array to a named dict.

        Args:
            x: Array of parameter values, shape ``(n_dims,)``.

        Returns:
            Dict mapping parameter names to scalar values.
        """
        return dict(zip(self.parameter_names, x, strict=True))

    def __call__(self, x: dict[str, Float]) -> Float:
        """Alias for `log_prob`."""
        return self.log_prob(x)

    @abstractmethod
    def log_prob(self, z: dict[str, Float]) -> Float:
        """Evaluate the log-probability of a sample.

        Args:
            z: Dict of parameter values.

        Returns:
            Log-probability scalar.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, rng_key: Key, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """Draw samples from the prior.

        Args:
            rng_key: JAX PRNG key.
            n_samples: Number of samples to draw.

        Returns:
            Dict mapping parameter names to arrays of shape ``(n_samples,)``.
        """
        raise NotImplementedError


@jaxtyped(typechecker=typechecker)
class CompositePrior(Prior):
    """Composite prior consisting of multiple component priors.

    Base class for [`SequentialTransformPrior`][jimgw.core.prior.SequentialTransformPrior] and [`CombinePrior`][jimgw.core.prior.CombinePrior].
    Used to build complex prior distributions from simpler ones.

    Attributes:
        base_prior: Component prior objects.
        parameter_names: Names of all parameters in this composite prior.
    """

    base_prior: tuple[Prior, ...]

    def __repr__(self):
        return f"Composite(priors={self.base_prior}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        priors: list[Prior],
    ):
        """
        Args:
            priors: List of component prior objects.
        """
        self.base_prior = tuple(priors)
        self.parameter_names = tuple(
            [name for prior in priors for name in prior.parameter_names]
        )

    def trace_prior_parent(self, output: Optional[list[Prior]] = None) -> list[Prior]:
        """Recursively collect all leaf (non-composite) priors.

        Args:
            output: Accumulator list. If ``None``, a new list is created.

        Returns:
            List of all leaf prior objects in this composite.
        """
        if output is None:
            output = []
        for subprior in self.base_prior:
            if isinstance(subprior, CompositePrior):
                output = subprior.trace_prior_parent(output)
            else:
                output.append(subprior)
        return output


@jaxtyped(typechecker=typechecker)
class LogisticDistribution(Prior):
    """One-dimensional logistic distribution prior."""

    @property
    def is_normalized(self) -> bool:
        return True

    def __repr__(self):
        return f"LogisticDistribution(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        super().__init__(parameter_names)
        assert self.n_dims == 1, "LogisticDistribution needs to be 1D distributions"

    def sample(
        self, rng_key: Key, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """Sample from a logistic distribution.

        Args:
            rng_key: JAX PRNG key.
            n_samples: Number of samples to draw.

        Returns:
            Dict mapping parameter name to samples of shape ``(n_samples,)``.
        """
        samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        samples = logit(samples)
        return self.add_name(samples[None])

    def log_prob(self, z: dict[str, Float]) -> Float:
        variable = z[self.parameter_names[0]]
        return -variable - 2 * jnp.log(1 + jnp.exp(-variable))


@jaxtyped(typechecker=typechecker)
class StandardNormalDistribution(Prior):
    """One-dimensional standard normal (Gaussian) distribution prior."""

    @property
    def is_normalized(self) -> bool:
        return True

    def __repr__(self):
        return f"StandardNormalDistribution(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        super().__init__(parameter_names)
        assert self.n_dims == 1, (
            "StandardNormalDistribution needs to be 1D distributions"
        )

    def sample(
        self, rng_key: Key, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """Sample from a standard normal distribution.

        Args:
            rng_key: JAX PRNG key.
            n_samples: Number of samples to draw.

        Returns:
            Dict mapping parameter name to samples of shape ``(n_samples,)``.
        """
        samples = jax.random.normal(rng_key, (n_samples,))
        return self.add_name(samples[None])

    def log_prob(self, z: dict[str, Float]) -> Float:
        variable = z[self.parameter_names[0]]
        return -0.5 * (variable**2 + jnp.log(2 * jnp.pi))


@jaxtyped(typechecker=typechecker)
class UniformDistribution(Prior):
    """One-dimensional uniform distribution prior over [0, 1]."""

    xmin: float = 0.0
    xmax: float = 1.0

    @property
    def is_normalized(self) -> bool:
        return True

    def __repr__(self):
        return f"UniformDistribution(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        super().__init__(parameter_names)
        assert self.n_dims == 1, "UniformDistribution needs to be 1D distributions"

    def sample(
        self, rng_key: Key, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """Sample from a uniform distribution.

        Args:
            rng_key: JAX PRNG key.
            n_samples: Number of samples to draw.

        Returns:
            Dict mapping parameter name to samples of shape ``(n_samples,)``.
        """
        samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        return self.add_name(samples[None])

    def log_prob(self, z: dict[str, Float]) -> Float:
        variable = z[self.parameter_names[0]]
        return jnp.where(
            jnp.logical_and(variable >= 0.0, variable <= 1.0), 0.0, -jnp.inf
        )


class SequentialTransformPrior(CompositePrior):
    """Prior distribution transformed by a sequence of bijective transforms.

    Attributes:
        base_prior (tuple[Prior, ...]): The base prior to transform (must be length 1).
        transforms (tuple[BijectiveTransform, ...]): Transforms applied sequentially
            in the forward direction.
        parameter_names (tuple[str, ...]): Names of the parameters after all transforms.
    """

    transforms: tuple[BijectiveTransform, ...]

    @property
    def is_normalized(self) -> bool:
        return self.base_prior[0].is_normalized

    def __repr__(self):
        return f"Sequential(priors={self.base_prior}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        base_prior: list[Prior],
        transforms: list[BijectiveTransform],
    ):
        """
        Args:
            base_prior: A single-element list containing the base prior.
            transforms: Ordered list of bijective transforms to apply to
                samples from the base prior.
        """
        assert len(base_prior) == 1, (
            "SequentialTransformPrior only takes one base prior"
        )
        super().__init__(base_prior)
        self.transforms = tuple(transforms)
        for transform in self.transforms:
            self.parameter_names = tuple(transform.propagate_name(self.parameter_names))

    def sample(
        self, rng_key: Key, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """Sample by drawing from the base prior and applying all transforms.

        Args:
            rng_key: JAX PRNG key.
            n_samples: Number of samples to draw.

        Returns:
            Transformed samples keyed by parameter name.
        """
        output = self.base_prior[0].sample(rng_key, n_samples)
        return jax.vmap(self.transform)(output)

    def log_prob(self, z: dict[str, Float]) -> Float:
        """Evaluate the log-probability of a transformed sample z.

        Applies the inverse transforms in reverse order, accumulating
        log-Jacobian determinants, then evaluates the base prior.

        Args:
            z: Sample in the transformed (output) space.

        Returns:
            Log-probability of z under the induced distribution.
        """
        output = 0
        for transform in reversed(self.transforms):
            z, log_jacobian = transform.inverse(z)
            output += log_jacobian
        output += self.base_prior[0].log_prob(z)
        return output

    def transform(self, x: dict[str, Float]) -> dict[str, Float]:
        """Apply all transforms sequentially (forward direction).

        Args:
            x: Sample in the base prior space.

        Returns:
            Transformed sample.
        """
        for transform in self.transforms:
            x = transform.forward(x)
        return x


class BoundedMixin:
    """
    Mixin class that adds bounds checking to log_prob.

    This mixin should be placed BEFORE the main prior class in the inheritance list
    (e.g., `class MyPrior(BoundedMixin, SequentialTransformPrior)`) to ensure the
    bounds check is applied before delegating to the base prior's log_prob.

    Classes using this mixin can override `xmin` and `xmax` attributes to set bounds.
    By default, the bounds are (-inf, inf), meaning no bounds checking.

    The mixin returns -inf for values outside [xmin, xmax].

    Note: This is a mixin and should not be used standalone. It relies on the
    presence of `parameter_names` attribute from the Prior class and the
    implementation of `log_prob` in the base class.
    """

    xmin: float = -jnp.inf
    xmax: float = jnp.inf

    @property
    def is_normalized(self) -> bool:
        return False

    def log_prob(self, z: dict[str, Float]) -> Float:
        x = z[self.parameter_names[0]]  # type: ignore[attr-defined]
        base_log_prob = super().log_prob(z)  # type: ignore[misc]
        return jnp.where(
            jnp.logical_and(x >= self.xmin, x <= self.xmax),
            base_log_prob,
            -jnp.inf,
        )


class CombinePrior(CompositePrior):
    """Multivariate prior constructed by joining multiple independent priors.

    The joint log-probability is the sum of the individual log-probabilities,
    which is valid when all component priors are independent.

    Attributes:
        base_prior: Independent component priors.
        parameter_names: Names of all parameters in the combined prior.
    """

    base_prior: tuple[Prior, ...] = field(default_factory=tuple)

    @property
    def is_normalized(self) -> bool:
        return all(p.is_normalized for p in self.base_prior)

    def __repr__(self):
        return (
            f"Combine(priors={self.base_prior}, parameter_names={self.parameter_names})"
        )

    def __init__(
        self,
        priors: list[Prior],
    ):
        """
        Args:
            priors: List of independent prior objects to combine.
        """
        super().__init__(priors)

    def sample(
        self, rng_key: Key, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """Sample from all component priors independently.

        Args:
            rng_key: JAX PRNG key (split internally for each component).
            n_samples: Number of samples to draw.

        Returns:
            Combined samples from all component priors, keyed by parameter name.
        """
        output = {}
        for prior in self.base_prior:
            rng_key, subkey = jax.random.split(rng_key)
            output.update(prior.sample(subkey, n_samples))
        return output

    def log_prob(self, z: dict[str, Float]) -> Float:
        """Evaluate the joint log-probability as the sum of component log-probabilities.

        Args:
            z: Dictionary of parameter values.

        Returns:
            Sum of log-probabilities from all component priors.
        """
        output = 0.0
        for prior in self.base_prior:
            output += prior.log_prob(z)
        return output


@jaxtyped(typechecker=typechecker)
class UniformPrior(SequentialTransformPrior):
    """Uniform prior over a finite interval ``[xmin, xmax]``.

    Attributes:
        xmin: Lower bound of the interval.
        xmax: Upper bound of the interval.
    """

    xmin: float
    xmax: float

    def __repr__(self):
        return f"UniformPrior(xmin={self.xmin}, xmax={self.xmax}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        xmin: float,
        xmax: float,
        parameter_names: list[str],
    ):
        assert len(parameter_names) == 1, "UniformPrior needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        super().__init__(
            [UniformDistribution([f"{parameter_names[0]}_base"])],
            [
                ScaleTransform(
                    (
                        [f"{parameter_names[0]}_base"],
                        [f"{parameter_names[0]}-({xmin})"],
                    ),
                    xmax - xmin,
                ),
                OffsetTransform(
                    ([f"{parameter_names[0]}-({xmin})"], parameter_names),
                    xmin,
                ),
            ],
        )


@jaxtyped(typechecker=typechecker)
class GaussianPrior(SequentialTransformPrior):
    """Gaussian (normal) prior with specified mean and standard deviation.

    Attributes:
        mu: Mean of the distribution.
        sigma: Standard deviation of the distribution.
    """

    mu: float
    sigma: float

    def __repr__(self):
        return f"GaussianPrior(mu={self.mu}, sigma={self.sigma}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        mu: float,
        sigma: float,
        parameter_names: list[str],
    ):
        """
        Args:
            mu: Mean of the distribution.
            sigma: Standard deviation of the distribution.
            parameter_names: List with a single parameter name.
        """
        assert len(parameter_names) == 1, "GaussianPrior needs to be 1D distributions"
        self.mu = mu
        self.sigma = sigma
        super().__init__(
            [StandardNormalDistribution([f"{parameter_names[0]}_base"])],
            [
                ScaleTransform(
                    (
                        [f"{parameter_names[0]}_base"],
                        [f"{parameter_names[0]}-({mu})"],
                    ),
                    sigma,
                ),
                OffsetTransform(
                    ([f"{parameter_names[0]}-({mu})"], parameter_names),
                    mu,
                ),
            ],
        )


@jaxtyped(typechecker=typechecker)
class SinePrior(BoundedMixin, SequentialTransformPrior):
    """Prior with PDF proportional to ``sin(x)`` over ``[0, π]``."""

    xmin: float = 0.0
    xmax: float = jnp.pi

    @property
    def is_normalized(self) -> bool:
        return True

    def __repr__(self):
        return f"SinePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str]):
        assert len(parameter_names) == 1, "SinePrior needs to be 1D distributions"
        super().__init__(
            [UniformPrior(-1.0, 1.0, [f"cos({parameter_names[0]})"])],
            [
                reverse_bijective_transform(
                    CosineTransform(
                        (
                            [f"{parameter_names[0]}"],
                            [f"cos({parameter_names[0]})"],
                        )
                    )
                )
            ],
        )


@jaxtyped(typechecker=typechecker)
class CosinePrior(BoundedMixin, SequentialTransformPrior):
    """Prior with PDF proportional to ``cos(x)`` over ``[-π/2, π/2]``."""

    xmin: float = -jnp.pi / 2
    xmax: float = jnp.pi / 2

    @property
    def is_normalized(self) -> bool:
        return True

    def __repr__(self):
        return f"CosinePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str]):
        assert len(parameter_names) == 1, "CosinePrior needs to be 1D distributions"
        super().__init__(
            [SinePrior([f"{parameter_names[0]}+pi/2"])],
            [
                OffsetTransform(
                    (
                        (
                            [f"{parameter_names[0]}+pi/2"],
                            [f"{parameter_names[0]}"],
                        )
                    ),
                    -jnp.pi / 2,
                )
            ],
        )


@jaxtyped(typechecker=typechecker)
class UniformSpherePrior(CombinePrior):
    """Uniform prior over a sphere, parameterized by magnitude, polar angle, and azimuth."""

    def __repr__(self):
        return f"UniformSpherePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], max_mag: float = 1.0):
        """
        Args:
            parameter_names: Single-element list with the base parameter name.
                Expands to ``<name>_mag``, ``<name>_theta``, ``<name>_phi``.
            max_mag: Maximum magnitude of the vector.
        """
        assert len(parameter_names) == 1, (
            "UniformSpherePrior only takes the name of the vector"
        )
        parameter_names = [
            f"{parameter_names[0]}_{suffix}" for suffix in ("mag", "theta", "phi")
        ]
        super().__init__(
            [
                UniformPrior(0.0, max_mag, [parameter_names[0]]),
                SinePrior([parameter_names[1]]),
                UniformPrior(0.0, 2 * jnp.pi, [parameter_names[2]]),
            ]
        )


@jaxtyped(typechecker=typechecker)
class RayleighPrior(BoundedMixin, SequentialTransformPrior):
    """Rayleigh distribution prior with scale parameter sigma.

    Attributes:
        sigma: Scale parameter of the Rayleigh distribution.
    """

    sigma: float
    xmin: float = 0.0
    xmax: float = jnp.inf

    @property
    def is_normalized(self) -> bool:
        return True

    def __repr__(self):
        return (
            f"RayleighPrior(sigma={self.sigma}, parameter_names={self.parameter_names})"
        )

    def __init__(
        self,
        sigma: float,
        parameter_names: list[str],
    ):
        """
        Args:
            sigma: Scale parameter of the Rayleigh distribution.
            parameter_names: List with a single parameter name.
        """
        assert len(parameter_names) == 1, "RayleighPrior needs to be 1D distributions"
        self.sigma = sigma
        super().__init__(
            [UniformPrior(0.0, 1.0, [f"{parameter_names[0]}_base"])],
            [
                RayleighTransform(
                    ([f"{parameter_names[0]}_base"], parameter_names),
                    sigma=sigma,
                )
            ],
        )


@jaxtyped(typechecker=typechecker)
class PowerLawPrior(SequentialTransformPrior):
    """Power-law prior over ``[xmin, xmax]`` with exponent alpha.

    Attributes:
        xmin: Lower bound of the interval (must be positive).
        xmax: Upper bound of the interval.
        alpha: Power-law exponent.
    """

    xmin: float
    xmax: float
    alpha: float

    def __repr__(self):
        return f"PowerLawPrior(xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha}, naming={self.parameter_names})"

    def __init__(
        self,
        xmin: float,
        xmax: float,
        alpha: float,
        parameter_names: list[str],
    ):
        """
        Args:
            xmin: Lower bound (must be positive).
            xmax: Upper bound.
            alpha: Power-law exponent.
            parameter_names: List with a single parameter name.
        """
        assert len(parameter_names) == 1, "Power law needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        self.alpha = alpha
        assert self.xmin < self.xmax, "xmin must be less than xmax"
        assert self.xmin > 0.0, "x must be positive"
        super().__init__(
            [UniformDistribution([f"{parameter_names[0]}_base"])],
            [
                PowerLawTransform(
                    (
                        [f"{parameter_names[0]}_base"],
                        parameter_names,
                    ),
                    xmin,
                    xmax,
                    alpha,
                ),
            ],
        )
