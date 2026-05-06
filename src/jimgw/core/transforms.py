from abc import ABC
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.scipy.special import logit
from beartype import beartype as typechecker
from jaxtyping import Float, Array, jaxtyped


class Transform(ABC):
    """Abstract base class for parameter-space transforms.

    Transforms are responsible for mapping named parameter dictionaries between
    two coordinate spaces. Subclasses implement the actual numerical mapping;
    this base class handles name bookkeeping.

    Attributes:
        name_mapping: A pair ``(from_names, to_names)`` describing which input
            parameters are consumed and which output parameters are produced.
    """

    name_mapping: tuple[list[str], list[str]]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ) -> None:
        """
        Args:
            name_mapping: Pair of ``(input_parameter_names, output_parameter_names)``.
        """
        self.name_mapping = name_mapping

    def propagate_name(self, x: Sequence[str]) -> tuple[str, ...]:
        """Propagate a parameter-name tuple through this transform.

        Removes the names consumed by this transform and appends the names
        it produces, preserving the order of unchanged names.

        Args:
            x: Sequence of parameter names in the current space.

        Returns:
            New tuple of parameter names after applying this transform.
        """
        # Remove names in from_list, then append names in to_list (preserving order)
        result = [name for name in x if name not in self.name_mapping[0]]
        result += [name for name in self.name_mapping[1] if name not in result]
        return tuple(result)


class NtoMTransform(Transform):
    """N-to-M transform: consumes N named parameters and produces M named parameters.

    Only a forward pass is defined. This is the most general transform type.

    Attributes:
        transform_func (Callable): Callable that maps an input dict to an output dict.
    """

    transform_func: Callable[[dict[str, Float]], dict[str, Float]]

    def __repr__(self):
        return f"NtoMTransform(name_mapping={self.name_mapping})"

    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        """Push forward the input to the transformed coordinate.

        Args:
            x: Parameter dictionary in the input space.

        Returns:
            Parameter dictionary in the transformed space.
        """
        x_copy = x.copy()
        output_params = self.transform_func(x_copy)
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy


class NtoNTransform(NtoMTransform):
    """N-to-N transform with automatic Jacobian computation via forward-mode AD.

    The number of input and output parameters must be equal. The log-Jacobian
    determinant of the transform is computed automatically using ``jax.jacfwd``.
    """

    def __repr__(self):
        return f"NtoNTransform(name_mapping={self.name_mapping})"

    @property
    def n_dim(self) -> int:
        """Number of parameters consumed/produced by this transform."""
        return len(self.name_mapping[0])

    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """Transform to the output coordinate and return the log-Jacobian determinant.

        Args:
            x: Parameter dictionary in the input space.

        Returns:
            Tuple of the transformed parameter dictionary and the log absolute
            Jacobian determinant.
        """
        x_copy = x.copy()
        transform_params = dict((key, x_copy[key]) for key in self.name_mapping[0])
        output_params = self.transform_func(transform_params)
        jacobian = jax.jacfwd(self.transform_func)(transform_params)
        jacobian = jnp.array(jax.tree.leaves(jacobian))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy, jacobian


class BijectiveTransform(NtoNTransform):
    """Bijective (invertible) N-to-N transform.

    Extends [`NtoNTransform`][jimgw.core.transforms.NtoNTransform] with an inverse
    transform function and corresponding ``inverse`` and ``backward`` methods.

    Attributes:
        inverse_transform_func: Callable implementing the inverse map.
    """

    inverse_transform_func: Callable[[dict[str, Float]], dict[str, Float]]

    def __repr__(self):
        return f"BijectiveTransform(name_mapping={self.name_mapping})"

    def inverse(self, y: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """Inverse transform from the output coordinate back to the input coordinate.

        Args:
            y: Parameter dictionary in the transformed (output) space.

        Returns:
            Tuple of the inverse-transformed parameter dictionary and the log
            absolute Jacobian determinant.
        """
        y_copy = y.copy()
        transform_params = dict((key, y_copy[key]) for key in self.name_mapping[1])
        output_params = self.inverse_transform_func(transform_params)
        jacobian = jax.jacfwd(self.inverse_transform_func)(transform_params)
        jacobian = jnp.array(jax.tree.leaves(jacobian))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy, jacobian

    def backward(self, y: dict[str, Float]) -> dict[str, Float]:
        """Pull back from the output coordinate to the input coordinate (no Jacobian).

        Args:
            y: Parameter dictionary in the transformed (output) space.

        Returns:
            Parameter dictionary in the original (input) space.
        """
        y_copy = y.copy()
        output_params = self.inverse_transform_func(y_copy)
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy


class ConditionalBijectiveTransform(BijectiveTransform):
    """Bijective transform that depends on additional conditioning parameters.

    Like [`BijectiveTransform`][jimgw.core.transforms.BijectiveTransform], but the
    transform and inverse functions also receive a set of fixed conditioning
    parameters (e.g. masses when converting spin angles). The Jacobian is computed
    only with respect to the primary (non-conditioning) parameters.

    Attributes:
        conditional_names: Names of the conditioning parameters.
    """

    conditional_names: list[str]

    def __repr__(self):
        return f"ConditionalBijectiveTransform(name_mapping={self.name_mapping}, conditional_names={self.conditional_names})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        conditional_names: list[str],
    ) -> None:
        """
        Args:
            name_mapping: Pair of ``(input_names, output_names)`` for the primary
                parameters.
            conditional_names: Names of the conditioning parameters that are passed
                through unchanged.
        """
        super().__init__(name_mapping)
        self.conditional_names = conditional_names

    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """Apply the conditional forward transform and compute the log-Jacobian.

        Args:
            x: Parameter dictionary containing both primary and conditioning parameters.

        Returns:
            The transformed parameter dictionary and the log absolute Jacobian
            determinant w.r.t. the primary parameters.
        """
        x_copy = x.copy()
        transform_params = dict((key, x_copy[key]) for key in self.name_mapping[0])
        transform_params.update(
            dict((key, x_copy[key]) for key in self.conditional_names)
        )
        output_params = self.transform_func(transform_params)
        jacobian = jax.jacfwd(self.transform_func)(transform_params)
        jacobian_copy = {
            key1: {key2: jacobian[key1][key2] for key2 in self.name_mapping[0]}
            for key1 in self.name_mapping[1]
        }
        jacobian = jnp.array(jax.tree.leaves(jacobian_copy))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy, jacobian

    def inverse(self, y: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """Apply the conditional inverse transform and compute the log-Jacobian.

        Args:
            y: Parameter dictionary in the output space, containing both primary
                (output) and conditioning parameters.

        Returns:
            The inverse-transformed parameter dictionary and the log absolute
            Jacobian determinant w.r.t. the primary output parameters.
        """
        y_copy = y.copy()
        transform_params = dict((key, y_copy[key]) for key in self.name_mapping[1])
        transform_params.update(
            dict((key, y_copy[key]) for key in self.conditional_names)
        )
        output_params = self.inverse_transform_func(transform_params)
        jacobian = jax.jacfwd(self.inverse_transform_func)(transform_params)
        jacobian_copy = {
            key1: {key2: jacobian[key1][key2] for key2 in self.name_mapping[1]}
            for key1 in self.name_mapping[0]
        }
        jacobian = jnp.array(jax.tree.leaves(jacobian_copy))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy, jacobian


@jaxtyped(typechecker=typechecker)
class ScaleTransform(BijectiveTransform):
    """Elementwise scaling transform: ``y = x * scale``.

    Attributes:
        scale: Multiplicative scale factor.
    """

    scale: Float

    def __repr__(self):
        return f"ScaleTransform(name_mapping={self.name_mapping}, scale={self.scale})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        scale: Float,
    ) -> None:
        """
        Args:
            name_mapping: ``(input_names, output_names)``.
            scale: Scale factor applied in the forward direction.
        """
        super().__init__(name_mapping)
        self.scale = scale
        self.transform_func = lambda x: {
            name_mapping[1][i]: x[name_mapping[0][i]] * self.scale
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: x[name_mapping[1][i]] / self.scale
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class OffsetTransform(BijectiveTransform):
    """Elementwise offset (translation) transform: ``y = x + offset``.

    Attributes:
        offset: Additive offset applied in the forward direction.
    """

    offset: Float

    def __repr__(self):
        return (
            f"OffsetTransform(name_mapping={self.name_mapping}, offset={self.offset})"
        )

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        offset: Float,
    ) -> None:
        """
        Args:
            name_mapping: ``(input_names, output_names)``.
            offset: Offset added in the forward direction.
        """
        super().__init__(name_mapping)
        self.offset = offset
        self.transform_func = lambda x: {
            name_mapping[1][i]: x[name_mapping[0][i]] + self.offset
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: x[name_mapping[1][i]] - self.offset
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class LogitTransform(BijectiveTransform):
    """Logit transform: ``y = sigmoid(x)`` (forward) / ``x = logit(y)`` (backward)."""

    def __repr__(self):
        return f"LogitTransform(name_mapping={self.name_mapping})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        """
        Args:
            name_mapping: Pair of ``(input_names, output_names)``.
        """
        super().__init__(name_mapping)
        self.transform_func = lambda x: {
            name_mapping[1][i]: 1 / (1 + jnp.exp(-x[name_mapping[0][i]]))
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: logit(x[name_mapping[1][i]])
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class SineTransform(BijectiveTransform):
    """Sine transformation: ``y = sin(x)``, with ``x ∈ [-π/2, π/2]``."""

    def __repr__(self):
        return f"SineTransform(name_mapping={self.name_mapping})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        """
        Args:
            name_mapping: Pair of ``(input_names, output_names)``.
        """
        super().__init__(name_mapping)
        self.transform_func = lambda x: {
            name_mapping[1][i]: jnp.sin(x[name_mapping[0][i]])
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.arcsin(x[name_mapping[1][i]])
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class CosineTransform(BijectiveTransform):
    """Cosine transformation: ``y = cos(x)``, with ``x ∈ [0, π]``."""

    def __repr__(self):
        return f"CosineTransform(name_mapping={self.name_mapping})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        """
        Args:
            name_mapping: Pair of ``(input_names, output_names)``.
        """
        super().__init__(name_mapping)
        self.transform_func = lambda x: {
            name_mapping[1][i]: jnp.cos(x[name_mapping[0][i]])
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.arccos(x[name_mapping[1][i]])
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class BoundToBound(BijectiveTransform):
    """Linear rescaling from one bounded interval to another.

    Maps ``x ∈ [original_lower, original_upper]`` to
    ``y ∈ [target_lower, target_upper]`` via a linear (affine) transform.

    Attributes:
        original_lower_bound (Float[Array, " n_dim"]): Lower bound(s) of the input interval.
        original_upper_bound (Float[Array, " n_dim"]): Upper bound(s) of the input interval.
        target_lower_bound (Float[Array, " n_dim"]): Lower bound(s) of the output interval.
        target_upper_bound (Float[Array, " n_dim"]): Upper bound(s) of the output interval.
    """

    original_lower_bound: Float[Array, " n_dim"]
    original_upper_bound: Float[Array, " n_dim"]
    target_lower_bound: Float[Array, " n_dim"]
    target_upper_bound: Float[Array, " n_dim"]

    def __repr__(self):
        return f"BoundToBound(name_mapping={self.name_mapping}, original_lower_bound={self.original_lower_bound}, original_upper_bound={self.original_upper_bound}, target_lower_bound={self.target_lower_bound}, target_upper_bound={self.target_upper_bound})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        original_lower_bound: Float | Float[Array, " n_dim"],
        original_upper_bound: Float | Float[Array, " n_dim"],
        target_lower_bound: Float | Float[Array, " n_dim"],
        target_upper_bound: Float | Float[Array, " n_dim"],
    ):
        """
        Args:
            name_mapping: Pair of ``(input_names, output_names)``.
            original_lower_bound: Lower bound(s) of the source interval.
            original_upper_bound: Upper bound(s) of the source interval.
            target_lower_bound: Lower bound(s) of the target interval.
            target_upper_bound: Upper bound(s) of the target interval.
        """
        super().__init__(name_mapping)
        self.original_lower_bound = jnp.atleast_1d(original_lower_bound)
        self.original_upper_bound = jnp.atleast_1d(original_upper_bound)
        self.target_lower_bound = jnp.atleast_1d(target_lower_bound)
        self.target_upper_bound = jnp.atleast_1d(target_upper_bound)

        self.transform_func = lambda x: {
            name_mapping[1][i]: (x[name_mapping[0][i]] - self.original_lower_bound[i])
            * (self.target_upper_bound[i] - self.target_lower_bound[i])
            / (self.original_upper_bound[i] - self.original_lower_bound[i])
            + self.target_lower_bound[i]
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: (x[name_mapping[1][i]] - self.target_lower_bound[i])
            * (self.original_upper_bound[i] - self.original_lower_bound[i])
            / (self.target_upper_bound[i] - self.target_lower_bound[i])
            + self.original_lower_bound[i]
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class BoundToUnbound(BijectiveTransform):
    """Logit-based transform from a bounded interval to the real line.

    Maps ``x ∈ (original_lower, original_upper)`` to ``y ∈ (-∞, +∞)`` via

    .. math::

        y = \\text{logit}\\!\\left(\\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}\\right).

    The inverse maps back with the sigmoid function.

    Attributes:
        original_lower_bound (Float[Array, " n_dim"]): Lower bound(s) of the input interval.
        original_upper_bound (Float[Array, " n_dim"]): Upper bound(s) of the input interval.
    """

    original_lower_bound: Float[Array, " n_dim"]
    original_upper_bound: Float[Array, " n_dim"]

    def __repr__(self):
        return f"BoundToUnbound(name_mapping={self.name_mapping}, original_lower_bound={self.original_lower_bound}, original_upper_bound={self.original_upper_bound})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        original_lower_bound: Float | Float[Array, " n_dim"],
        original_upper_bound: Float | Float[Array, " n_dim"],
    ):
        """
        Args:
            name_mapping: Pair of ``(input_names, output_names)``.
            original_lower_bound: Lower bound(s) of the source interval.
            original_upper_bound: Upper bound(s) of the source interval.
        """
        super().__init__(name_mapping)
        self.original_lower_bound = jnp.atleast_1d(original_lower_bound)
        self.original_upper_bound = jnp.atleast_1d(original_upper_bound)

        self.transform_func = lambda x: {
            name_mapping[1][i]: logit(
                (x[name_mapping[0][i]] - self.original_lower_bound[i])
                / (self.original_upper_bound[i] - self.original_lower_bound[i])
            )
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: (
                self.original_upper_bound[i] - self.original_lower_bound[i]
            )
            / (1 + jnp.exp(-x[name_mapping[1][i]]))
            + self.original_lower_bound[i]
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class SingleSidedUnboundTransform(BijectiveTransform):
    """Log transform from a lower-bounded domain to the real line: ``y = log(x - lower)``.

    Attributes:
        original_lower_bound: Lower bound(s) of the input domain.
    """

    original_lower_bound: Float[Array, " n_dim"]

    def __repr__(self):
        return f"SingleSidedUnboundTransform(name_mapping={self.name_mapping}, original_lower_bound={self.original_lower_bound})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        original_lower_bound: Float | Float[Array, " n_dim"],
    ):
        """
        Args:
            name_mapping: Pair of ``(input_names, output_names)``.
            original_lower_bound: Lower bound(s) of the input domain.
        """
        super().__init__(name_mapping)
        self.original_lower_bound = jnp.atleast_1d(original_lower_bound)

        self.transform_func = lambda x: {
            name_mapping[1][i]: jnp.log(
                x[name_mapping[0][i]] - self.original_lower_bound[i]
            )
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.exp(x[name_mapping[1][i]])
            + self.original_lower_bound[i]
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class PowerLawTransform(BijectiveTransform):
    """Power-law CDF transform from Uniform(0, 1) to a power-law distribution.

    Maps the unit interval to ``[xmin, xmax]`` with the CDF of ``p(x) ∝ x^alpha``.
    The ``alpha = -1`` (log-uniform) case is handled separately.

    Attributes:
        xmin: Lower bound of the output domain.
        xmax: Upper bound of the output domain.
        alpha: Power-law exponent.
    """

    xmin: Float
    xmax: Float
    alpha: Float

    def __repr__(self):
        return f"PowerLawTransform(name_mapping={self.name_mapping}, xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        xmin: Float,
        xmax: Float,
        alpha: Float,
    ):
        """
        Args:
            name_mapping: Pair of ``(input_names, output_names)``.
            xmin: Lower bound of the output domain.
            xmax: Upper bound of the output domain.
            alpha: Power-law exponent (``alpha = -1`` gives log-uniform).
        """
        super().__init__(name_mapping)
        self.xmin = xmin
        self.xmax = xmax
        self.alpha = alpha
        if alpha == -1.0:
            self.transform_func = lambda x: {
                name_mapping[1][i]: self.xmin
                * jnp.exp(x[name_mapping[0][i]] * jnp.log(self.xmax / self.xmin))
                for i in range(len(name_mapping[0]))
            }
            self.inverse_transform_func = lambda x: {
                name_mapping[0][i]: (
                    jnp.log(x[name_mapping[1][i]] / self.xmin)
                    / jnp.log(self.xmax / self.xmin)
                )
                for i in range(len(name_mapping[1]))
            }
        else:
            alphap1 = 1.0 + self.alpha
            self.transform_func = lambda x: {
                name_mapping[1][i]: (
                    self.xmin**alphap1
                    + x[name_mapping[0][i]] * (self.xmax**alphap1 - self.xmin**alphap1)
                )
                ** (1.0 / alphap1)
                for i in range(len(name_mapping[0]))
            }
            self.inverse_transform_func = lambda x: {
                name_mapping[0][i]: (
                    (x[name_mapping[1][i]] ** alphap1 - self.xmin**alphap1)
                    / (self.xmax**alphap1 - self.xmin**alphap1)
                )
                for i in range(len(name_mapping[1]))
            }


@jaxtyped(typechecker=typechecker)
class CartesianToPolarTransform(BijectiveTransform):
    """Cartesian-to-polar transform: ``(x, y) → (theta, r)`` with ``theta ∈ [0, 2π]``."""

    def __repr__(self):
        return f"CartesianToPolarTransform(name_mapping={self.name_mapping})"

    def __init__(
        self,
        parameter_name: str,
    ):
        """
        Args:
            parameter_name: Base name; consumes ``<name>_x`` and ``<name>_y``,
                produces ``<name>_theta`` and ``<name>_r``.
        """
        super().__init__(
            name_mapping=(
                [f"{parameter_name}_x", f"{parameter_name}_y"],
                [f"{parameter_name}_theta", f"{parameter_name}_r"],
            )
        )
        self.transform_func = lambda x: {
            f"{parameter_name}_theta": jnp.arctan2(
                x[f"{parameter_name}_y"], x[f"{parameter_name}_x"]
            )
            + jnp.pi,
            f"{parameter_name}_r": jnp.sqrt(
                x[f"{parameter_name}_x"] ** 2 + x[f"{parameter_name}_y"] ** 2
            ),
        }
        self.inverse_transform_func = lambda x: {
            f"{parameter_name}_x": x[f"{parameter_name}_r"]
            * jnp.cos(x[f"{parameter_name}_theta"]),
            f"{parameter_name}_y": x[f"{parameter_name}_r"]
            * jnp.sin(x[f"{parameter_name}_theta"]),
        }


@jaxtyped(typechecker=typechecker)
class PeriodicTransform(BijectiveTransform):
    """Transform a periodic parameter onto a 2D circle.

    Maps ``(r, θ)`` — where ``θ ∈ [xmin, xmax]`` is the periodic angle and
    ``r`` is a scale — to Cartesian coordinates ``(x, y) = r(cos θ', sin θ')``
    (with ``θ' = 2π(θ - xmin)/(xmax - xmin)``).  The inverse recovers
    ``(r, θ)`` from ``(x, y)``.

    Attributes:
        xmin (Float): Lower bound of the periodic parameter.
        xmax (Float): Upper bound of the periodic parameter.
    """

    def __repr__(self):
        return f"PeriodicTransform(name_mapping={self.name_mapping}, xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        xmin: Float,
        xmax: Float,
    ):
        """
        Args:
            name_mapping: Pair of ``([scale_name, angle_name], [x_name, y_name])``.
            xmin: Lower bound of the periodic parameter.
            xmax: Upper bound of the periodic parameter.
        """
        super().__init__(name_mapping)
        self.xmin = xmin
        self.xmax = xmax
        scaling = 2 * jnp.pi / (self.xmax - self.xmin)
        self.transform_func = lambda x: {
            f"{name_mapping[1][0]}": x[name_mapping[0][0]]
            * jnp.cos(scaling * (x[name_mapping[0][1]] - self.xmin)),
            f"{name_mapping[1][1]}": x[name_mapping[0][0]]
            * jnp.sin(scaling * (x[name_mapping[0][1]] - self.xmin)),
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][1]: self.xmin
            + (jnp.pi + jnp.arctan2(x[name_mapping[1][1]], x[name_mapping[1][0]]))
            / scaling,
            name_mapping[0][0]: jnp.sqrt(
                x[name_mapping[1][0]] ** 2 + x[name_mapping[1][1]] ** 2
            ),
        }


@jaxtyped(typechecker=typechecker)
class RayleighTransform(BijectiveTransform):
    """CDF transform from Uniform(0, 1) to a Rayleigh distribution.

    Maps ``u ∈ [0, 1]`` to ``x = sigma * sqrt(-2 * log(u))``.

    Attributes:
        sigma: Scale parameter of the Rayleigh distribution.
    """

    def __repr__(self):
        return (
            f"RayleighTransform(name_mapping={self.name_mapping}, sigma={self.sigma})"
        )

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        sigma: Float,
    ):
        """
        Args:
            name_mapping: Pair of ``(input_names, output_names)``.
            sigma: Scale parameter of the Rayleigh distribution.
        """
        super().__init__(name_mapping)
        self.sigma = sigma
        self.transform_func = lambda x: {
            name_mapping[1][i]: sigma * jnp.sqrt(-2 * jnp.log(x[name_mapping[0][i]]))
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.exp(-((x[name_mapping[1][i]] / sigma) ** 2) / 2)
            for i in range(len(name_mapping[1]))
        }


def reverse_bijective_transform(
    original_transform: BijectiveTransform,
) -> BijectiveTransform:
    """Construct the inverse of a bijective transform.

    Swaps the forward and inverse functions and the name mapping of
    ``original_transform``, returning a new [`BijectiveTransform`][jimgw.core.transforms.BijectiveTransform] (or
    [`ConditionalBijectiveTransform`][jimgw.core.transforms.ConditionalBijectiveTransform]) whose forward direction is the
    original's inverse.

    Args:
        original_transform: The transform to invert.

    Returns:
        A new transform that is the reverse of the input.
    """
    reversed_name_mapping = (
        original_transform.name_mapping[1],
        original_transform.name_mapping[0],
    )
    if isinstance(original_transform, ConditionalBijectiveTransform):
        reversed_transform = ConditionalBijectiveTransform(
            name_mapping=reversed_name_mapping,
            conditional_names=original_transform.conditional_names,
        )
    else:
        reversed_transform = BijectiveTransform(name_mapping=reversed_name_mapping)
    reversed_transform.transform_func = original_transform.inverse_transform_func
    reversed_transform.inverse_transform_func = original_transform.transform_func
    reversed_transform.__repr__ = lambda: f"Reversed{repr(original_transform)}"

    return reversed_transform
