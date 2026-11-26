import jax.numpy as jnp
from jax.scipy.special import i0e
from jaxtyping import Array, Float

EPS = 1e-15


def log_i0(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    A numerically stable method to evaluate log of
    a modified Bessel function of order 0.
    It is used in the phase-marginalized likelihood.

    Args:
        x (array-like): Value(s) at which to evaluate the function.

    Returns:
        array-like: The natural logarithm of the Bessel function.
    """
    return jnp.log(i0e(x)) + x


def safe_arctan2(
    y: Float[Array, "..."],
    x: Float[Array, "..."],
    default_value: float | Float[Array, "..."] = 0.0,
) -> Float[Array, "..."]:
    """
    A numerically stable method to evaluate arctan2 upon taking gradient.

    The gradient (jnp.jacfwd) of the default jnp.arctan2 is undefined
    at (0, 0) and returns NaN. This function circumvents this issue by
    specifying a default value at that point.

    Args:
        y (array-like): y-coordinate of the point.
        x (array-like): x-coordinate of the point.
        default_value (float): arctan2 value to return at (0, 0). Default is 0.0.

    Returns:
        array-like: The signed azimuthal angle, in radians, within [-pi, pi].
    """
    at_origin = (jnp.abs(x) < EPS) & (jnp.abs(y) < EPS)
    # Keep the unused branch away from (0, 0) so that its gradient is not NaN.
    angle = jnp.atan2(jnp.where(at_origin, 1.0, y), jnp.where(at_origin, 1.0, x))
    return jnp.where(at_origin, default_value * jnp.ones_like(x), angle)


def safe_polar_angle(
    x: Float[Array, "..."], y: Float[Array, "..."], z: Float[Array, "..."]
) -> Float[Array, "..."]:
    """
    A numerically stable method to compute the polar angle upon taking gradient.

    The polar angle is computed as:
        theta = ArcTan2[ Sqrt[x^2 + y^2], z ],
    which keeps full precision near the poles.
    The gradient (jnp.jacfwd) of this method, however,
    will return NaN when both x and y are 0.
    This function circumvents this issue by computing the simplified
    expression of the function at the troubled point.

    Args:
        x (array-like): x-coordinate of the point.
        y (array-like): y-coordinate of the point.
        z (array-like): z-coordinate of the point.

    Returns:
        array-like: The polar angle, in radians, within [0, pi].
    """
    on_axis = (jnp.abs(x) < EPS) & (jnp.abs(y) < EPS)
    # Keep the unused branch away from the axis so that its gradient is not NaN.
    x_safe = jnp.where(on_axis, 1.0, x)
    y_safe = jnp.where(on_axis, 1.0, y)
    return jnp.where(
        on_axis,
        jnp.arccos(jnp.sign(z)),
        jnp.arctan2(jnp.hypot(x_safe, y_safe), z),
    )


def carte_to_spherical_angles(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    z: Float[Array, "..."],
    default_value: float | Float[Array, "..."] = 0.0,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    A numerically stable method to compute the spherical angles upon taking gradient.

    For more details, see:
    * `safe_polar_angle` for the polar angle.
    * `safe_arctan2` for the azimuthal angle.

    Args:
        x (array-like): x-coordinate of the point.
        y (array-like): y-coordinate of the point.
        z (array-like): z-coordinate of the point.
        default_value (float): arctan2 value to return at (0, 0). Default is 0.0.

    Returns:
        tuple: A tuple containing:
            - theta (array-like): The polar angle, in radians, within [0, pi].
            - phi (array-like): The signed azimuthal angle, in radians, within [-pi, pi].
    """
    return safe_polar_angle(x, y, z), safe_arctan2(y, x, default_value)


def round_up_to_power_of_two(n: int) -> int:
    """Round up to the nearest power of two, returning 1 for nonpositive inputs."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()
