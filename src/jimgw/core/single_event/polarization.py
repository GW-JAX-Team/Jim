# Credit some part of the source code from bilby

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jimgw.typing import FloatLike

KNOWN_POLS = "pcxybl"


class Polarization(eqx.Module):
    """Object defining a given polarization mode, with utilities to produce
    corresponding tensor in an Earth centric frame.

    Args:
        name (str): One of 'p' (plus), 'c' (cross), 'x' (vector x), 'y' (vector y),
            'b' (breathing), or 'l' (longitudinal).
    """

    name: str

    def __init__(self, name: str):
        self.name = name.lower()
        if self.name not in KNOWN_POLS:
            e = f"unknown mode '{self.name}'; must be one of: {KNOWN_POLS}"
            raise ValueError(e)

    def _tensor_from_basis(
        self, x: Float[Array, "3 ..."], y: Float[Array, "3 ..."]
    ) -> Float[Array, "3 3 ..."]:
        """Polarization tensor from the orthonormal wave-frame basis vectors (x, y).

        Args:
            x: First basis vector, Cartesian components on the first axis.
            y: Second basis vector, same shape as ``x``.

        Returns:
            The 3x3 polarization tensor, with any trailing axes of ``x`` and ``y``.
        """
        outer_fmt = "i...,j...->ij..."
        if self.name == "p":
            return jnp.einsum(outer_fmt, x, x) - jnp.einsum(outer_fmt, y, y)
        elif self.name == "c":
            return jnp.einsum(outer_fmt, x, y) + jnp.einsum(outer_fmt, y, x)
        elif self.name == "x":
            z = jnp.cross(x, y, axis=0)
            return jnp.einsum(outer_fmt, x, z) + jnp.einsum(outer_fmt, z, x)
        elif self.name == "y":
            z = jnp.cross(x, y, axis=0)
            return jnp.einsum(outer_fmt, y, z) + jnp.einsum(outer_fmt, z, y)
        elif self.name == "b":
            return jnp.einsum(outer_fmt, x, x) + jnp.einsum(outer_fmt, y, y)
        elif self.name == "l":
            z = jnp.cross(x, y, axis=0)
            return jnp.einsum(outer_fmt, z, z)
        else:
            raise ValueError(f"unrecognized polarization {self.name}")

    def tensor_from_sky(
        self,
        ra: FloatLike,
        dec: FloatLike,
        psi: FloatLike,
        gmst: FloatLike,
    ) -> Float[Array, "3 3 ..."]:
        """Polarization tensor in Earth-fixed coordinates from the sky position.

        Args:
            ra (Float): Right ascension in radians.
            dec (Float): Declination in radians.
            psi (Float): Polarization angle in radians.
            gmst (Float): Greenwich mean sidereal time (GMST) in radians, or an
                array of them.

        Returns:
            Float[Array, "3 3 ..."]: The 3x3 polarization tensor; for an array of
            ``gmst``, one tensor per value along the trailing axes.
        """
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        # Broadcast so that an array of gmst gives one tensor per element.
        phi, theta = jnp.broadcast_arrays(ra - gmst, jnp.pi / 2 - dec)

        u = jnp.array(
            [
                jnp.cos(phi) * jnp.cos(theta),
                jnp.cos(theta) * jnp.sin(phi),
                -jnp.sin(theta),
            ]
        )
        v = jnp.array([-jnp.sin(phi), jnp.cos(phi), phi * 0.0])
        m = -u * jnp.sin(psi) - v * jnp.cos(psi)
        n = -u * jnp.cos(psi) + v * jnp.sin(psi)

        return self._tensor_from_basis(m, n)
