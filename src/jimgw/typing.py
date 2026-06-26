"""Convenience type aliases for jaxtyping scalar annotations."""

from typing import TypeAlias

from jaxtyping import Array, Complex, Float, Int

FloatScalar: TypeAlias = Float[Array, ""]
IntScalar: TypeAlias = Int[Array, ""]
ComplexScalar: TypeAlias = Complex[Array, ""]
FloatLike: TypeAlias = float | Float[Array, ""]
