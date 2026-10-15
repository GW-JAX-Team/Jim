"""Scalar JAX array annotations with explicit float, integer or complex dtypes.

``FloatLike`` and ``ComplexLike`` also accept Python floats and complex numbers. Jaxtyping's ``Scalar`` and
``ScalarLike`` accept other dtypes, so they cannot replace these aliases.
"""

from typing import TypeAlias

from jaxtyping import Array, Complex, Float, Int

FloatScalar: TypeAlias = Float[Array, ""]
IntScalar: TypeAlias = Int[Array, ""]
ComplexScalar: TypeAlias = Complex[Array, ""]
FloatLike: TypeAlias = float | Float[Array, ""]
ComplexLike: TypeAlias = complex | Complex[Array, ""]
