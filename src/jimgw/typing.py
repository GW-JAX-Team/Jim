"""Scalar JAX array annotations with explicit float, integer or complex dtypes.

``FloatLike`` and ``ComplexLike`` also accept Python floats and complex numbers. Jaxtyping's ``Scalar`` and
``ScalarLike`` accept other dtypes, so they cannot replace these aliases.
"""

from jaxtyping import Array, Complex, Float, Int

type FloatScalar = Float[Array, ""]
type IntScalar = Int[Array, ""]
type ComplexScalar = Complex[Array, ""]
type FloatLike = float | Float[Array, ""]
type ComplexLike = complex | Complex[Array, ""]
