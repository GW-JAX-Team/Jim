import jax
import jax.numpy as jnp
import pytest

from jimgw.core.single_event.utils import complex_inner_product, inner_product


def _real_formula(h1, h2, psd, df):
    """Real-component definition, independent of the complex reduction."""
    product = h1.real * h2.real + h1.imag * h2.imag
    return 4.0 * jnp.sum(product / psd) * df


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("complex_h1", [False, True])
@pytest.mark.parametrize("complex_h2", [False, True])
def test_real_inner_product_values_and_gradients(dtype, complex_h1, complex_h2):
    h1 = jnp.array([0.0, 1.5, -2.0, 3.0, -0.5], dtype=dtype)
    h2 = jnp.array([2.0, -0.25, 0.0, 1.0, -1.5], dtype=dtype)
    if complex_h1:
        h1 = h1 + 1j * jnp.array([1.0, -2.0, 0.0, 0.5, 3.0], dtype=dtype)
    if complex_h2:
        h2 = h2 + 1j * jnp.array([-2.0, 0.0, 1.0, 2.5, -0.5], dtype=dtype)
    psd = jnp.array([0.5, 1.0, 2.0, 3.0, 8.0], dtype=dtype)
    df = jnp.asarray(0.25, dtype=dtype)
    args = h1, h2, psd, df
    tolerance = 2e-6 if dtype == jnp.float32 else 1e-12

    expected = _real_formula(*args)
    assert jnp.allclose(inner_product(*args), expected, rtol=tolerance, atol=tolerance)
    assert jnp.allclose(
        jax.jit(inner_product)(*args), expected, rtol=tolerance, atol=tolerance
    )

    expected_gradients = jax.grad(_real_formula, argnums=(0, 1, 2, 3))(*args)
    gradients = jax.jit(jax.grad(inner_product, argnums=(0, 1, 2, 3)))(*args)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        assert jnp.allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("complex_strain", [False, True])
def test_real_self_inner_product_values_and_gradients(complex_strain):
    strain = jnp.array([0.0, 0.0, 1.0, -2.0])
    if complex_strain:
        strain = strain + 1j * jnp.array([0.0, 1.0, 0.0, -3.0])
    psd = jnp.array([0.5, 1.0, 2.0, 4.0])
    df = jnp.asarray(0.5)

    def self_product(h, psd, df):
        return inner_product(h, h, psd, df)

    def reference(h, psd, df):
        return _real_formula(h, h, psd, df)

    args = strain, psd, df
    assert jnp.allclose(jax.jit(self_product)(*args), reference(*args))
    gradients = jax.jit(jax.grad(self_product, argnums=(0, 1, 2)))(*args)
    expected_gradients = jax.grad(reference, argnums=(0, 1, 2))(*args)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        assert jnp.all(jnp.isfinite(actual))
        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_complex_inner_product_preserves_imaginary_part_and_convention():
    h1 = jnp.array([1.0 + 2.0j, -0.5 + 1.5j])
    h2 = jnp.array([-1.0 + 0.5j, 2.0 - 3.0j])
    psd = jnp.array([0.5, 2.0])
    df = jnp.asarray(0.25)
    expected = -2.75 + 4.25j

    actual = jax.jit(complex_inner_product)(h1, h2, psd, df)
    assert jnp.abs(actual.imag) > 0.0
    assert jnp.allclose(actual, expected)
    assert jnp.allclose(actual, jnp.conj(complex_inner_product(h2, h1, psd, df)))
