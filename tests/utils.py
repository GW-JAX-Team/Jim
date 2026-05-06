"""Shared test utilities for all test modules.

This module contains common helper functions used across unit tests,
integration tests, and cross-validation tests.
"""

import jax.numpy as jnp


# Assertion helpers
def assert_all_finite(arr):
    """Assert all values in the array are finite.

    Args:
        arr: Array to check for finite values.

    Raises:
        AssertionError: If array contains non-finite values (NaN or inf).
    """
    assert jnp.all(jnp.isfinite(arr)), "Array contains non-finite values."


def assert_all_in_range(arr, low, high):
    """Assert all values in the array are within the specified range.

    Args:
        arr: Array to check.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Raises:
        AssertionError: If any values are outside [low, high].
    """
    assert jnp.all((arr >= low) & (arr <= high)), f"Values not in [{low}, {high}]"


def common_keys_allclose(
    dict_1: dict, dict_2: dict, atol: float = 1e-8, rtol: float = 1e-5
):
    """Check the common values between two result dictionaries are close.

    Args:
        dict_1: Result dictionary from the 1st transform.
        dict_2: Result dictionary from the 2nd transform.
        atol: Absolute tolerance between the two values, default 1e-8.
        rtol: Relative tolerance between the two values, default 1e-5.

    Returns:
        bool: True if all common values are close, False otherwise.

    Note:
        The default values for atol and rtol here follows those
        in jax.numpy.isclose, for their definitions, see:
        https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isclose.html
    """
    common_keys = set.intersection({*dict_1.keys()}, {*dict_2.keys()})
    if not common_keys:
        raise ValueError(
            f"No common keys found between dictionaries. "
            f"dict_1 keys: {set(dict_1.keys())}, dict_2 keys: {set(dict_2.keys())}"
        )
    tuples = jnp.array([[dict_1[key], dict_2[key]] for key in common_keys])
    tuple_array = jnp.swapaxes(tuples, 0, 1)
    return jnp.allclose(*tuple_array, atol=atol, rtol=rtol)


# Cross-validation specific utilities
def check_bilby_available() -> None:
    """Check if bilby is available.

    This is used by cross-validation tests to skip when bilby is not installed.

    Raises:
        ImportError: If bilby cannot be imported.
    """
    try:
        import bilby  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "cross_validation tests require bilby. Install with: pip install bilby"
        )
