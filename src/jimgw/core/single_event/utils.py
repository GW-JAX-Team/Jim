"""General single-event likelihood utilities.

Contains the noise-weighted inner product functions used by the likelihood
and the fixed-parameter application helper.

- For mass, spin, and sky/detector coordinate transforms, see :mod:`jimgw.core.single_event.transform_utils`.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Complex
from typing import Callable


def apply_fixed_parameters(
    params: dict[str, Float],
    fixed_parameters: dict[
        str, Float | Callable[[dict[str, Float]], Float | dict[str, Float]]
    ],
) -> dict[str, Float]:
    """Merge ``fixed_parameters`` into *params*, resolving callables in-place.

    For each entry in ``fixed_parameters``:

    - If the value is **callable**, it is called with the current *params* dict.
      If the result is a ``dict``, the value stored under the matching key is used;
      otherwise the scalar result is used directly.
    - If the value is **not callable**, it is inserted as-is.

    Args:
        params (dict[str, Float]): Parameter dictionary to update in-place. Callers
            that need to preserve the original should pass a copy.
        fixed_parameters (dict): Fixed overrides. Values may be scalar constants or
            callables ``f(params) -> Float | dict[str, Float]`` applied in insertion
            order.

    Returns:
        dict[str, Float]: The same ``params`` dict, mutated in-place with the fixed
            parameters applied.

    Raises:
        KeyError: If a callable returns a dict that does not contain the expected key.
    """
    for key, value in fixed_parameters.items():
        if callable(value):
            result = value(params)
            if isinstance(result, dict):
                if key not in result:
                    raise KeyError(
                        f"apply_fixed_parameters: callable {value!r} returned a dict "
                        f"that does not contain the expected key {key!r}. "
                        f"Returned keys: {list(result.keys())}"
                    )
                params[key] = result[key]
            else:
                params[key] = result
        else:
            params[key] = value
    return params


def complex_inner_product(
    h1: Float[Array, " n_freq"],
    h2: Float[Array, " n_freq"],
    psd: Float[Array, " n_freq"],
    df: Float,
) -> Complex:
    """Compute the complex noise-weighted inner product of two frequency-domain waveforms.

    The first waveform ``h1`` is complex-conjugated. The result is:

    .. math::

        \\langle h_1, h_2 \\rangle = 4 \\Delta f \\sum_k \\frac{h_1^*(f_k)\\, h_2(f_k)}{S_n(f_k)}

    Args:
        h1 (Float[Array, " n_freq"]): First waveform (complex array).
        h2 (Float[Array, " n_freq"]): Second waveform (complex array).
        psd (Float[Array, " n_freq"]): One-sided power spectral density at each
            frequency bin.
        df (Float): Frequency bin spacing in Hz.

    Returns:
        Complex: Complex noise-weighted inner product. When ``h2`` is the detector
            data, this is the complex match-filtered SNR.
    """
    return 4.0 * jnp.sum(jnp.conj(h1) * h2 / psd) * df


def inner_product(
    h1: Float[Array, " n_freq"],
    h2: Float[Array, " n_freq"],
    psd: Float[Array, " n_freq"],
    df: Float,
) -> Float:
    """Compute the real noise-weighted inner product of two frequency-domain waveforms.

    Returns the real part of `complex_inner_product`:

    .. math::

        (h_1 | h_2) = \\operatorname{Re}\\langle h_1, h_2 \\rangle
            = 4 \\Delta f \\sum_k \\operatorname{Re}\\!\\left[
              \\frac{h_1^*(f_k)\\, h_2(f_k)}{S_n(f_k)} \\right]

    Args:
        h1 (Float[Array, " n_freq"]): First waveform (complex array).
        h2 (Float[Array, " n_freq"]): Second waveform (complex array).
        psd (Float[Array, " n_freq"]): One-sided power spectral density at each
            frequency bin.
        df (Float): Frequency bin spacing in Hz.

    Returns:
        Float: Real noise-weighted inner product. When both waveforms are equal,
            this equals the optimal SNR squared.
    """
    return complex_inner_product(h1, h2, psd, df).real
