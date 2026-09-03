"""Unit tests for the ``Ripple*`` waveform bindings.

Jim's waveforms are thin ``functools.partial`` bindings onto
``ripplegw.waveform(name)`` -- the physics (accuracy, distance scaling, mode
content, tidal parametrizations, ...) is tested in ripple itself and is not
re-tested here.  What these tests check is only that every binding jim exposes
works *in jim*: it constructs, is callable, and returns a well-formed
``{"p", "c"}`` polarization dict of the right shape with finite values --
including under ``jax.jit`` (jim JIT-compiles the likelihood, so every waveform
must be traceable).
"""

import functools

import jax
import jax.numpy as jnp
import pytest

import jimgw.core.single_event.waveform as waveform_module
from tests.utils import assert_all_finite

_SLOW_BINDINGS = frozenset(
    {
        "RippleIMRPhenomXAS_NRTidalv3",
        "RippleIMRPhenomXHM",
        "RippleIMRPhenomXP",
        "RippleIMRPhenomXP_NRTidalv3",
        "RippleIMRPhenomXPHM",
    }
)

WAVEFORM_CASES = [
    pytest.param(
        name,
        marks=pytest.mark.slow if name in _SLOW_BINDINGS else [],
    )
    for name, binding in vars(waveform_module).items()
    if isinstance(binding, functools.partial)
]

# One superset of source parameters; each waveform picks the keys that appear in
# its own ``parameter_names``.  Values are physically reasonable for a BNS so the
# tidal models are exercised in-domain; the CBC models ignore the tidal keys.
_ALL_PARAMS = {
    "M_c": 1.4,
    "eta": 0.245,
    "d_L": 400.0,
    "phase_c": 0.3,
    "iota": 0.7,
    "s1_x": 0.1,
    "s1_y": 0.05,
    "s1_z": 0.2,
    "s2_x": -0.05,
    "s2_y": 0.1,
    "s2_z": -0.15,
    "lambda_1": 400.0,
    "lambda_2": 300.0,
    # SineGaussian burst parameters
    "Q": 10.0,
    "f_0": 120.0,
    "hrss": 1e-21,
    "phase": 1.1,
    "e": 0.1,
}

_FD_FREQUENCIES = jnp.linspace(20.0, 1024.0, 256)
_TD_TIMES = jnp.arange(-1.0, 1.0, 1.0 / 2048.0)


def _build(name):
    """Instantiate ``name`` and return ``(waveform, evaluation_grid)``.

    SineGaussian is a time-domain burst (evaluated on a time grid centred at 0,
    constructed with no ``f_ref``); every other binding is frequency-domain.
    """
    constructor = getattr(waveform_module, name)
    if name == "RippleSineGaussian":
        return constructor(), _TD_TIMES
    return constructor(f_ref=20.0), _FD_FREQUENCIES


def _params_for(waveform):
    return {key: _ALL_PARAMS[key] for key in waveform.parameter_names}


@pytest.mark.parametrize("name", WAVEFORM_CASES)
def test_waveform_binding(name):
    """Each binding builds, evaluates to a finite {p, c} dict of the right shape,
    and is JIT-traceable with a result matching the eager call.
    """
    waveform, grid = _build(name)
    assert callable(waveform)

    params = _params_for(waveform)
    h = waveform(grid, params)
    h_jit = jax.jit(lambda g, p: waveform(g, p))(grid, params)

    assert set(h) == {"p", "c"}
    for polarization in ("p", "c"):
        assert h[polarization].shape == grid.shape
        assert jnp.any(jnp.abs(h[polarization]) > 0)
        assert_all_finite(h[polarization])
        assert_all_finite(h_jit[polarization])
        # strain amplitudes are ~1e-24, far below jnp.allclose's default atol,
        # so tie the tolerance to the polarization's own magnitude.
        atol = 1e-6 * jnp.max(jnp.abs(h[polarization]))
        assert jnp.allclose(h_jit[polarization], h[polarization], atol=atol, rtol=1e-6)
