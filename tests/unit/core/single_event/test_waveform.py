"""Unit tests for waveform models."""

import jax
import jax.numpy as jnp
import pytest

from jimgw.core.single_event.waveform import (
    RippleIMRPhenomD,
    RippleIMRPhenomPv2,
    RippleTaylorF2,
    RippleIMRPhenomD_NRTidalv2,
)
from tests.utils import assert_all_finite


# Module-level fixtures
@pytest.fixture
def phenomD_params():
    """Standard parameter set for RippleIMRPhenomD tests."""
    return {
        "M_c": 30.0,
        "eta": 0.25,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "d_L": 400.0,
        "t_c": 0.0,
        "phase_c": 0.0,
        "iota": 0.0,
    }


@pytest.fixture
def phenomPv2_params():
    """Standard parameter set for RippleIMRPhenomPv2 tests."""
    return {
        "M_c": 30.0,
        "eta": 0.25,
        "s1_x": 0.1,
        "s1_y": 0.2,
        "s1_z": 0.3,
        "s2_x": -0.1,
        "s2_y": -0.2,
        "s2_z": -0.3,
        "d_L": 400.0,
        "t_c": 0.0,
        "phase_c": 0.0,
        "iota": 0.5,
    }


@pytest.fixture
def taylorF2_params_lambda12():
    """Standard parameter set for RippleTaylorF2 using lambda_1, lambda_2."""
    return {
        "M_c": 1.2,
        "eta": 0.24,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "lambda_1": 400.0,
        "lambda_2": 300.0,
        "d_L": 40.0,
        "phase_c": 0.0,
        "iota": 0.5,
    }


@pytest.fixture
def taylorF2_params_lambda_tildes():
    """Standard parameter set for RippleTaylorF2 using lambda tilde parameters."""
    return {
        "M_c": 1.2,
        "eta": 0.24,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "lambda_tilde": 350.0,
        "delta_lambda_tilde": 0.0,
        "d_L": 40.0,
        "phase_c": 0.0,
        "iota": 0.5,
    }


@pytest.fixture
def nrtidal_params_lambda12():
    """Standard parameter set for RippleIMRPhenomD_NRTidalv2 using lambda_1, lambda_2."""
    return {
        "M_c": 1.2,
        "eta": 0.24,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "lambda_1": 400.0,
        "lambda_2": 300.0,
        "d_L": 40.0,
        "phase_c": 0.0,
        "iota": 0.5,
    }


@pytest.fixture
def nrtidal_params_lambda_tildes():
    """Standard parameter set for RippleIMRPhenomD_NRTidalv2 using lambda tilde parameters."""
    return {
        "M_c": 1.2,
        "eta": 0.24,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "lambda_tilde": 350.0,
        "delta_lambda_tilde": 0.0,
        "d_L": 40.0,
        "phase_c": 0.0,
        "iota": 0.5,
    }


class TestRippleIMRPhenomD:
    """Test suite for RippleIMRPhenomD waveform model."""

    def test_initialization_and_call(self, phenomD_params):
        """Test waveform initialization and basic generation."""
        f_ref = 20.0
        waveform = RippleIMRPhenomD(f_ref=f_ref)
        assert waveform.f_ref == f_ref
        assert callable(waveform)

        # Generate waveform
        frequencies = jnp.linspace(20.0, 512.0, 100)
        h = waveform(frequencies, phenomD_params)

        # Check output structure
        assert "p" in h
        assert "c" in h
        assert h["p"].shape == frequencies.shape
        assert h["c"].shape == frequencies.shape

        # Check waveform is non-zero and finite
        assert jnp.any(jnp.abs(h["p"]) > 0)
        assert jnp.any(jnp.abs(h["c"]) > 0)
        assert_all_finite(h["p"])
        assert_all_finite(h["c"])

    def test_distance_scaling(self, phenomD_params):
        """Test that waveform amplitude scales inversely with distance."""
        waveform = RippleIMRPhenomD(f_ref=20.0)
        frequencies = jnp.linspace(20.0, 512.0, 50)

        # Near distance
        params_near = phenomD_params.copy()
        params_near["d_L"] = 100.0

        # Far distance (4x)
        params_far = phenomD_params.copy()
        params_far["d_L"] = 400.0

        h_near = waveform(frequencies, params_near)
        h_far = waveform(frequencies, params_far)

        # Amplitude should scale as 1/d_L (~4x for 4x distance)
        ratio = jnp.abs(h_near["p"]) / jnp.abs(h_far["p"])
        assert jnp.allclose(ratio, 4.0, rtol=0.1)

    def test_jit_compilation(self, phenomD_params):
        """Test that waveform generation can be JIT compiled."""
        waveform = RippleIMRPhenomD(f_ref=20.0)

        @jax.jit
        def generate_waveform(frequencies, params):
            return waveform(frequencies, params)

        frequencies = jnp.linspace(20.0, 512.0, 50)
        h = generate_waveform(frequencies, phenomD_params)

        assert_all_finite(h["p"])
        assert_all_finite(h["c"])


class TestRippleIMRPhenomPv2:
    """Test suite for RippleIMRPhenomPv2 waveform model."""

    def test_initialization_and_call(self, phenomPv2_params):
        """Test waveform initialization and basic generation."""
        f_ref = 20.0
        waveform = RippleIMRPhenomPv2(f_ref=f_ref)
        assert waveform.f_ref == f_ref
        assert callable(waveform)

        # Generate waveform
        frequencies = jnp.linspace(20.0, 512.0, 100)
        h = waveform(frequencies, phenomPv2_params)

        # Check output structure
        assert "p" in h
        assert "c" in h
        assert h["p"].shape == frequencies.shape
        assert h["c"].shape == frequencies.shape

        # Check waveform is non-zero and finite
        assert jnp.any(jnp.abs(h["p"]) > 0)
        assert jnp.any(jnp.abs(h["c"]) > 0)
        assert_all_finite(h["p"])
        assert_all_finite(h["c"])

    def test_distance_scaling(self, phenomPv2_params):
        """Test that waveform amplitude scales inversely with distance."""
        waveform = RippleIMRPhenomPv2(f_ref=20.0)
        frequencies = jnp.linspace(20.0, 512.0, 50)

        # Near distance
        params_near = phenomPv2_params.copy()
        params_near["d_L"] = 100.0

        # Far distance (4x)
        params_far = phenomPv2_params.copy()
        params_far["d_L"] = 400.0

        h_near = waveform(frequencies, params_near)
        h_far = waveform(frequencies, params_far)

        # Amplitude should scale as 1/d_L (~4x for 4x distance)
        ratio = jnp.abs(h_near["p"]) / jnp.abs(h_far["p"])
        assert jnp.allclose(ratio, 4.0, rtol=0.1)

    def test_jit_compilation(self, phenomPv2_params):
        """Test that waveform generation can be JIT compiled."""
        waveform = RippleIMRPhenomPv2(f_ref=20.0)

        @jax.jit
        def generate_waveform(frequencies, params):
            return waveform(frequencies, params)

        frequencies = jnp.linspace(20.0, 512.0, 50)
        h = generate_waveform(frequencies, phenomPv2_params)

        assert_all_finite(h["p"])
        assert_all_finite(h["c"])


class TestRippleTaylorF2:
    """Test suite for RippleTaylorF2 waveform model."""

    def test_initialization_with_lambda12(self, taylorF2_params_lambda12):
        """Test waveform with lambda_1, lambda_2 parametrization."""
        waveform = RippleTaylorF2(f_ref=20.0, use_lambda_tildes=False)
        assert waveform.f_ref == 20.0
        assert waveform.use_lambda_tildes is False

        frequencies = jnp.linspace(20.0, 512.0, 100)
        h = waveform(frequencies, taylorF2_params_lambda12)

        assert "p" in h
        assert "c" in h
        assert h["p"].shape == frequencies.shape
        assert jnp.any(jnp.abs(h["p"]) > 0)
        assert_all_finite(h["p"])
        assert_all_finite(h["c"])

    def test_initialization_with_lambda_tildes(self, taylorF2_params_lambda_tildes):
        """Test waveform with lambda_tilde, delta_lambda_tilde parametrization."""
        waveform = RippleTaylorF2(f_ref=20.0, use_lambda_tildes=True)
        assert waveform.use_lambda_tildes is True

        frequencies = jnp.linspace(20.0, 512.0, 100)
        h = waveform(frequencies, taylorF2_params_lambda_tildes)

        assert "p" in h
        assert "c" in h
        assert h["p"].shape == frequencies.shape
        assert jnp.any(jnp.abs(h["p"]) > 0)
        assert_all_finite(h["p"])
        assert_all_finite(h["c"])


class TestRippleIMRPhenomD_NRTidalv2:
    """Test suite for RippleIMRPhenomD_NRTidalv2 waveform model."""

    def test_initialization_with_lambda12(self, nrtidal_params_lambda12):
        """Test waveform with lambda_1, lambda_2 parametrization."""
        waveform = RippleIMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=False)
        assert waveform.f_ref == 20.0
        assert waveform.use_lambda_tildes is False
        assert waveform.no_taper is False

        frequencies = jnp.linspace(20.0, 512.0, 100)
        h = waveform(frequencies, nrtidal_params_lambda12)

        assert "p" in h
        assert "c" in h
        assert h["p"].shape == frequencies.shape
        assert jnp.any(jnp.abs(h["p"]) > 0)
        assert_all_finite(h["p"])
        assert_all_finite(h["c"])

    def test_initialization_with_lambda_tildes(self, nrtidal_params_lambda_tildes):
        """Test waveform with lambda_tilde, delta_lambda_tilde parametrization."""
        waveform = RippleIMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=True)
        assert waveform.use_lambda_tildes is True

        frequencies = jnp.linspace(20.0, 512.0, 100)
        h = waveform(frequencies, nrtidal_params_lambda_tildes)

        assert "p" in h
        assert "c" in h
        assert h["p"].shape == frequencies.shape
        assert jnp.any(jnp.abs(h["p"]) > 0)
        assert_all_finite(h["p"])
        assert_all_finite(h["c"])

    def test_no_taper_option(self, nrtidal_params_lambda12):
        """Test initialization with no_taper option."""
        waveform = RippleIMRPhenomD_NRTidalv2(
            f_ref=20.0, use_lambda_tildes=False, no_taper=True
        )
        assert waveform.no_taper is True

        frequencies = jnp.linspace(20.0, 512.0, 100)
        h = waveform(frequencies, nrtidal_params_lambda12)

        assert_all_finite(h["p"])
        assert_all_finite(h["c"])
