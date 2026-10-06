import logging
import time
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.constants import EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
from jimgw.core.single_event.data import PowerSpectrum
from jimgw.core.single_event.detector import get_CE, get_ET, get_H1, get_L1, get_V1
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from tests.utils import assert_all_in_range

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

GPS_TIME = 1126259462.0
DURATION = 4.0
F_MIN, F_MAX = 20.0, 1024.0
SAMPLING_FREQUENCY = F_MAX * 2

# Likelihood-space (fully expanded) parameters used as the reference injection.
REFERENCE_PARAMS = {
    "M_c": 28.0,
    "eta": 0.24,
    "s1_x": 0.3,
    "s1_y": 0.2,
    "s1_z": 0.1,
    "s2_x": -0.1,
    "s2_y": 0.2,
    "s2_z": -0.3,
    "d_L": 440.0,
    "phase_c": 0.0,
    "iota": 0.0,
    "ra": 1.5,
    "dec": 0.5,
    "psi": 0.3,
    "t_c": 0.0,
}


def make_detector(getter=get_H1):
    """A detector from *getter*, carrying the H1 fixture PSD whatever its name."""
    det = getter()
    psd = PowerSpectrum.from_file(str(FIXTURES_DIR / "GW150914_psd_H1.npz"))
    det.set_psd(psd)
    return det


def inject_reference(det, trigger_time=GPS_TIME, **overrides):
    """Inject the reference signal (zero noise) into *det*."""
    params = {**REFERENCE_PARAMS, **overrides}
    det.inject_signal(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        trigger_time=trigger_time,
        waveform_model=RippleIMRPhenomD(f_ref=20.0),
        parameters=params,
        f_min=F_MIN,
        f_max=F_MAX,
        zero_noise=True,
    )


def inject_noisy(det, rng_key):
    """Inject the reference signal plus noise drawn with *rng_key* into *det*."""
    det.inject_signal(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        trigger_time=GPS_TIME,
        waveform_model=RippleIMRPhenomD(f_ref=20.0),
        parameters=dict(REFERENCE_PARAMS),
        f_min=F_MIN,
        f_max=F_MAX,
        zero_noise=False,
        rng_key=rng_key,
    )


def injected_noise(getter, rng_key):
    """The noise that was injected: noisy data minus the zero-noise data."""
    clean = make_detector(getter)
    inject_reference(clean)
    noisy = make_detector(getter)
    inject_noisy(noisy, rng_key)
    return np.asarray(noisy.sliced_fd_data - clean.sliced_fd_data), noisy


# ---------------------------------------------------------------------------
# inject_signal tests
# ---------------------------------------------------------------------------


class TestInjectSignal:
    """Tests for inject_signal: core behavior and the transform pipeline."""

    # ------------------------------------------------------------------
    # Core behavior
    # ------------------------------------------------------------------

    def test_zero_noise_creates_data(self):
        """Data object is populated after a zero-noise injection."""
        det = make_detector()
        inject_reference(det)

        assert det.data is not None
        assert len(det.data.td) == int(DURATION * SAMPLING_FREQUENCY)
        assert det.data.start_time == GPS_TIME - DURATION + 2.0

    def test_zero_noise_signal_nonzero_in_band(self):
        """Injected signal is non-zero inside the frequency band."""
        det = make_detector()
        inject_reference(det)

        assert jnp.any(jnp.abs(det.sliced_fd_data) > 0)

    def test_zero_noise_frequency_bounds_respected(self):
        """Sliced frequencies lie within the requested band."""
        det = make_detector()
        inject_reference(det)

        assert_all_in_range(det.sliced_frequencies, F_MIN, F_MAX)

    def test_noisy_injection_differs_from_zero_noise(self):
        """Adding noise produces data that differs from the zero-noise case."""
        det_clean = make_detector()
        inject_reference(det_clean)

        det_noisy = make_detector()
        params = dict(REFERENCE_PARAMS)
        det_noisy.inject_signal(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            waveform_model=RippleIMRPhenomD(f_ref=20.0),
            parameters=params,
            f_min=F_MIN,
            f_max=F_MAX,
            zero_noise=False,
            rng_key=jax.random.key(42),
        )

        assert not jnp.allclose(
            det_clean.sliced_fd_data,
            det_noisy.sliced_fd_data,
            rtol=1e-05,
            atol=1e-23,
        )


# ---------------------------------------------------------------------------
# Injected noise: a function of (rng_key, detector name) and nothing else
# ---------------------------------------------------------------------------


class TestInjectedNoise:
    KEY = jax.random.key(42)

    def test_noisy_injection_requires_rng_key(self):
        det = make_detector()
        psd = det.psd
        with pytest.raises(ValueError, match="rng_key is required"):
            det.inject_signal(
                duration=DURATION,
                sampling_frequency=SAMPLING_FREQUENCY,
                trigger_time=GPS_TIME,
                waveform_model=RippleIMRPhenomD(f_ref=20.0),
                parameters=dict(REFERENCE_PARAMS),
                f_min=F_MIN,
                f_max=F_MAX,
                zero_noise=False,
            )
        # The failed call must leave the detector as it was.
        assert det.data.is_empty
        assert det.psd is psd
        assert det.frequency_bounds == (0.0, jnp.inf)

    def test_same_key_gives_identical_data(self):
        a, b = make_detector(), make_detector()
        inject_noisy(a, self.KEY)
        inject_noisy(b, self.KEY)
        np.testing.assert_array_equal(a.sliced_fd_data, b.sliced_fd_data)

    def test_different_keys_give_different_noise(self):
        a, b = make_detector(), make_detector()
        inject_noisy(a, jax.random.key(1))
        inject_noisy(b, jax.random.key(2))
        assert np.max(np.abs(np.asarray(a.sliced_fd_data - b.sliced_fd_data))) > 0

    def test_data_does_not_depend_on_the_clock(self, monkeypatch):
        a, b = make_detector(), make_detector()
        monkeypatch.setattr(time, "time", lambda: 1.0e9)
        inject_noisy(a, self.KEY)
        monkeypatch.setattr(time, "time", lambda: 2.0e9)
        inject_noisy(b, self.KEY)
        np.testing.assert_array_equal(a.sliced_fd_data, b.sliced_fd_data)

    def test_one_key_gives_every_detector_its_own_noise(self):
        # All four detectors carry the same PSD here, so identical random draws would
        # give identical noise: the detector name is what tells them apart.  V1 and
        # CE have name tags above 2**31.  The noise is whitened first: the raw noise
        # is dominated by a few low-frequency bins, which makes its correlation noisy.
        whitened = {}
        for getter in (get_H1, get_L1, get_V1, get_CE):
            noise, det = injected_noise(getter, self.KEY)
            whitened[getter.__name__] = noise / np.sqrt(np.asarray(det.sliced_psd))
        for (name_a, a), (name_b, b) in combinations(whitened.items(), 2):
            correlation = np.abs(np.vdot(a, b)) / (
                np.linalg.norm(a) * np.linalg.norm(b)
            )
            assert correlation < 0.1, (name_a, name_b, correlation)

    def test_noise_does_not_depend_on_injection_order(self):
        def inject_in_order(getters):
            data = {}
            for getter in getters:
                det = make_detector(getter)
                inject_noisy(det, self.KEY)
                data[getter.__name__] = np.asarray(det.sliced_fd_data)
            return data

        forward = inject_in_order([get_H1, get_L1])
        backward = inject_in_order([get_L1, get_H1])
        for name, data in forward.items():
            np.testing.assert_array_equal(data, backward[name])

    def test_injected_noise_matches_the_psd_the_likelihood_reads(self):
        # Whitened by the PSD the likelihood uses (sliced_psd), the injected noise has
        # unit variance in its real and imaginary parts: noise and analysis share a PSD.
        noise, det = injected_noise(get_H1, self.KEY)
        whitened = noise / np.sqrt(np.asarray(det.sliced_psd) * DURATION / 4)
        assert np.var(whitened.real) == pytest.approx(1.0, abs=0.1)
        assert np.var(whitened.imag) == pytest.approx(1.0, abs=0.1)


class TestInjectionRequirements:
    """What inject_signal needs, with and without noise."""

    @pytest.mark.parametrize("zero_noise", [True, False])
    def test_psd_is_required_in_both_noise_modes(self, zero_noise):
        det = get_H1()  # no PSD set
        with pytest.raises(ValueError, match="No PSD is set on detector H1"):
            det.inject_signal(
                duration=DURATION,
                sampling_frequency=SAMPLING_FREQUENCY,
                trigger_time=GPS_TIME,
                waveform_model=RippleIMRPhenomD(f_ref=20.0),
                parameters=dict(REFERENCE_PARAMS),
                f_min=F_MIN,
                f_max=F_MAX,
                zero_noise=zero_noise,
                rng_key=None if zero_noise else jax.random.key(0),
            )
        # The failed call must leave the detector as it was.
        assert det.data.is_empty
        assert det.frequency_bounds == (0.0, jnp.inf)

    def test_zero_noise_needs_no_rng_key(self, caplog, monkeypatch):
        # The "jimgw" logger does not propagate, which hides records from caplog.
        monkeypatch.setattr(logging.getLogger("jimgw"), "propagate", True)
        det = make_detector()
        with caplog.at_level("WARNING"):
            inject_reference(det)  # zero_noise=True and no key
        assert not [r for r in caplog.records if "rng_key" in r.message]
        assert not det.data.is_empty

    def test_rng_key_with_zero_noise_is_ignored_with_a_warning(
        self, caplog, monkeypatch
    ):
        # No noise is drawn, so the key is useless: warn, and leave the data alone.
        monkeypatch.setattr(logging.getLogger("jimgw"), "propagate", True)
        with_key, without_key = make_detector(), make_detector()
        inject_reference(without_key)
        with caplog.at_level("WARNING"):
            with_key.inject_signal(
                duration=DURATION,
                sampling_frequency=SAMPLING_FREQUENCY,
                trigger_time=GPS_TIME,
                waveform_model=RippleIMRPhenomD(f_ref=20.0),
                parameters=dict(REFERENCE_PARAMS),
                f_min=F_MIN,
                f_max=F_MAX,
                zero_noise=True,
                rng_key=jax.random.key(0),
            )
        np.testing.assert_array_equal(
            with_key.sliced_fd_data, without_key.sliced_fd_data
        )


# ---------------------------------------------------------------------------
# ET geometry tests
# ---------------------------------------------------------------------------


class TestET:
    """Tests for get_ET(): geometric consistency of the triangular ET configuration."""

    ET_ARM_LENGTH_M = 1e4  # 10 km

    def setup_method(self):
        self.ifos = get_ET()

    def test_returns_three_detectors(self):
        """get_ET returns exactly three GroundBased2G instances."""
        assert len(self.ifos) == 3

    def test_detector_names(self):
        """Sub-detectors are named ET1, ET2, ET3 in order."""
        assert [ifo.name for ifo in self.ifos] == ["ET1", "ET2", "ET3"]

    def test_arm_opening_angle_is_60_degrees(self):
        """Each sub-detector has 60° (π/3) between its x and y arms."""
        for ifo in self.ifos:
            delta = ifo.yarm_azimuth - ifo.xarm_azimuth
            assert abs(delta - np.pi / 3) < 1e-10, (
                f"{ifo.name}: arm opening angle is {np.degrees(delta):.4f}°, expected 60°"
            )

    def test_arms_rotated_240_degrees_between_detectors(self):
        """Consecutive sub-detectors have arm azimuths rotated by 240° (4π/3 rad)."""
        rotation = (4 / 3) * np.pi
        for i in range(2):
            dx = self.ifos[i + 1].xarm_azimuth - self.ifos[i].xarm_azimuth
            dy = self.ifos[i + 1].yarm_azimuth - self.ifos[i].yarm_azimuth
            assert abs(dx - rotation) < 1e-10, (
                f"ET{i + 1}→ET{i + 2} xarm rotation: {dx:.6f} rad, expected {rotation:.6f} rad"
            )
            assert abs(dy - rotation) < 1e-10, (
                f"ET{i + 1}→ET{i + 2} yarm rotation: {dy:.6f} rad, expected {rotation:.6f} rad"
            )

    def test_vertex_separations_match_arm_length(self):
        """
        Haversine distance between every pair of ET vertex positions should
        equal the arm length (10 km) to within 50 m.

        This checks both the propagation formula and that the triangle closes,
        following the approach used in bilby's TriangularInterferometerTest.
        """
        # Use the same WGS-84 radius get_ET uses: computed at ET1's latitude
        # (the initial latitude, before any vertex propagation).
        _a = EARTH_SEMI_MAJOR_AXIS / 1e3
        _b = EARTH_SEMI_MINOR_AXIS / 1e3
        lat0 = float(self.ifos[0].latitude)
        R = (
            _a * _b / np.sqrt(_a**2 * np.sin(lat0) ** 2 + _b**2 * np.cos(lat0) ** 2)
        ) * 1e3
        for ifo_a, ifo_b in combinations(self.ifos, 2):
            lat1 = float(ifo_a.latitude)
            lon1 = float(ifo_a.longitude)
            lat2 = float(ifo_b.latitude)
            lon2 = float(ifo_b.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            assert abs(dist - self.ET_ARM_LENGTH_M) < 50.0, (
                f"{ifo_a.name}↔{ifo_b.name}: {dist:.0f} m "
                f"(expected ~{self.ET_ARM_LENGTH_M:.0f} m ± 50 m)"
            )
