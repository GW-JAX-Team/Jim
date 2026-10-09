import logging
import time
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.constants import (
    C_SI,
    EARTH_ROTATION_RATE,
    EARTH_SEMI_MAJOR_AXIS,
    EARTH_SEMI_MINOR_AXIS,
    MTSUN,
)
from jimgw.core.single_event import detector as detector_module
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import (
    GroundBased2G,
    GroundBased3G,
    get_CE,
    get_ET,
    get_H1,
    get_L1,
    get_V1,
    time_to_merger,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
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
        """get_ET returns exactly three GroundBased3G instances."""
        assert len(self.ifos) == 3
        assert all(isinstance(ifo, GroundBased3G) for ifo in self.ifos)

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


# ---------------------------------------------------------------------------
# Next-generation detectors (GroundBased3G)
# ---------------------------------------------------------------------------


ROTATION_TRIGGER_TIME = 1400000000.0

# Aligned-spin binary neutron star in likelihood space.
BNS_PARAMS = {
    "M_c": 1.22,
    "eta": 0.2495,
    "s1_z": 0.03,
    "s2_z": -0.02,
    "ra": 1.1,
    "dec": -0.4,
    "psi": 0.7,
    "t_c": 0.3,
    "trigger_time": ROTATION_TRIGGER_TIME,
    # Wrapped, so that shifting it by 2 pi is exact to rounding of order one.
    "gmst": float(compute_gmst(ROTATION_TRIGGER_TIME)) % (2 * np.pi),
}


def _start_at(det, start_time):
    """Give *det* an empty data segment beginning at *start_time*."""
    det.set_data(Data(td=jnp.zeros(8), delta_t=1.0, start_time=start_time))
    return det


def _as_2g(det):
    """A GroundBased2G with the geometry of *det*."""
    return GroundBased2G(
        det.name,
        latitude=det.latitude,
        longitude=det.longitude,
        elevation=det.elevation,
        xarm_azimuth=det.xarm_azimuth,
        yarm_azimuth=det.yarm_azimuth,
        xarm_tilt=det.xarm_tilt,
        yarm_tilt=det.yarm_tilt,
        modes="".join(p.name for p in det.polarization_mode),
    )


def _unit_polarizations(frequency):
    """Unit-amplitude plus and cross polarizations with distinct phases."""
    plus = jnp.exp(1j * 2 * jnp.pi * frequency * 0.01)
    return {"p": plus, "c": 0.5j * plus}


def _et1_3g(**kwargs):
    return _start_at(
        GroundBased3G.from_detector(get_ET()[0], **kwargs),
        ROTATION_TRIGGER_TIME - 100,
    )


def _newtonian_time_to_merger(frequency, M_c):
    """Leading-order chirp time, computed independently of time_to_merger."""
    M_c_s = M_c * MTSUN
    return 5 / 256 * M_c_s * (np.pi * M_c_s * frequency) ** (-8 / 3)


def _round_trip_transfer(x, mu, n=200001):
    """Round-trip average of a unit plane wave along an arm, by quadrature.

    A photon leaves the vertex at -L/c, reaches the end mirror at 0 and returns
    at L/c, while the metric perturbation along the arm is exp(2 pi i f (t -
    k.r / c)) with mu = k.a. Distances are in units of L and times in L/c, so
    the frequency is x = f L / c.
    """
    s = np.linspace(0.0, 1.0, n)
    outbound = np.exp(2j * np.pi * x * ((s - 1.0) - mu * s))
    inbound = np.exp(2j * np.pi * x * (s - mu * (1.0 - s)))
    return 0.5 * (np.trapezoid(outbound, s) + np.trapezoid(inbound, s))


class TestTimeToMerger:
    """2PN stationary-phase time to merger."""

    def test_bns_time_to_merger_from_5_hz(self):
        """A 1.22 M_sun chirp mass BNS spends about 107 min above 5 Hz (arXiv:2503.09627)."""
        tau = float(time_to_merger(jnp.array(5.0), 1.22, 0.25)) / 60
        assert 106.0 < tau < 108.0, f"tau(5 Hz) = {tau:.2f} min"

    def test_reduces_to_newtonian_at_low_frequency(self):
        """The PN series tends to the leading-order chirp time as f -> 0."""
        f = jnp.array(1e-4)
        tau = float(time_to_merger(f, 1.22, 0.2495, 0.3, 0.2))
        tau_n = _newtonian_time_to_merger(1e-4, 1.22)
        np.testing.assert_allclose(tau, tau_n, rtol=1e-4, atol=0.0)

    def test_first_post_newtonian_coefficient(self):
        """(tau / tau_N - 1) / x tends to the 1PN coefficient 4/3 (743/336 + 11 eta / 4)."""
        M_c, eta = 1.22, 0.2
        M = M_c / eta ** (3 / 5)
        # Low enough that the 1.5PN term, relatively O(x^1/2), stays below 1e-3.
        f = 1e-9
        x = (np.pi * M * MTSUN * f) ** (2 / 3)
        tau = float(time_to_merger(jnp.array(f), M_c, eta))
        measured = (tau / _newtonian_time_to_merger(f, M_c) - 1) / x
        expected = 4 / 3 * (743 / 336 + 11 / 4 * eta)
        np.testing.assert_allclose(measured, expected, rtol=1e-3, atol=0.0)

    def test_aligned_spins_lengthen_the_inspiral(self):
        """Spins aligned with the orbit slow the inspiral (orbital hang-up)."""
        f = jnp.array(10.0)
        aligned = float(time_to_merger(f, 1.22, 0.2495, 0.5, 0.5))
        non_spinning = float(time_to_merger(f, 1.22, 0.2495))
        anti_aligned = float(time_to_merger(f, 1.22, 0.2495, -0.5, -0.5))
        assert anti_aligned < non_spinning < aligned

    def test_decreases_with_frequency(self):
        """Higher frequencies are emitted later, closer to merger."""
        f = jnp.geomspace(2.0, 2048.0, 4000)
        tau = time_to_merger(f, 1.22, 0.2495, 0.03, -0.02)
        assert jnp.all(jnp.diff(tau) < 0)
        assert jnp.all(tau > 0)

    @pytest.mark.parametrize("mode", [1, 3, 4, -3])
    def test_mode_rescales_the_frequency(self, mode):
        """tau_m(f) = tau_22(2 f / |m|)."""
        f = jnp.geomspace(5.0, 500.0, 50)
        np.testing.assert_allclose(
            time_to_merger(f, 1.22, 0.2495, 0.1, 0.0, mode=mode),
            time_to_merger(2 * f / abs(mode), 1.22, 0.2495, 0.1, 0.0),
            rtol=1e-14,
            atol=0.0,
        )

    def test_zero_mode_is_rejected(self):
        """m = 0 has no frequency-to-time mapping; it must not divide by zero."""
        with pytest.raises(ValueError, match="non-zero"):
            time_to_merger(jnp.array([10.0]), 1.22, 0.2495, mode=0)


class TestGroundBased3G:
    """Construction of GroundBased3G, and its reduction to GroundBased2G."""

    frequency = jnp.geomspace(2.0, 2048.0, 512)

    def test_finite_arm_length_needs_a_positive_arm_length(self):
        with pytest.raises(ValueError, match="arm_length"):
            GroundBased3G("X", finite_arm_length=True)

    def test_rotation_is_on_and_finite_arm_length_off_by_default(self):
        det = GroundBased3G("X")
        assert det.earth_rotation
        assert not det.finite_arm_length

    def test_from_detector_copies_the_geometry(self):
        source = get_ET()[2]
        det = GroundBased3G.from_detector(
            source, arm_length=1e4, finite_arm_length=True
        )
        assert det.name == source.name
        assert jnp.array_equal(det.tensor, source.tensor)
        assert jnp.array_equal(det.vertex, source.vertex)
        assert det.arm_length == 1e4
        assert det.finite_arm_length and det.earth_rotation

    def test_from_detector_copies_the_polarization_modes(self):
        det = GroundBased3G.from_detector(GroundBased2G("X", modes="pcbl"))
        assert [p.name for p in det.polarization_mode] == ["p", "c", "b", "l"]

    @pytest.mark.parametrize("getter", [lambda: get_ET()[1], get_H1, get_CE])
    def test_reduces_to_2g_without_rotation(self, getter):
        """With rotation and finite arm length off, the response equals GroundBased2G bit for bit."""
        det_2g = _start_at(_as_2g(getter()), ROTATION_TRIGGER_TIME - 100)
        det_3g = _start_at(
            GroundBased3G.from_detector(det_2g, earth_rotation=False),
            ROTATION_TRIGGER_TIME - 100,
        )
        h_sky = _unit_polarizations(self.frequency)
        expected = det_2g.fd_response(self.frequency, h_sky, BNS_PARAMS)
        actual = det_3g.fd_response(self.frequency, h_sky, BNS_PARAMS)
        assert jnp.array_equal(actual, expected)


class TestEarthRotation:
    """The response follows the Earth's rotation across the signal."""

    frequency = jnp.geomspace(2.0, 2048.0, 512)

    def test_rotation_rate_matches_sidereal_day(self):
        """2 pi x 1.0027379 / 86400 s = 7.292e-5 rad/s."""
        np.testing.assert_allclose(
            EARTH_ROTATION_RATE, 2 * np.pi * 1.0027379 / 86400, rtol=1e-7, atol=0.0
        )

    def test_rotation_rate_matches_gmst(self):
        """GMST advances at the rotation rate over a day."""
        t0 = ROTATION_TRIGGER_TIME
        advance = float(compute_gmst(t0 + 86400.0) - compute_gmst(t0))
        np.testing.assert_allclose(
            advance, EARTH_ROTATION_RATE * 86400.0, rtol=1e-7, atol=0.0
        )

    def test_rotation_changes_the_response(self):
        """A BNS from 2 Hz sweeps hours of rotation; the response must change."""
        on = _et1_3g()
        off = _et1_3g(earth_rotation=False)
        h_sky = _unit_polarizations(self.frequency)
        diff = jnp.abs(
            on.fd_response(self.frequency, h_sky, BNS_PARAMS)
            - off.fd_response(self.frequency, h_sky, BNS_PARAMS)
        )
        # Hours of rotation at 2 Hz, but only seconds near merger.
        assert diff[0] > 0.1
        assert diff[-1] < 1e-2 * diff[0]

    def test_each_frequency_uses_the_2g_response_at_its_sidereal_angle(self):
        """The response at f is the 2G response with gmst replaced by GMST(f)."""
        det_3g = _et1_3g()
        det_2g = _start_at(_as_2g(get_ET()[0]), ROTATION_TRIGGER_TIME - 100)
        h_sky = _unit_polarizations(self.frequency)
        response = det_3g.fd_response(self.frequency, h_sky, BNS_PARAMS)
        gmst_f = det_3g.gmst_at_frequency(self.frequency, BNS_PARAMS)
        for i in [0, 100, 300, 511]:
            f_i = self.frequency[i : i + 1]
            h_i = {k: v[i : i + 1] for k, v in h_sky.items()}
            expected = det_2g.fd_response(f_i, h_i, {**BNS_PARAMS, "gmst": gmst_f[i]})
            np.testing.assert_allclose(response[i], expected[0], rtol=1e-12, atol=0.0)

    def test_sidereal_angle_matches_gmst_at_emission(self):
        """GMST(f) agrees with the GMST of the emission time t_trigger + t_c - tau(f)."""
        det = _et1_3g()
        gmst_f = det.gmst_at_frequency(self.frequency, BNS_PARAMS)
        tau = time_to_merger(
            self.frequency,
            BNS_PARAMS["M_c"],
            BNS_PARAMS["eta"],
            BNS_PARAMS["s1_z"],
            BNS_PARAMS["s2_z"],
        )
        emission = ROTATION_TRIGGER_TIME + BNS_PARAMS["t_c"] - tau
        for i in [0, 50, 200, 511]:
            # tau(2 Hz) is about 21 hours, so this also checks the linearisation.
            wrapped_difference = np.angle(
                np.exp(1j * (gmst_f[i] - compute_gmst(emission[i])))
            )
            assert abs(wrapped_difference) < 1e-6

    def test_sidereal_angle_increases_with_frequency(self):
        """Later emission means a larger sidereal angle, anchored at merger."""
        det = _et1_3g()
        gmst_f = det.gmst_at_frequency(self.frequency, BNS_PARAMS)
        at_merger = (
            np.mod(BNS_PARAMS["gmst"], 2 * np.pi)
            + EARTH_ROTATION_RATE * BNS_PARAMS["t_c"]
        )
        assert jnp.all(jnp.diff(gmst_f) > 0)
        assert jnp.all(gmst_f < at_merger)
        # The zero-frequency bin has no emission time; it is put at merger.
        dc = det.gmst_at_frequency(jnp.array([0.0, 5.0]), BNS_PARAMS)
        assert jnp.all(jnp.isfinite(dc))
        np.testing.assert_allclose(dc[0], at_merger, rtol=1e-15, atol=0.0)

    def test_full_turn_of_gmst_leaves_the_response_unchanged(self):
        """The response is 2 pi periodic in the sidereal angle."""
        det = _et1_3g(finite_arm_length=True, arm_length=1e4)
        h_sky = _unit_polarizations(self.frequency)
        shifted = {**BNS_PARAMS, "gmst": BNS_PARAMS["gmst"] + 2 * np.pi}
        np.testing.assert_allclose(
            det.fd_response(self.frequency, h_sky, shifted),
            det.fd_response(self.frequency, h_sky, BNS_PARAMS),
            rtol=1e-11,
            atol=0.0,
        )

    def test_only_the_hour_angle_matters(self):
        """Rotating the sky and the Earth together leaves the response unchanged."""
        det = _et1_3g(finite_arm_length=True, arm_length=1e4)
        h_sky = _unit_polarizations(self.frequency)
        delta = 0.9
        shifted = {
            **BNS_PARAMS,
            "ra": BNS_PARAMS["ra"] + delta,
            "gmst": BNS_PARAMS["gmst"] + delta,
        }
        np.testing.assert_allclose(
            det.fd_response(self.frequency, h_sky, shifted),
            det.fd_response(self.frequency, h_sky, BNS_PARAMS),
            rtol=1e-11,
            atol=0.0,
        )


class TestFiniteArmLength:
    @pytest.mark.parametrize("x", [0.03, 0.4, 1.7])
    @pytest.mark.parametrize("mu", [-0.8, -0.1, 0.0, 0.55, 1.0])
    def test_transfer_function_matches_light_travel_integral(self, x, mu):
        """The transfer function equals the round-trip integral, sign convention included."""
        np.testing.assert_allclose(
            complex(GroundBased3G._arm_transfer_function(x, mu)),
            _round_trip_transfer(x, mu),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_long_wavelength_limit(self):
        """The transfer function is one when the arm is short compared to the wavelength."""
        y = jnp.linspace(-1.0, 1.0, 11)
        assert jnp.array_equal(
            GroundBased3G._arm_transfer_function(0.0, y), jnp.ones_like(y)
        )
        frequency = jnp.geomspace(2.0, 2048.0, 256)
        h_sky = _unit_polarizations(frequency)
        # The leading correction is O(f L / c), about 1e-14 here.
        short = _et1_3g(finite_arm_length=True, arm_length=1e-6)
        point = _et1_3g()
        np.testing.assert_allclose(
            short.fd_response(frequency, h_sky, BNS_PARAMS),
            point.fd_response(frequency, h_sky, BNS_PARAMS),
            rtol=1e-8,
            atol=0.0,
        )

    def test_arm_patterns_use_the_propagation_direction(self):
        """Each arm is weighted by the transfer function of k.a, with k pointing away from the source."""
        det = _et1_3g(finite_arm_length=True, arm_length=1e4)
        frequency = jnp.array([50.0, 800.0, 3000.0])
        ra, dec, psi, gmst = 0.4, 0.9, 1.3, 2.2
        patterns = det._finite_arm_antenna_pattern(frequency, ra, dec, psi, gmst)

        hour_angle = ra - gmst
        k = -np.array(
            [
                np.cos(dec) * np.cos(hour_angle),
                np.cos(dec) * np.sin(hour_angle),
                np.sin(dec),
            ]
        )
        x_arm, y_arm = (np.asarray(a) for a in det.arms)
        x = np.asarray(frequency) * 1e4 / C_SI
        for polarization in det.polarization_mode:
            e = np.asarray(polarization.tensor_from_sky(ra, dec, psi, gmst))
            f_x = 0.5 * x_arm @ e @ x_arm
            f_y = 0.5 * y_arm @ e @ y_arm
            expected = [
                f_x * _round_trip_transfer(xi, k @ x_arm)
                - f_y * _round_trip_transfer(xi, k @ y_arm)
                for xi in x
            ]
            np.testing.assert_allclose(
                patterns[polarization.name], expected, rtol=1e-7, atol=0.0
            )

    def test_finite_arm_length_without_rotation(self, monkeypatch):
        """Finite arm length alone is the rotating response for a still Earth, and needs no masses."""
        frequency = jnp.geomspace(2.0, 2048.0, 64)
        h_sky = _unit_polarizations(frequency)
        static = _et1_3g(finite_arm_length=True, arm_length=1e4, earth_rotation=False)
        rotating = _et1_3g(finite_arm_length=True, arm_length=1e4)
        point = _et1_3g(earth_rotation=False)

        response = static.fd_response(frequency, h_sky, BNS_PARAMS)
        monkeypatch.setattr(detector_module, "EARTH_ROTATION_RATE", 0.0)
        np.testing.assert_allclose(
            rotating.fd_response(frequency, h_sky, BNS_PARAMS),
            response,
            rtol=1e-12,
            atol=0.0,
        )
        # The arms are 7% of a wavelength long at 2 kHz.
        assert (
            jnp.abs(response - point.fd_response(frequency, h_sky, BNS_PARAMS))[-1]
            > 1e-2
        )
        sky_only = {
            k: v
            for k, v in BNS_PARAMS.items()
            if k not in ("M_c", "eta", "s1_z", "s2_z")
        }
        assert jnp.array_equal(static.fd_response(frequency, h_sky, sky_only), response)


class TestJaxCompatibility:
    """The response under jit and vmap, and its gradient, as the samplers use it."""

    # The zero-frequency bin has no emission time and must not poison the gradient.
    frequency = jnp.concatenate([jnp.zeros(1), jnp.geomspace(2.0, 2048.0, 63)])

    @pytest.mark.parametrize(
        "kwargs", [{}, {"finite_arm_length": True, "arm_length": 1e4}]
    )
    def test_jit_matches_eager(self, kwargs):
        det = _et1_3g(**kwargs)
        h_sky = _unit_polarizations(self.frequency)
        np.testing.assert_allclose(
            jax.jit(det.fd_response)(self.frequency, h_sky, BNS_PARAMS),
            det.fd_response(self.frequency, h_sky, BNS_PARAMS),
            rtol=0.0,
            atol=1e-12,
        )

    def test_vmap_matches_a_loop(self):
        det = _et1_3g()
        h_sky = _unit_polarizations(self.frequency)
        M_c = BNS_PARAMS["M_c"] * (1 + 1e-3 * jnp.arange(4))

        def response(m):
            return det.fd_response(self.frequency, h_sky, {**BNS_PARAMS, "M_c": m})

        np.testing.assert_allclose(
            jax.vmap(response)(M_c),
            jnp.stack([response(m) for m in M_c]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_gradient_is_finite_with_a_zero_frequency_bin(self):
        det = _et1_3g(finite_arm_length=True, arm_length=1e4)
        h_sky = _unit_polarizations(self.frequency)

        def loss(params):
            return jnp.sum(jnp.real(det.fd_response(self.frequency, h_sky, params)))

        grads = jax.grad(loss)({k: jnp.asarray(v) for k, v in BNS_PARAMS.items()})
        for name in ("M_c", "eta", "s1_z", "s2_z", "t_c", "ra", "dec", "psi"):
            assert jnp.isfinite(grads[name]), name

    @pytest.mark.parametrize(
        "option, names",
        [
            ({}, ("M_c", "t_c")),
            (
                {"earth_rotation": False, "finite_arm_length": True, "arm_length": 1e4},
                ("ra", "dec", "psi"),
            ),
        ],
    )
    def test_gradient_of_an_option_matches_finite_differences(self, option, names):
        """The gradient of what the option adds to the response."""

        def detector(**options):
            det = GroundBased3G.from_detector(get_ET()[0], **options)
            return _start_at(det, ROTATION_TRIGGER_TIME - 4)

        with_option, without = detector(**option), detector(earth_rotation=False)
        h_sky = _unit_polarizations(self.frequency)
        reference = without.fd_response(self.frequency, h_sky, BNS_PARAMS)

        def added(params):
            difference = with_option.fd_response(
                self.frequency, h_sky, params
            ) - without.fd_response(self.frequency, h_sky, params)
            return jnp.sum(jnp.real(jnp.conj(reference) * difference))

        params = {k: jnp.asarray(v) for k, v in BNS_PARAMS.items()}
        grad = jax.grad(added)(params)
        step = 1e-7
        for name in names:
            up, down = ({**params, name: params[name] + s * step} for s in (1, -1))
            finite_difference = (added(up) - added(down)) / (2 * step)
            np.testing.assert_allclose(
                grad[name], finite_difference, rtol=1e-5, atol=0.0, err_msg=name
            )


class TestNextGenerationPresets:
    """get_ET and get_CE are GroundBased3G detectors that respond as GroundBased2G by default."""

    @staticmethod
    def _detectors(name, **kwargs):
        return get_ET(**kwargs) if name == "ET" else [get_CE(**kwargs)]

    @pytest.mark.parametrize("name, arm_length", [("ET", 1e4), ("CE", 4e4)])
    def test_arm_length_and_default_options(self, name, arm_length):
        for det in self._detectors(name):
            assert isinstance(det, GroundBased3G)
            assert det.arm_length == arm_length
            assert not (det.earth_rotation or det.finite_arm_length)

    @pytest.mark.parametrize("name", ["ET", "CE"])
    def test_default_response_is_the_2g_response(self, name):
        """As shipped, the presets reproduce the former GroundBased2G ones bit for bit."""
        frequency = jnp.geomspace(2.0, 2048.0, 64)
        h_sky = _unit_polarizations(frequency)
        for det in self._detectors(name):
            det_2g = _start_at(_as_2g(det), ROTATION_TRIGGER_TIME - 100)
            _start_at(det, ROTATION_TRIGGER_TIME - 100)
            assert jnp.array_equal(
                det.fd_response(frequency, h_sky, BNS_PARAMS),
                det_2g.fd_response(frequency, h_sky, BNS_PARAMS),
            )

    @pytest.mark.parametrize("name", ["ET", "CE"])
    def test_options_reach_every_detector(self, name):
        for det in self._detectors(name, earth_rotation=True, finite_arm_length=True):
            assert det.earth_rotation
            assert det.finite_arm_length
