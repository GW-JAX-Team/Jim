import logging
import time
from importlib import resources
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

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
    default_asd_files,
    get_CE,
    get_detector_preset,
    get_ET,
    get_ET_2L,
    get_H1,
    get_L1,
    get_V1,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.utils import complex_inner_product, inner_product
from jimgw.core.single_event.waveform import (
    RippleIMRPhenomD,
    RippleIMRPhenomHM,
    RippleIMRPhenomPv2,
    RippleIMRPhenomXHM,
    RippleIMRPhenomXPHM,
    RippleSineGaussian,
)
from tests.utils import (
    assert_all_in_range,
    sliced_fd_data,
    sliced_frequencies,
    sliced_psd,
)

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"

# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

GPS_TIME = 1126259462.0
DURATION = 4.0
F_MIN, F_MAX = 20.0, 1024.0
SAMPLING_FREQUENCY = F_MAX * 2


@pytest.mark.parametrize(
    "field",
    [
        "latitude",
        "longitude",
        "elevation",
        "xarm_azimuth",
        "yarm_azimuth",
        "xarm_tilt",
        "yarm_tilt",
    ],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_detector_geometry_must_be_finite(field, bad):
    with pytest.raises(ValueError, match=f"{field} must be a finite real scalar"):
        GroundBased2G("test", **{field: bad})


def test_static_detector_factory_remains_usable_inside_jit():
    def pattern(ra):
        return get_H1().antenna_pattern(ra, 0.2, 0.3, 0.4)

    expected = pattern(jnp.asarray(0.1))
    compiled = jax.jit(pattern)(jnp.asarray(0.1))
    for mode in expected:
        assert jnp.allclose(compiled[mode], expected[mode])


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

# Overrides of REFERENCE_PARAMS for a second, different signal.
SECOND_SIGNAL = {
    "M_c": 22.0,
    "eta": 0.2,
    "d_L": 300.0,
    "t_c": 0.2,
    "ra": 2.0,
    "psi": 1.0,
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
    det.set_zero_noise(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        trigger_time=trigger_time,
        f_min=F_MIN,
        f_max=F_MAX,
    )
    return det.inject_signal(RippleIMRPhenomD(f_ref=20.0), params)


def inject_noisy(det, rng_key):
    """Inject the reference signal plus noise drawn with *rng_key* into *det*."""
    det.simulate_and_set_noise(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        trigger_time=GPS_TIME,
        f_min=F_MIN,
        f_max=F_MAX,
        rng_key=rng_key,
    )
    return det.inject_signal(RippleIMRPhenomD(f_ref=20.0), dict(REFERENCE_PARAMS))


def injected_noise(getter, rng_key):
    """The noise that was injected: noisy data minus the zero-noise data."""
    clean = make_detector(getter)
    inject_reference(clean)
    noisy = make_detector(getter)
    inject_noisy(noisy, rng_key)
    return np.asarray(sliced_fd_data(noisy) - sliced_fd_data(clean)), noisy


# ----------------------------------------------------------------------
# inject_signal tests
# ----------------------------------------------------------------------


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

    def test_zero_noise_signal_nonzero_in_frequency_range(self):
        """Injected signal is non-zero inside the frequency range."""
        det = make_detector()
        inject_reference(det)

        assert jnp.any(jnp.abs(sliced_fd_data(det)) > 0)

    def test_zero_noise_frequency_bounds_respected(self):
        """Sliced frequencies lie within the requested frequency range."""
        det = make_detector()
        inject_reference(det)

        assert_all_in_range(sliced_frequencies(det), F_MIN, F_MAX)

    def test_noisy_injection_differs_from_zero_noise(self):
        """Adding noise produces data that differs from the zero-noise case."""
        det_clean = make_detector()
        inject_reference(det_clean)

        det_noisy = make_detector()
        inject_noisy(det_noisy, jax.random.key(42))

        assert not jnp.allclose(
            sliced_fd_data(det_clean),
            sliced_fd_data(det_noisy),
            rtol=1e-05,
            atol=1e-23,
        )


class TestSegmentCreators:
    """``set_zero_noise`` and ``simulate_and_set_noise`` create the data segment."""

    def test_set_zero_noise_creates_a_segment_of_zeros(self):
        det = make_detector()
        det.set_zero_noise(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            f_min=F_MIN,
            f_max=F_MAX,
        )

        assert len(det.data.td) == int(DURATION * SAMPLING_FREQUENCY)
        assert det.data.start_time == GPS_TIME - DURATION + 2.0
        assert det.data.frequency_bounds == (F_MIN, F_MAX)
        assert not jnp.any(det.data.td)

    def test_the_segment_ends_post_trigger_duration_after_the_trigger(self):
        det = make_detector()
        det.set_zero_noise(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            f_min=F_MIN,
            f_max=F_MAX,
            post_trigger_duration=1.0,
        )
        assert det.data.start_time + DURATION == GPS_TIME + 1.0

    def test_segments_shorter_than_the_default_taper_can_be_created(self):
        # The default taper needs 0.8 s; the zero and simulated data carry no taper.
        zero, noisy = make_detector(), make_detector()
        short = {
            "duration": 0.5,
            "sampling_frequency": SAMPLING_FREQUENCY,
            "trigger_time": GPS_TIME,
            "f_min": F_MIN,
            "f_max": F_MAX,
            "post_trigger_duration": 0.25,
        }
        zero.set_zero_noise(**short)
        noisy.simulate_and_set_noise(**short, rng_key=jax.random.key(0))
        assert zero.data.duration == noisy.data.duration == 0.5

    def test_the_creators_store_the_trigger_time_in_the_data(self):
        zero, noisy = make_detector(), make_detector()
        zero.set_zero_noise(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            f_min=F_MIN,
            f_max=F_MAX,
        )
        noisy.simulate_and_set_noise(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            f_min=F_MIN,
            f_max=F_MAX,
            rng_key=jax.random.key(0),
        )
        assert zero.data.trigger_time == noisy.data.trigger_time == GPS_TIME

    def test_injecting_keeps_the_trigger_time_in_the_data(self):
        det = make_detector()
        inject_reference(det)
        assert det.data.trigger_time == GPS_TIME

    def test_the_trigger_time_is_not_the_default_when_the_segment_ends_elsewhere(
        self,
    ):
        det = make_detector()
        det.set_zero_noise(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            f_min=F_MIN,
            f_max=F_MAX,
            post_trigger_duration=1.0,  # the default would be 2 s before the end
        )
        det.inject_signal(RippleIMRPhenomD(f_ref=20.0), dict(REFERENCE_PARAMS))
        assert det.data.trigger_time == GPS_TIME

    def test_noise_is_cut_to_the_frequency_range(self):
        det = make_detector()
        det.simulate_and_set_noise(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            f_min=F_MIN,
            f_max=F_MAX,
            rng_key=jax.random.key(0),
        )

        fd = np.asarray(det.data.fd)
        inside = np.asarray(det.data.frequency_mask)
        assert np.all(fd[~inside] == 0)
        assert np.all(fd[inside] != 0)
        assert det.data.frequency_bounds == (F_MIN, F_MAX)


class TestSeveralSignals:
    """Each call to ``inject_signal`` adds one signal; the noise is drawn once."""

    KEY = jax.random.key(7)

    @staticmethod
    def inject_second(det):
        return det.inject_signal(
            RippleIMRPhenomD(f_ref=20.0), {**REFERENCE_PARAMS, **SECOND_SIGNAL}
        )

    def test_signals_add_up(self):
        both = make_detector()
        inject_reference(both)
        self.inject_second(both)
        first, second = make_detector(), make_detector()
        inject_reference(first)
        inject_reference(second, **SECOND_SIGNAL)

        np.testing.assert_array_equal(
            sliced_fd_data(both), sliced_fd_data(first) + sliced_fd_data(second)
        )

    def test_the_noise_is_drawn_once(self):
        noise = make_detector()
        noise.simulate_and_set_noise(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            f_min=F_MIN,
            f_max=F_MAX,
            rng_key=self.KEY,
        )
        noise_fd = np.asarray(sliced_fd_data(noise))
        noisy = make_detector()
        inject_noisy(noisy, self.KEY)
        self.inject_second(noisy)
        signals = make_detector()
        inject_reference(signals)
        self.inject_second(signals)

        np.testing.assert_allclose(
            np.asarray(sliced_fd_data(noisy)) - noise_fd,
            np.asarray(sliced_fd_data(signals)),
            rtol=1e-9,
            atol=1e-12 * np.max(np.abs(noise_fd)),
        )

    def test_inject_signal_returns_the_snrs(self):
        det = make_detector()
        optimal_snr, match_filtered_snr = inject_reference(det)

        # Without noise the data are the signal, so both SNRs are its norm.
        data, psd = sliced_fd_data(det), sliced_psd(det)
        norm = float(inner_product(data, data, psd, 1 / DURATION) ** 0.5)
        assert float(optimal_snr) == pytest.approx(norm)
        assert complex(match_filtered_snr) == pytest.approx(norm)

    def test_the_matched_filter_snr_is_measured_against_all_the_data(self):
        first, second = make_detector(), make_detector()
        inject_reference(first)
        optimal_second, _ = inject_reference(second, **SECOND_SIGNAL)
        det = make_detector()
        inject_reference(det)
        optimal_snr, match_filtered_snr = self.inject_second(det)

        # (second | first + second) / |second|
        overlap = complex_inner_product(
            sliced_fd_data(second), sliced_fd_data(first), sliced_psd(det), 1 / DURATION
        )
        expected = (overlap + optimal_second**2) / optimal_second
        assert float(optimal_snr) == pytest.approx(float(optimal_second))
        assert complex(match_filtered_snr) == pytest.approx(complex(expected))


# ----------------------------------------------------------------------
# Injected noise: a function of (rng_key, detector name) and nothing else
# ----------------------------------------------------------------------


class TestInjectedNoise:
    KEY = jax.random.key(42)

    def test_same_key_gives_identical_data(self):
        a, b = make_detector(), make_detector()
        inject_noisy(a, self.KEY)
        inject_noisy(b, self.KEY)
        np.testing.assert_array_equal(sliced_fd_data(a), sliced_fd_data(b))

    def test_different_keys_give_different_noise(self):
        a, b = make_detector(), make_detector()
        inject_noisy(a, jax.random.key(1))
        inject_noisy(b, jax.random.key(2))
        assert np.max(np.abs(np.asarray(sliced_fd_data(a) - sliced_fd_data(b)))) > 0

    def test_data_does_not_depend_on_the_clock(self, monkeypatch):
        a, b = make_detector(), make_detector()
        monkeypatch.setattr(time, "time", lambda: 1.0e9)
        inject_noisy(a, self.KEY)
        monkeypatch.setattr(time, "time", lambda: 2.0e9)
        inject_noisy(b, self.KEY)
        np.testing.assert_array_equal(sliced_fd_data(a), sliced_fd_data(b))

    def test_one_key_gives_every_detector_its_own_noise(self):
        # All four detectors carry the same PSD here, so identical random draws would
        # give identical noise: the detector name is what tells them apart.  V1 and
        # CE have name tags above 2**31.  The noise is whitened first: the raw noise
        # is dominated by a few low-frequency bins, which makes its correlation noisy.
        whitened = {}
        for getter in (get_H1, get_L1, get_V1, get_CE):
            noise, det = injected_noise(getter, self.KEY)
            whitened[getter.__name__] = noise / np.sqrt(np.asarray(sliced_psd(det)))
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
                data[getter.__name__] = np.asarray(sliced_fd_data(det))
            return data

        forward = inject_in_order([get_H1, get_L1])
        backward = inject_in_order([get_L1, get_H1])
        for name, data in forward.items():
            np.testing.assert_array_equal(data, backward[name])

    def test_injected_noise_matches_the_psd_the_likelihood_reads(self):
        # Whitened by the PSD the likelihood uses (sliced_psd), the injected noise has
        # unit variance in its real and imaginary parts: noise and analysis share a PSD.
        noise, det = injected_noise(get_H1, self.KEY)
        whitened = noise / np.sqrt(np.asarray(sliced_psd(det)) * DURATION / 4)
        assert np.var(whitened.real) == pytest.approx(1.0, abs=0.1)
        assert np.var(whitened.imag) == pytest.approx(1.0, abs=0.1)


class TestWhitenedTimeDomainData:
    @pytest.mark.parametrize(("duration", "rate"), [(4.0, 2048.0), (3.0, 2047.0)])
    def test_every_sample_is_kept_and_the_noise_has_unit_variance(self, duration, rate):
        det = make_detector()
        kwargs = {
            **CREATOR_KWARGS,
            "duration": duration,
            "sampling_frequency": rate,
            "f_max": 1000.0,
        }
        det.simulate_and_set_noise(**kwargs, rng_key=jax.random.key(0))
        whitened = det.whitened_time_domain_data
        assert whitened.shape == (det.data.n_time,)
        assert float(jnp.var(whitened)) == pytest.approx(1.0, abs=0.1)


class TestInjectionRequirements:
    """What the creators and inject_signal need."""

    @staticmethod
    def inject(det):
        return det.inject_signal(RippleIMRPhenomD(f_ref=20.0), dict(REFERENCE_PARAMS))

    def test_noise_needs_a_psd(self):
        det = get_H1()  # no PSD set
        with pytest.raises(ValueError, match="No PSD is set on detector H1"):
            det.simulate_and_set_noise(
                duration=DURATION,
                sampling_frequency=SAMPLING_FREQUENCY,
                trigger_time=GPS_TIME,
                f_min=F_MIN,
                f_max=F_MAX,
                rng_key=jax.random.key(0),
            )
        # The failed call must leave the detector as it was.
        assert det.data.is_empty
        assert det.data.frequency_bounds == (0.0, jnp.inf)

    def test_inject_signal_needs_a_psd(self):
        det = get_H1()  # no PSD set
        det.set_zero_noise(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            f_min=F_MIN,
            f_max=F_MAX,
        )
        data = det.data
        with pytest.raises(ValueError, match="No PSD is set on detector H1"):
            self.inject(det)
        assert det.data is data

    def test_inject_signal_needs_data(self):
        det = make_detector()  # PSD, but no data
        with pytest.raises(ValueError, match="No data is set on detector H1"):
            self.inject(det)
        assert det.data.is_empty

    def test_real_data_inject_at_the_default_trigger_time(self):
        det = make_detector()
        det.set_data(real_h1_data())  # data that no creator made
        det.data.frequency_bounds = (F_MIN, F_MAX)
        optimal_snr, _ = self.inject(det)
        assert float(optimal_snr) > 0
        # 2 s before the end of the data
        assert det.data.trigger_time == det.data.start_time + det.data.duration - 2.0

    def test_inject_signal_needs_the_trigger_time_inside_the_data(self):
        det = make_detector()
        data = real_h1_data()
        data.trigger_time = GPS_TIME + 1000.0
        det.set_data(data)
        det.data.frequency_bounds = (F_MIN, F_MAX)
        with pytest.raises(ValueError, match="outside the data of detector 'H1'"):
            self.inject(det)

    def test_new_data_does_not_keep_the_old_trigger_time(self):
        det = make_detector()
        det.set_zero_noise(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME + 1000.0,
            f_min=F_MIN,
            f_max=F_MAX,
        )
        det.set_data(real_h1_data())
        det.data.frequency_bounds = (F_MIN, F_MAX)
        data = det.data
        assert data.trigger_time == data.start_time + data.duration - 2.0
        assert data.trigger_time != GPS_TIME + 1000.0

    def test_real_data_takes_a_trigger_time_set_by_hand(self):
        det = make_detector()
        data = real_h1_data()
        data.trigger_time = GPS_TIME
        det.set_data(data)
        det.data.frequency_bounds = (F_MIN, F_MAX)
        optimal_snr, _ = self.inject(det)
        assert float(optimal_snr) > 0


# ----------------------------------------------------------------------
# Slices of the data and the PSD
# ----------------------------------------------------------------------


def real_h1_data():
    return Data.from_file(str(FIXTURES_DIR / "GW150914_strain_H1.npz"))


class TestSlices:
    """The detector slices its data and PSD to a frequency range."""

    def test_reading_the_slices_fixes_the_data(self):
        real = real_h1_data()
        det = make_detector()
        det.set_data(real)
        det.data.frequency_bounds = (F_MIN, F_MAX)
        assert len(sliced_fd_data(det)) > 0
        assert real.has_fd
        with pytest.raises(RuntimeError, match="frequency-domain data have been fixed"):
            real.set_tukey_window(alpha=0.2)

    def test_whitened_data_is_not_zero_when_nothing_has_transformed_the_data(self):
        # Setting the frequency range does not Fourier transform the data as a side
        # effect, so the whitened data must do it itself instead of reading an
        # all-zero array.
        det = make_detector()
        det.set_data(real_h1_data())
        det.data.frequency_bounds = (F_MIN, F_MAX)
        assert jnp.any(jnp.abs(det.whitened_frequency_domain_data) > 0)

    def test_slices_follow_the_current_data(self):
        det = make_detector()
        inject_reference(det)  # 4 s at 2048 Hz, frequency range 20-1024 Hz
        assert len(sliced_frequencies(det)) == 4017
        # A longer segment has a finer frequency grid: the slice must follow it at once.
        longer = Data.from_td(
            td=jnp.zeros(16384), delta_t=1 / 2048.0, start_time=GPS_TIME
        )
        longer.frequency_bounds = (F_MIN, F_MAX)
        det.set_data(longer)
        assert len(sliced_frequencies(det)) == 8033  # (1024 - 20) / 0.125 + 1
        assert len(sliced_fd_data(det)) == len(sliced_psd(det)) == 8033

    def test_slices_follow_a_new_psd(self):
        det = make_detector()
        inject_reference(det)
        before = np.asarray(sliced_psd(det))
        det.set_psd(PowerSpectrum(det.psd.values * 4.0, det.psd.frequencies))
        np.testing.assert_allclose(np.asarray(sliced_psd(det)), 4.0 * before)

    def test_slice_to_frequency_range_returns_the_range_and_leaves_the_detector_alone(
        self,
    ):
        det = make_detector()
        inject_reference(det)
        frequencies, fd_data, psd = det.slice_to_frequency_range(30.0, 512.0)
        assert float(frequencies[0]) == 30.0
        assert float(frequencies[-1]) == 512.0
        assert fd_data.shape == frequencies.shape == psd.shape
        assert det.data.frequency_bounds == (F_MIN, F_MAX)

    @pytest.mark.parametrize(("f_min", "f_max"), [(10.0, 512.0), (30.0, 2000.0)])
    def test_slice_to_frequency_range_refuses_a_range_wider_than_the_data(
        self, f_min, f_max
    ):
        det = make_detector()
        inject_reference(det)
        with pytest.raises(
            ValueError, match="wider than the frequency range of the data"
        ):
            det.slice_to_frequency_range(f_min, f_max)

    def test_slice_to_frequency_range_needs_data_and_a_psd(self):
        with pytest.raises(ValueError, match="needs both data and a PSD"):
            get_H1().slice_to_frequency_range(20.0, 512.0)

    def test_slice_rejects_psd_whose_reciprocal_overflows(self):
        det = zero_noise_detector()
        frequencies = det.psd.frequencies
        index = int(jnp.argmin(jnp.abs(frequencies - 100.0)))
        values = (
            jnp.asarray(det.psd.values).at[index].set(jnp.finfo(jnp.float64).tiny / 16)
        )
        det.set_psd(PowerSpectrum(values, frequencies))
        with pytest.raises(ValueError, match="finite reciprocal"):
            det.slice_to_frequency_range(30.0, 512.0)

    def test_psd_reciprocal_overflow_outside_the_frequency_range_remains_allowed(self):
        det = zero_noise_detector()
        frequencies = det.psd.frequencies
        index = int(jnp.argmin(jnp.abs(frequencies - 10.0)))
        values = (
            jnp.asarray(det.psd.values).at[index].set(jnp.finfo(jnp.float64).tiny / 16)
        )
        det.set_psd(PowerSpectrum(values, frequencies))
        _, fd_data, psd = det.slice_to_frequency_range(30.0, 512.0)
        assert jnp.all(jnp.isfinite(fd_data))
        assert jnp.all(jnp.isfinite(1 / psd))


# ----------------------------------------------------------------------
# Weird usage: every bad call raises before anything changes
# ----------------------------------------------------------------------

DETECTOR_LOGGER = "jimgw.core.single_event.detector"
CREATOR_KWARGS = {
    "duration": DURATION,
    "sampling_frequency": SAMPLING_FREQUENCY,
    "trigger_time": GPS_TIME,
    "f_min": F_MIN,
    "f_max": F_MAX,
}
CREATORS = ["set_zero_noise", "simulate_and_set_noise"]
WAVEFORM = RippleIMRPhenomD(f_ref=20.0)


def state_of(det):
    """What a failing call must leave alone."""
    return det.data, det.psd, det.data.frequency_bounds, det.data.has_fd


def assert_untouched(det, state):
    data, psd, bounds, has_fd = state
    assert det.data is data
    assert det.psd is psd
    assert det.data.frequency_bounds == bounds
    assert det.data.has_fd == has_fd  # the window can only change while this is False


def create(det, creator, **overrides):
    kwargs = {**CREATOR_KWARGS, **overrides}
    if creator == "simulate_and_set_noise":
        kwargs.setdefault("rng_key", jax.random.key(0))
    return getattr(det, creator)(**kwargs)


def zero_noise_detector():
    det = make_detector()
    det.set_zero_noise(**CREATOR_KWARGS)
    return det


def holds_a_signal(det):
    """Zero-noise data are nonzero somewhere exactly when a signal was injected."""
    return bool(jnp.any(det.data.fd != 0))


@pytest.mark.parametrize("creator", CREATORS)
class TestCreatorArguments:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"duration": 0.0}, "must be positive"),
            ({"duration": -4.0}, "must be positive"),
            ({"duration": jnp.nan}, "must be positive"),
            ({"duration": jnp.inf}, "must be positive"),
            ({"sampling_frequency": 0.0}, "must be positive"),
            ({"sampling_frequency": jnp.nan}, "must be positive"),
            ({"sampling_frequency": jnp.inf}, "must be positive"),
            ({"trigger_time": jnp.nan}, "trigger_time must be finite"),
            ({"trigger_time": jnp.inf}, "trigger_time must be finite"),
            ({"post_trigger_duration": -1.0}, "post_trigger_duration must be between"),
            ({"post_trigger_duration": DURATION + 1}, "post_trigger_duration must be"),
            ({"post_trigger_duration": jnp.nan}, "post_trigger_duration must be"),
            ({"f_min": -5.0}, "must not be negative"),
            ({"f_min": jnp.nan}, "must not be NaN"),
            ({"f_min": 500.0, "f_max": 400.0}, "below f_max"),
            ({"f_max": 5000.0}, "Nyquist"),
            ({"f_min": 20.0, "f_max": 20.1}, "at least two"),
            (
                {
                    "duration": 1.0,
                    "sampling_frequency": 1.0,
                    "post_trigger_duration": 0.5,
                },
                "at least two time samples",
            ),
        ],
    )
    def test_a_bad_argument_raises_and_changes_nothing(self, creator, overrides, match):
        det = zero_noise_detector()
        before = state_of(det)
        with pytest.raises(ValueError, match=match):
            create(det, creator, **overrides)
        assert_untouched(det, before)

    def test_the_default_post_trigger_duration_must_fit_a_short_segment(self, creator):
        det = zero_noise_detector()
        before = state_of(det)
        with pytest.raises(ValueError, match="post_trigger_duration must be between"):
            create(det, creator, duration=1.0)  # the default 2 s is longer
        assert_untouched(det, before)

    def test_an_odd_number_of_samples_works(self, creator):
        det = make_detector()
        create(
            det,
            creator,
            duration=3.0,
            sampling_frequency=2047.0,
            f_max=1000.0,
            post_trigger_duration=1.0,
        )
        assert det.data.n_time == 6141
        det.inject_signal(WAVEFORM, dict(REFERENCE_PARAMS))

    def test_replaces_whatever_the_detector_held(self, creator):
        det = make_detector()
        inject_reference(det)
        first = det.data
        create(det, creator, trigger_time=GPS_TIME + 10.0)
        assert det.data is not first
        assert det.data.trigger_time == GPS_TIME + 10.0

    @pytest.mark.parametrize("duration", [4.0001, 4.0004])
    @pytest.mark.parametrize("post", [0.0, 2.0])
    def test_fractional_sample_durations_use_the_actual_buffer_length(
        self, creator, duration, post
    ):
        det = make_detector()
        create(det, creator, duration=duration, post_trigger_duration=post)
        assert det.data.start_time + float(det.data.duration) == GPS_TIME + post
        assert (
            det.data.start_time <= GPS_TIME <= det.data.start_time + det.data.duration
        )

    def test_rounding_must_not_put_the_trigger_before_the_segment(self, creator):
        det = zero_noise_detector()
        before = state_of(det)
        with pytest.raises(ValueError, match="rounded segment duration"):
            create(det, creator, duration=4.0001, post_trigger_duration=4.0001)
        assert_untouched(det, before)


class TestNoiseCreatorInputs:
    @pytest.mark.parametrize("bad", [0.0, -1.0, jnp.nan, jnp.inf])
    def test_invalid_psd_outside_the_frequency_range_of_the_data_does_not_change_noise(
        self, bad
    ):
        frequencies = jnp.arange(9.0) / 4
        clean, outside_invalid = get_H1(), get_H1()
        clean.set_psd(PowerSpectrum(jnp.ones(9), frequencies))
        outside_invalid.set_psd(PowerSpectrum(jnp.ones(9).at[1].set(bad), frequencies))
        for detector in (clean, outside_invalid):
            detector.simulate_and_set_noise(
                duration=4.0,
                sampling_frequency=4.0,
                trigger_time=100.0,
                f_min=0.5,
                f_max=1.5,
                rng_key=jax.random.key(0),
            )

        np.testing.assert_array_equal(outside_invalid.data.fd, clean.data.fd)
        assert bool(jnp.all(jnp.isfinite(outside_invalid.data.fd)))
        np.testing.assert_array_equal(
            np.asarray(outside_invalid.psd.values[1]), np.asarray(bad)
        )

    def test_a_wrong_rng_key_changes_nothing(self):
        det = zero_noise_detector()
        before = state_of(det)
        with pytest.raises(TypeError):
            det.simulate_and_set_noise(**CREATOR_KWARGS, rng_key=0)
        assert_untouched(det, before)

    @pytest.mark.parametrize("key", [jax.random.key(0), jax.random.PRNGKey(0)])
    def test_both_kinds_of_jax_key_work(self, key):
        det = make_detector()
        det.simulate_and_set_noise(**CREATOR_KWARGS, rng_key=key)
        assert bool(jnp.any(det.data.fd != 0))  # noise was drawn

    @pytest.mark.parametrize("bad", [0.0, -1.0, jnp.nan, jnp.inf])
    def test_the_psd_must_be_finite_and_positive_in_the_range(self, bad):
        det = make_detector()
        index = int(jnp.argmin(jnp.abs(det.psd.frequencies - 100.0)))
        det.set_psd(
            PowerSpectrum(det.psd.values.at[index].set(bad), det.psd.frequencies)
        )
        before = state_of(det)
        with pytest.raises(ValueError, match="not finite and positive"):
            det.simulate_and_set_noise(**CREATOR_KWARGS, rng_key=jax.random.key(0))
        assert_untouched(det, before)


class TestLoadAndSetPsd:
    def test_psd_file_and_asd_file_are_mutually_exclusive(self):
        det = get_H1()
        path = str(FIXTURES_DIR / "GW150914_psd_H1.npz")
        with pytest.raises(ValueError, match="mutually exclusive"):
            det.load_and_set_psd(psd_file=path, asd_file=path)
        assert det.psd.is_empty

    @staticmethod
    def preset_interferometers():
        return {
            ifo.name: ifo
            for value in (*get_detector_preset().values(), get_ET_2L())
            for ifo in (value if isinstance(value, list) else [value])
        }

    def test_every_preset_interferometer_has_a_default(self):
        assert set(default_asd_files) == set(self.preset_interferometers())

    @pytest.mark.parametrize("name", sorted(default_asd_files))
    def test_default_is_the_packaged_design_asd(self, name):
        psd = self.preset_interferometers()[name].load_and_set_psd()

        curve = resources.files("jimgw.core.single_event") / "noise_curves"
        frequencies, asd = np.genfromtxt(
            str(curve / default_asd_files[name]), unpack=True
        )
        np.testing.assert_array_equal(np.asarray(psd.frequencies), frequencies)
        np.testing.assert_array_equal(np.asarray(psd.values), asd**2)
        in_range = (frequencies >= 20.0) & (frequencies <= 1024.0)
        assert np.all(np.isfinite(asd[in_range]) & (asd[in_range] > 0.0))

    def test_detector_without_a_default_asks_for_a_file(self):
        det = get_H1()
        det.name = "X1"
        with pytest.raises(ValueError, match="No built-in PSD for detector X1"):
            det.load_and_set_psd()
        assert det.psd.is_empty


class TestInjectionArguments:
    def inject(self, det, parameters=REFERENCE_PARAMS, **kwargs):
        return det.inject_signal(WAVEFORM, parameters, **kwargs)

    def assert_refused(self, det, error, match, **kwargs):
        before = state_of(det)
        with pytest.raises(error, match=match):
            self.inject(det, **kwargs)
        assert_untouched(det, before)

    @pytest.mark.parametrize("key", ["ra", "dec", "psi", "t_c", "M_c", "eta", "d_L"])
    def test_a_missing_parameter_is_named(self, key):
        parameters = {k: v for k, v in REFERENCE_PARAMS.items() if k != key}
        self.assert_refused(
            zero_noise_detector(),
            ValueError,
            rf"missing \['{key}'\]",
            parameters=parameters,
        )

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("d_L", jnp.nan),
            ("d_L", 0.0),
            ("M_c", jnp.inf),
            ("M_c", -5.0),
            ("ra", jnp.nan),
            ("t_c", jnp.inf),
        ],
    )
    def test_a_parameter_that_gives_no_finite_signal_is_refused(self, key, value):
        self.assert_refused(
            zero_noise_detector(),
            ValueError,
            "not finite at",
            parameters={**REFERENCE_PARAMS, key: value},
        )

    @pytest.mark.parametrize("eta", [0.3, 0.8, 0.0, -0.1, jnp.nan])
    def test_eta_must_not_exceed_a_quarter(self, eta):
        # The waveform stays finite for an eta above 1/4, e.g. a mass ratio given as eta.
        self.assert_refused(
            zero_noise_detector(),
            ValueError,
            "eta must be in",
            parameters={**REFERENCE_PARAMS, "eta": eta},
        )

    @pytest.mark.parametrize("bad", [0.0, -1.0, jnp.nan, jnp.inf])
    def test_the_psd_must_be_finite_and_positive_in_the_range(self, bad):
        det = zero_noise_detector()
        index = int(jnp.argmin(jnp.abs(det.psd.frequencies - 100.0)))
        values = jnp.asarray(det.psd.values).at[index].set(bad)
        det.set_psd(PowerSpectrum(values, det.psd.frequencies))
        self.assert_refused(det, ValueError, "not finite and positive")

    def test_equal_masses_are_accepted(self):
        det = zero_noise_detector()
        self.inject(det, {**REFERENCE_PARAMS, "eta": 0.25 + 1e-12})
        assert holds_a_signal(det)

    def test_extra_parameters_are_allowed(self):
        det = zero_noise_detector()
        self.inject(det, {**REFERENCE_PARAMS, "something_else": "ignored"})
        assert holds_a_signal(det)

    def test_real_data_need_a_frequency_range_first(self):
        det = make_detector()
        det.set_data(real_h1_data())
        before = state_of(det)
        with pytest.raises(ValueError, match="data.frequency_bounds"):
            self.inject(det)
        assert_untouched(det, before)
        det.data.frequency_bounds = (F_MIN, F_MAX)
        data = det.data
        self.inject(det)
        assert det.data is not data

    def test_a_range_with_fewer_than_two_frequencies_is_refused(self):
        det = zero_noise_detector()
        with pytest.raises(ValueError, match="at least two"):
            det.slice_to_frequency_range(F_MIN, F_MIN + 0.1)

    def test_a_signal_that_is_not_finite_is_refused(self):
        class NotANumber:
            parameter_names = ("M_c",)

            def __call__(self, frequency, params):
                nan = jnp.full(frequency.shape, jnp.nan, dtype=complex)
                return {"p": nan, "c": nan}

        det = zero_noise_detector()
        before = state_of(det)
        parameters = {
            k: REFERENCE_PARAMS[k] for k in ("M_c", "ra", "dec", "psi", "t_c")
        }
        with pytest.raises(ValueError, match="not finite at"):
            det.inject_signal(NotANumber(), parameters)
        assert_untouched(det, before)

    def test_zero_amplitude_is_refused_without_freezing_the_original_window(self):
        class ZeroSignal:
            parameter_names = ()

            def __call__(self, frequency, params):
                zero = jnp.zeros(frequency.shape, dtype=complex)
                return {"p": zero, "c": zero}

        det = zero_noise_detector()
        before = state_of(det)
        parameters = {k: REFERENCE_PARAMS[k] for k in ("ra", "dec", "psi", "t_c")}
        with pytest.raises(ValueError, match="positive signal-to-noise power"):
            det.inject_signal(ZeroSignal(), parameters)
        assert_untouched(det, before)
        det.data.set_tukey_window(alpha=0.1)

    @pytest.mark.parametrize("failure", ["signal", "psd", "grid", "containment"])
    def test_failed_injection_keeps_real_data_uncached_and_its_window_editable(
        self, failure
    ):
        det = make_detector()
        det.set_data(real_h1_data())
        det.data.frequency_bounds = (F_MIN, F_MAX)
        parameters = dict(REFERENCE_PARAMS)
        if failure == "signal":
            parameters["d_L"] = jnp.nan
        elif failure == "psd":
            values = jnp.asarray(det.psd.values).at[400].set(jnp.nan)
            det.set_psd(PowerSpectrum(values, det.psd.frequencies))
        elif failure == "grid":
            det.psd = PowerSpectrum(det.psd.values, det.psd.frequencies + 0.01)
        else:
            parameters["t_c"] = 3.0
        before = state_of(det)
        with pytest.raises(ValueError):
            self.inject(det, parameters)
        assert_untouched(det, before)
        det.data.set_tukey_window(alpha=0.1)

    def test_even_nyquist_injection_and_reported_snr_use_the_same_real_signal(self):
        class HighFrequencyPacket:
            parameter_names = ()

            def __call__(self, frequency, params):
                hp = jnp.exp(-0.5 * ((frequency - 900.0) / 50.0) ** 2) * (1 + 1j)
                return {"p": hp, "c": jnp.zeros_like(hp)}

        det = zero_noise_detector()
        parameters = {k: REFERENCE_PARAMS[k] for k in ("ra", "dec", "psi", "t_c")}
        rho, matched = det.inject_signal(
            HighFrequencyPacket(), parameters, allow_wraparound=True
        )
        signal = sliced_fd_data(det)
        norm = inner_product(signal, signal, sliced_psd(det), 1 / DURATION) ** 0.5
        assert float(rho) == pytest.approx(float(norm))
        assert complex(matched) == pytest.approx(float(norm))
        assert det.data.fd[-1].imag == 0
        np.testing.assert_allclose(det.data.fft(), det.data.fd, rtol=1e-12, atol=1e-13)

    def test_the_range_survives_an_injection(self):
        det = zero_noise_detector()
        self.inject(det)
        assert det.data.frequency_bounds == (F_MIN, F_MAX)

    def test_user_data_with_an_odd_number_of_samples_keep_their_grid(self):
        det = make_detector()
        data = Data.from_td(
            td=jnp.zeros(8193), delta_t=1 / 2048.0, start_time=GPS_TIME - 2.0
        )
        data.frequency_bounds = (F_MIN, F_MAX)
        det.set_data(data)
        psd = det.psd
        self.inject(det)
        assert det.data.n_time == 8193
        assert det.data.delta_t == 1 / 2048.0
        assert jnp.array_equal(det.data.frequencies, data.frequencies)
        assert det.psd is psd

    def test_injecting_into_real_data_adds_the_signal_to_its_windowed_fft(self):
        windowed = real_h1_data().fd  # an identical copy gives the same windowed FFT
        det = make_detector()
        det.set_data(real_h1_data())
        det.data.frequency_bounds = (F_MIN, F_MAX)
        self.inject(det)
        grid = det.data.frequencies
        projected = det.fd_response(
            grid,
            WAVEFORM(grid, dict(REFERENCE_PARAMS)),
            dict(REFERENCE_PARAMS),
        )
        expected = windowed + jnp.where(det.data.frequency_mask, projected, 0.0)
        np.testing.assert_array_equal(det.data.fd, expected)

    def test_the_order_of_the_signals_does_not_matter(self):
        ab, ba = zero_noise_detector(), zero_noise_detector()
        second = {**REFERENCE_PARAMS, **SECOND_SIGNAL}
        self.inject(ab)
        self.inject(ab, second)
        self.inject(ba, second)
        self.inject(ba)
        np.testing.assert_array_equal(ab.data.fd, ba.data.fd)

    def test_the_total_snr_squared_of_two_signals(self):
        # (A+B | A+B) = rho_A^2 + rho_B^2 + 2 Re (A|B)
        second = {**REFERENCE_PARAMS, **SECOND_SIGNAL}
        both, only_a, only_b = (zero_noise_detector() for _ in range(3))
        rho_a, _ = self.inject(both)
        rho_b, _ = self.inject(both, second)
        self.inject(only_a)
        self.inject(only_b, second)
        a, b = sliced_fd_data(only_a), sliced_fd_data(only_b)
        psd, df = sliced_psd(both), 1 / DURATION
        total = inner_product(sliced_fd_data(both), sliced_fd_data(both), psd, df)
        overlap = complex_inner_product(a, b, psd, df).real
        assert float(total) == pytest.approx(
            float(rho_a) ** 2 + float(rho_b) ** 2 + 2 * float(overlap), rel=1e-9
        )


class TestSignalMustFitTheSegment:
    """A conservative duration estimate catches short or badly placed injections."""

    def inject(self, det, waveform=WAVEFORM, **updates):
        return det.inject_signal(waveform, {**REFERENCE_PARAMS, **updates})

    @pytest.mark.parametrize("t_c", [1.9, 3.0, 7.0, -0.7, -3.0, 64.0, -64.0])
    def test_a_signal_placed_outside_the_segment_is_refused(self, t_c):
        det = zero_noise_detector()
        before = state_of(det)
        with pytest.raises(ValueError, match="does not fit the segment"):
            self.inject(det, t_c=t_c)
        assert_untouched(det, before)

    # The reference signal fits for t_c between about -0.65 s and +1.60 s: the
    # chirp needs time before the peak, and only the ringdown comes after it.
    @pytest.mark.parametrize("t_c", [0.0, 0.03, -0.25, -0.55, 1.2])
    def test_a_signal_inside_the_segment_is_injected(self, t_c):
        det = zero_noise_detector()
        self.inject(det, t_c=t_c)
        assert holds_a_signal(det)

    @pytest.mark.parametrize(
        ("spins", "t_c", "fits"),
        [
            ((0.0, 0.0), -0.70, True),
            ((0.0, 0.0), -0.75, False),
            ((0.1, -0.3), -0.60, True),
            ((0.1, -0.3), -0.70, False),
            ((-1.0, 0.2), -0.40, True),
            ((-1.0, 0.2), -0.55, False),
        ],
    )
    def test_the_larger_spin_sets_the_time_needed_before_the_peak(
        self, spins, t_c, fits
    ):
        # The earliest allowed t_c is about -0.73 s for zero spin, -0.65 s for a
        # larger spin of 0.3 and -0.47 s for a spin of 1 (in either direction).
        det = zero_noise_detector()
        params = {**REFERENCE_PARAMS, "s1_z": spins[0], "s2_z": spins[1], "t_c": t_c}
        if fits:
            det._check_signal_duration(WAVEFORM, params, GPS_TIME, False)
        else:
            with pytest.raises(ValueError, match="does not fit the segment"):
                det._check_signal_duration(WAVEFORM, params, GPS_TIME, False)

    @pytest.mark.parametrize("spin", ["s1_x", "s1_z", "s2_y"])
    def test_precessing_waveforms_count_the_whole_spin_vector(self, spin):
        det = zero_noise_detector()
        waveform = RippleIMRPhenomPv2(f_ref=20.0)
        spins = {f"s{body}_{axis}": 0.0 for body in "12" for axis in "xyz"}
        params = {**REFERENCE_PARAMS, **spins, "t_c": -0.62}
        det._check_signal_duration(waveform, params, GPS_TIME, False)
        with pytest.raises(ValueError, match="does not fit the segment"):
            det._check_signal_duration(waveform, {**params, spin: 0.8}, GPS_TIME, False)

    def test_a_waveform_without_spin_parameters_assumes_the_maximal_spin(self):
        class NoSpin:
            parameter_names = ("M_c", "eta")

        det = zero_noise_detector()
        params = {"M_c": 28.0, "eta": 0.24, "t_c": -0.40}
        det._check_signal_duration(NoSpin(), params, GPS_TIME, False)
        with pytest.raises(ValueError, match="does not fit the segment"):
            det._check_signal_duration(
                NoSpin(), {**params, "t_c": -0.55}, GPS_TIME, False
            )

    @pytest.mark.parametrize(("highest_mode", "fits"), [(2, True), (4, False)])
    def test_the_highest_mode_of_the_waveform_sets_the_starting_frequency(
        self, highest_mode, fits
    ):
        tagged = SimpleNamespace(
            parameter_names=("M_c", "eta"),
            waveform_metadata={"highest_mode": highest_mode},
        )
        det = zero_noise_detector()
        params = {"M_c": 28.0, "eta": 0.24, "t_c": 0.0}
        if fits:
            det._check_signal_duration(tagged, params, GPS_TIME, False)
        else:
            with pytest.raises(ValueError, match="does not fit the segment"):
                det._check_signal_duration(tagged, params, GPS_TIME, False)

    @pytest.mark.parametrize(
        ("chirp_mass", "fits"),
        [(28.0, True), (25.0, True), (20.0, False), (7.0, False)],
    )
    def test_a_long_signal_needs_a_long_segment(self, chirp_mass, fits):
        det = zero_noise_detector()
        if fits:
            self.inject(det, M_c=chirp_mass)
            assert holds_a_signal(det)
        else:
            before = state_of(det)
            with pytest.raises(ValueError, match="does not fit the segment"):
                self.inject(det, M_c=chirp_mass)
            assert_untouched(det, before)

    def test_a_longer_segment_fits_a_long_signal(self):
        det = make_detector()
        det.set_zero_noise(
            **{**CREATOR_KWARGS, "duration": 16.0, "post_trigger_duration": 2.0}
        )
        self.inject(det, M_c=7.0)
        assert holds_a_signal(det)

    def test_lowering_the_frequency_cutoff_requires_more_time(self):
        det = zero_noise_detector()
        det.data.frequency_bounds = (10.0, F_MAX)
        before = state_of(det)
        with pytest.raises(ValueError, match="does not fit the segment"):
            self.inject(det)
        assert_untouched(det, before)
        det.set_zero_noise(**{**CREATOR_KWARGS, "duration": 16.0, "f_min": 10.0})
        self.inject(det)
        assert holds_a_signal(det)

    @pytest.mark.parametrize(
        "factory", [RippleIMRPhenomHM, RippleIMRPhenomXHM, RippleIMRPhenomXPHM]
    )
    def test_higher_modes_need_a_longer_segment(self, factory):
        det = zero_noise_detector()
        waveform = factory(f_ref=20.0)
        with pytest.raises(ValueError, match="does not fit the segment"):
            det._check_signal_duration(waveform, REFERENCE_PARAMS, GPS_TIME, False)
        det.set_zero_noise(**{**CREATOR_KWARGS, "duration": 16.0})
        det._check_signal_duration(waveform, REFERENCE_PARAMS, GPS_TIME, False)

    @pytest.mark.parametrize("width", [0.1, 1.0])
    def test_unknown_duration_needs_an_explicit_override(self, width, caplog):
        class Packet:
            parameter_names = ()

            def __call__(self, frequency, params):
                hp = jnp.exp(-0.5 * ((frequency - 100.0) / width) ** 2) * (1e-23 + 0j)
                return {"p": hp, "c": jnp.zeros_like(hp)}

        det = zero_noise_detector()
        before = state_of(det)
        # Incidental mass values must not turn a burst into a CBC duration estimate.
        parameters = dict(REFERENCE_PARAMS)
        with pytest.raises(ValueError, match="Cannot estimate the signal duration"):
            det.inject_signal(Packet(), parameters)
        assert_untouched(det, before)
        with caplog.at_level(logging.WARNING, logger=DETECTOR_LOGGER):
            det.inject_signal(Packet(), parameters, allow_wraparound=True)
        assert "Cannot estimate the signal duration" in caplog.text
        assert holds_a_signal(det)

    @pytest.mark.parametrize("allow_wraparound", [False, True])
    def test_time_domain_models_cannot_be_injected_as_frequency_domain(
        self, allow_wraparound
    ):
        det = zero_noise_detector()
        before = state_of(det)
        with pytest.raises(ValueError, match="requires a frequency-domain waveform"):
            det.inject_signal(
                RippleSineGaussian(), {}, allow_wraparound=allow_wraparound
            )
        assert_untouched(det, before)

    @pytest.mark.parametrize("bad", [0.0, -1.0, jnp.nan, jnp.inf])
    def test_invalid_psd_outside_the_frequency_range_of_the_data_does_not_contaminate_the_check(
        self, bad
    ):
        det = zero_noise_detector()
        values = jnp.asarray(det.psd.values).at[79].set(bad)  # 19.75 Hz
        det.set_psd(PowerSpectrum(values, det.psd.frequencies))
        self.inject(det)
        assert holds_a_signal(det)

    def test_allow_wraparound_injects_with_a_warning(self, caplog):
        det = zero_noise_detector()
        with caplog.at_level(logging.WARNING, logger=DETECTOR_LOGGER):
            det.inject_signal(
                WAVEFORM, {**REFERENCE_PARAMS, "t_c": 3.0}, allow_wraparound=True
            )
        assert "does not fit the segment" in caplog.text
        assert holds_a_signal(det)

    def test_a_precessing_waveform_at_ordinary_parameters_fits(self):
        det = zero_noise_detector()
        parameters = {**REFERENCE_PARAMS, "iota": 0.8}
        det.inject_signal(RippleIMRPhenomPv2(f_ref=20.0), parameters)
        assert holds_a_signal(det)

    def test_duration_check_does_not_evaluate_or_differentiate_the_waveform(
        self, caplog
    ):
        class Opaque:
            parameter_names = WAVEFORM.parameter_names
            calls = 0

            def __call__(self, frequency, params):
                self.calls += 1
                float(frequency[0])  # fails if the checker tries to trace it
                return WAVEFORM(frequency, params)

        waveform = Opaque()
        det = zero_noise_detector()
        with caplog.at_level(logging.WARNING, logger=DETECTOR_LOGGER):
            self.inject(det, waveform=waveform)
        assert not caplog.records
        assert waveform.calls == 1
        assert holds_a_signal(det)


class TestResponseGmst:
    """``fd_response`` reads its reference timing from the attached data."""

    def test_gmst_is_that_of_the_trigger_time(self):
        params = {"ra": 1.5, "dec": 0.5, "psi": 0.3, "t_c": 0.1}
        det = make_detector()
        det.set_data(
            Data.from_td(
                td=jnp.zeros(8192), delta_t=1 / 2048.0, start_time=GPS_TIME - 2.0
            )
        )
        gmst = compute_gmst(GPS_TIME)
        antenna = det.antenna_pattern(1.5, 0.5, 0.3, gmst)
        h_sky = {
            name: jnp.ones(len(det.data.frequencies), dtype=complex) for name in antenna
        }
        delay = det.delay_from_geocenter(1.5, 0.5, gmst)
        shift = delay + (GPS_TIME - det.data.start_time + params["t_c"])
        expected = sum(h_sky[name] * antenna[name] for name in antenna) * jnp.exp(
            -2j * jnp.pi * det.data.frequencies * shift
        )

        response = det.fd_response(det.data.frequencies, h_sky, params)
        np.testing.assert_allclose(response, expected)

    def test_mutable_response_tracks_the_current_trigger_and_start(self):
        det = zero_noise_detector()
        params = {"ra": 1.5, "dec": 0.5, "psi": 0.3, "t_c": 0.1}
        frequency = det.data.frequencies
        h_sky = {"p": jnp.ones_like(frequency), "c": 1j * jnp.ones_like(frequency)}
        original = det.fd_response(frequency, h_sky, params)

        det.data.start_time -= 0.125
        shifted = det.fd_response(frequency, h_sky, params)
        np.testing.assert_allclose(
            shifted, original * jnp.exp(-2j * jnp.pi * frequency * 0.125), atol=1e-12
        )

        det.data.trigger_time = GPS_TIME + 100.0
        gmst = compute_gmst(det.data.trigger_time)
        antenna = det.antenna_pattern(params["ra"], params["dec"], params["psi"], gmst)
        delay = det.delay_from_geocenter(params["ra"], params["dec"], gmst)
        shift = delay + (det.data.trigger_time - det.data.start_time + params["t_c"])
        expected = sum(h_sky[name] * antenna[name] for name in antenna) * jnp.exp(
            -2j * jnp.pi * frequency * shift
        )
        np.testing.assert_allclose(det.fd_response(frequency, h_sky, params), expected)


class TestDataAndPsdSetters:
    def test_clear_data_and_psd_empties_both(self):
        det = zero_noise_detector()
        det.clear_data_and_psd()
        assert det.data.is_empty and det.psd.is_empty

    def test_set_data_passes_keywords_to_from_td(self):
        det = make_detector()
        det.set_data(jnp.zeros(8), delta_t=0.5, start_time=10.0)
        assert det.data.delta_t == 0.5
        assert det.data.start_time == 10.0

    def test_set_psd_passes_keywords_to_the_constructor(self):
        det = zero_noise_detector()
        frequencies = det.data.frequencies
        det.set_psd(jnp.ones_like(frequencies), frequencies=frequencies, name="own")
        assert det.psd.name == "own"


class TestDetectorFreezing:
    """Likelihood-owned detectors and their shared inputs become immutable."""

    def test_freezing_is_in_place_and_idempotent(self):
        det = zero_noise_detector()
        data, psd = det.data, det.psd
        det._freeze()
        fd = data.fd
        tensor = det.tensor
        det._freeze()

        assert det.is_frozen
        assert det.data is data and det.psd is psd
        assert data.is_frozen and psd.is_frozen
        assert data.fd is fd
        np.testing.assert_array_equal(det.tensor, tensor)
        assert isinstance(det.polarization_mode, tuple)
        _, sliced, _ = det.slice_to_frequency_range(F_MIN, F_MAX)
        np.testing.assert_array_equal(sliced, fd[data.frequency_mask])

    @pytest.mark.parametrize(
        "attribute",
        [
            "name",
            "data",
            "psd",
            "latitude",
            "polarization_mode",
            "_frozen",
        ],
    )
    def test_frozen_detector_rejects_attribute_changes(self, attribute):
        det = zero_noise_detector()
        det._freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            setattr(det, attribute, getattr(det, attribute))
        with pytest.raises(RuntimeError, match="frozen"):
            delattr(det, attribute)

    def test_tensor_tracks_geometry_until_freezing(self):
        det = zero_noise_detector()
        original = det.tensor
        det.xarm_azimuth += 0.25
        updated = det.tensor
        assert not bool(jnp.allclose(updated, original))

        det._freeze()
        np.testing.assert_array_equal(det.tensor, updated)

    def test_geometry_and_polarization_aliases_cannot_change_a_frozen_response(self):
        det = zero_noise_detector()
        geometry = {}
        for name in (
            "latitude",
            "longitude",
            "elevation",
            "xarm_azimuth",
            "yarm_azimuth",
            "xarm_tilt",
            "yarm_tilt",
        ):
            geometry[name] = np.array(getattr(det, name))
            setattr(det, name, geometry[name])
        modes = det.polarization_mode
        det._freeze()
        frequency = det.data.frequencies
        h_sky = {"p": jnp.ones_like(frequency), "c": 1j * jnp.ones_like(frequency)}
        expected = det.fd_response(frequency, h_sky, REFERENCE_PARAMS)

        for value in geometry.values():
            value[...] = 0.0
        modes.clear()
        assert all(isinstance(getattr(det, name), float) for name in geometry)
        assert len(det.polarization_mode) == 2
        np.testing.assert_array_equal(
            det.fd_response(frequency, h_sky, REFERENCE_PARAMS), expected
        )

    def test_freezing_does_not_change_the_response(self):
        det = zero_noise_detector()
        frequency = det.data.frequencies
        h_sky = {"p": jnp.ones_like(frequency), "c": 1j * jnp.ones_like(frequency)}
        expected = det.fd_response(frequency, h_sky, REFERENCE_PARAMS)
        det._freeze()

        np.testing.assert_array_equal(
            det.fd_response(frequency, h_sky, REFERENCE_PARAMS), expected
        )
        compiled = jax.jit(lambda params: det.fd_response(frequency, h_sky, params))
        np.testing.assert_allclose(compiled(REFERENCE_PARAMS), expected, atol=1e-12)

    @pytest.mark.parametrize(
        "method",
        [
            "clear_data_and_psd",
            "set_data",
            "set_psd",
            "load_and_set_psd",
            "set_zero_noise",
            "simulate_and_set_noise",
            "inject_signal",
        ],
    )
    def test_mutators_reject_frozen_detectors_before_work(self, method, monkeypatch):
        det = zero_noise_detector()
        det._freeze()
        data, psd = det.data, det.psd
        calls = {
            "clear_data_and_psd": ((), {}),
            "set_data": ((jnp.zeros(8),), {"delta_t": 0.1}),
            "set_psd": ((jnp.ones(5),), {"delta_f": 1.0}),
            "load_and_set_psd": ((), {}),
            "set_zero_noise": ((), CREATOR_KWARGS),
            "simulate_and_set_noise": (
                (),
                {**CREATOR_KWARGS, "rng_key": jax.random.key(0)},
            ),
            "inject_signal": ((object(), {}), {}),
        }

        def unexpected(*args, **kwargs):
            raise AssertionError(
                "Frozen mutators must fail before constructing or loading inputs."
            )

        monkeypatch.setattr(Data, "__init__", unexpected)
        monkeypatch.setattr(Data, "from_td", unexpected)
        monkeypatch.setattr(PowerSpectrum, "__init__", unexpected)
        monkeypatch.setattr(PowerSpectrum, "from_file", unexpected)
        monkeypatch.setattr(type(det), "_zero_data", unexpected)
        args, kwargs = calls[method]
        with pytest.raises(RuntimeError, match="frozen"):
            getattr(det, method)(*args, **kwargs)
        assert det.data is data and det.psd is psd


# ----------------------------------------------------------------------
# ET geometry tests
# ----------------------------------------------------------------------


class TestET:
    """Tests for get_ET(): geometric consistency of the triangular ET configuration."""

    ET_ARM_LENGTH_M = 1e4  # 10 km

    def setup_method(self):
        self.ifos = get_ET()

    def test_returns_three_detectors(self):
        """get_ET returns exactly three GroundBased3G instances."""
        assert len(self.ifos) == 3
        assert all(isinstance(ifo, GroundBased3G) for ifo in self.ifos)

    def test_et1_is_at_the_emr_site_of_the_2l_design(self):
        """The triangle and the EMR detector of the 2L design share their site."""
        et1, emr = self.ifos[0], get_ET_2L()[1]
        assert et1.latitude == emr.latitude
        assert et1.longitude == emr.longitude
        assert et1.elevation == emr.elevation

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


# ----------------------------------------------------------------------
# Next-generation detectors (GroundBased3G)
# ----------------------------------------------------------------------


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
}

# The sidereal angle at the trigger time of the data, wrapped, so that shifting it
# by 2 pi is exact to rounding of order one.
BNS_GMST = float(compute_gmst(ROTATION_TRIGGER_TIME)) % (2 * np.pi)


def _start_at(det, start_time):
    """Give *det* a data segment beginning at *start_time* and triggered at ROTATION_TRIGGER_TIME."""
    det.set_data(
        Data.from_td(
            jnp.zeros(8),
            1.0,
            start_time=start_time,
            trigger_time=ROTATION_TRIGGER_TIME,
        )
    )
    return det


def _sidereal_angle_at_trigger(monkeypatch, gmst):
    """Make every detector see *gmst* as the sidereal angle at the trigger time."""
    monkeypatch.setattr(
        detector_module, "cached_greenwich_mean_sidereal_time", lambda _: gmst
    )


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
        tau = float(GroundBased3G.time_to_merger(jnp.array(5.0), 1.22, 0.25)) / 60
        assert 106.0 < tau < 108.0, f"tau(5 Hz) = {tau:.2f} min"

    def test_reduces_to_newtonian_at_low_frequency(self):
        """The PN series tends to the leading-order chirp time as f -> 0."""
        f = jnp.array(1e-4)
        tau = float(GroundBased3G.time_to_merger(f, 1.22, 0.2495, 0.3, 0.2))
        tau_n = _newtonian_time_to_merger(1e-4, 1.22)
        np.testing.assert_allclose(tau, tau_n, rtol=1e-4, atol=0.0)

    def test_first_post_newtonian_coefficient(self):
        """(tau / tau_N - 1) / x tends to the 1PN coefficient 4/3 (743/336 + 11 eta / 4)."""
        M_c, eta = 1.22, 0.2
        M = M_c / eta ** (3 / 5)
        # Low enough that the 1.5PN term, relatively O(x^1/2), stays below 1e-3.
        f = 1e-9
        x = (np.pi * M * MTSUN * f) ** (2 / 3)
        tau = float(GroundBased3G.time_to_merger(jnp.array(f), M_c, eta))
        measured = (tau / _newtonian_time_to_merger(f, M_c) - 1) / x
        expected = 4 / 3 * (743 / 336 + 11 / 4 * eta)
        np.testing.assert_allclose(measured, expected, rtol=1e-3, atol=0.0)

    def test_aligned_spins_lengthen_the_inspiral(self):
        """Spins aligned with the orbit slow the inspiral (orbital hang-up)."""
        f = jnp.array(10.0)
        aligned = float(GroundBased3G.time_to_merger(f, 1.22, 0.2495, 0.5, 0.5))
        non_spinning = float(GroundBased3G.time_to_merger(f, 1.22, 0.2495))
        anti_aligned = float(GroundBased3G.time_to_merger(f, 1.22, 0.2495, -0.5, -0.5))
        assert anti_aligned < non_spinning < aligned

    def test_decreases_with_frequency(self):
        """Higher frequencies are emitted later, closer to merger."""
        f = jnp.geomspace(2.0, 2048.0, 4000)
        tau = GroundBased3G.time_to_merger(f, 1.22, 0.2495, 0.03, -0.02)
        assert jnp.all(jnp.diff(tau) < 0)
        assert jnp.all(tau > 0)

    @pytest.mark.parametrize("mode", [1, 3, 4, -3])
    def test_mode_rescales_the_frequency(self, mode):
        """tau_m(f) = tau_22(2 f / |m|)."""
        f = jnp.geomspace(5.0, 500.0, 50)
        np.testing.assert_allclose(
            GroundBased3G.time_to_merger(f, 1.22, 0.2495, 0.1, 0.0, mode=mode),
            GroundBased3G.time_to_merger(2 * f / abs(mode), 1.22, 0.2495, 0.1, 0.0),
            rtol=1e-14,
            atol=0.0,
        )

    def test_zero_mode_is_rejected(self):
        """m = 0 has no frequency-to-time mapping; it must not divide by zero."""
        with pytest.raises(ValueError, match="non-zero"):
            GroundBased3G.time_to_merger(jnp.array([10.0]), 1.22, 0.2495, mode=0)


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

    def test_each_frequency_uses_the_2g_response_at_its_sidereal_angle(
        self, monkeypatch
    ):
        """The response at f is the 2G response with gmst replaced by GMST(f)."""
        det_3g = _et1_3g()
        det_2g = _start_at(_as_2g(get_ET()[0]), ROTATION_TRIGGER_TIME - 100)
        h_sky = _unit_polarizations(self.frequency)
        response = det_3g.fd_response(self.frequency, h_sky, BNS_PARAMS)
        gmst_f = det_3g.gmst_at_frequency(self.frequency, BNS_PARAMS)
        for i in [0, 100, 300, 511]:
            f_i = self.frequency[i : i + 1]
            h_i = {k: v[i : i + 1] for k, v in h_sky.items()}
            _sidereal_angle_at_trigger(monkeypatch, gmst_f[i])
            expected = det_2g.fd_response(f_i, h_i, BNS_PARAMS)
            np.testing.assert_allclose(response[i], expected[0], rtol=1e-12, atol=0.0)

    def test_sidereal_angle_matches_gmst_at_emission(self):
        """GMST(f) agrees with the GMST of the emission time t_trigger + t_c - tau(f)."""
        det = _et1_3g()
        gmst_f = det.gmst_at_frequency(self.frequency, BNS_PARAMS)
        tau = GroundBased3G.time_to_merger(
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
        at_merger = BNS_GMST + EARTH_ROTATION_RATE * BNS_PARAMS["t_c"]
        assert jnp.all(jnp.diff(gmst_f) > 0)
        assert jnp.all(gmst_f < at_merger)
        # The zero-frequency bin has no emission time; it is put at merger.
        dc = det.gmst_at_frequency(jnp.array([0.0, 5.0]), BNS_PARAMS)
        assert jnp.all(jnp.isfinite(dc))
        np.testing.assert_allclose(dc[0], at_merger, rtol=1e-15, atol=0.0)

    def test_full_turn_of_gmst_leaves_the_response_unchanged(self, monkeypatch):
        """The response is 2 pi periodic in the sidereal angle."""
        det = _et1_3g(finite_arm_length=True, arm_length=1e4)
        h_sky = _unit_polarizations(self.frequency)
        reference = det.fd_response(self.frequency, h_sky, BNS_PARAMS)
        _sidereal_angle_at_trigger(monkeypatch, BNS_GMST + 2 * np.pi)
        np.testing.assert_allclose(
            det.fd_response(self.frequency, h_sky, BNS_PARAMS),
            reference,
            rtol=1e-11,
            atol=0.0,
        )

    def test_only_the_hour_angle_matters(self, monkeypatch):
        """Rotating the sky and the Earth together leaves the response unchanged."""
        det = _et1_3g(finite_arm_length=True, arm_length=1e4)
        h_sky = _unit_polarizations(self.frequency)
        reference = det.fd_response(self.frequency, h_sky, BNS_PARAMS)
        delta = 0.9
        shifted = {**BNS_PARAMS, "ra": BNS_PARAMS["ra"] + delta}
        _sidereal_angle_at_trigger(monkeypatch, BNS_GMST + delta)
        np.testing.assert_allclose(
            det.fd_response(self.frequency, h_sky, shifted),
            reference,
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


class TestET2L:
    """get_ET_2L: two 15 km L-shaped detectors, in Sardinia and in the Euregio Meuse-Rhine."""

    def test_two_l_shaped_detectors_with_15_km_arms(self):
        ifos = get_ET_2L(alignment_angle=45.0)
        assert [ifo.name for ifo in ifos] == ["ET_Sar", "ET_EMR"]
        for ifo in ifos:
            assert isinstance(ifo, GroundBased3G)
            assert ifo.arm_length == 1.5e4
            np.testing.assert_allclose(ifo.yarm_azimuth - ifo.xarm_azimuth, np.pi / 2)

    def test_the_sites(self):
        sardinia, emr = get_ET_2L(alignment_angle=0.0)
        np.testing.assert_allclose(
            np.degrees([sardinia.latitude, sardinia.longitude]),
            [40 + 31 / 60, 9 + 25 / 60],
        )
        np.testing.assert_allclose(
            np.degrees([emr.latitude, emr.longitude]),
            [50 + 43 / 60 + 23 / 3600, 5 + 55 / 60 + 14 / 3600],
        )

    def test_the_misaligned_case_has_the_arm_azimuths_of_the_bilby_files(self):
        sardinia, emr = get_ET_2L(alignment_angle=45.0)
        np.testing.assert_allclose(
            np.degrees([sardinia.xarm_azimuth, sardinia.yarm_azimuth]),
            [70.5674, 160.5674],
        )
        np.testing.assert_allclose(
            np.degrees([emr.xarm_azimuth, emr.yarm_azimuth]), [25.5674, 115.5674]
        )

    @pytest.mark.parametrize("angle", [0.0, 10.0, 45.0, 90.0])
    def test_the_alignment_angle_rotates_only_the_emr_arms(self, angle):
        sardinia, emr = get_ET_2L(alignment_angle=angle)
        np.testing.assert_allclose(np.degrees(sardinia.xarm_azimuth), 70.5674)
        np.testing.assert_allclose(
            np.degrees(sardinia.xarm_azimuth - emr.xarm_azimuth), angle, atol=1e-12
        )
        np.testing.assert_allclose(
            np.degrees(sardinia.yarm_azimuth - emr.yarm_azimuth), angle, atol=1e-12
        )

    def test_the_default_alignment_angle_is_42_5_degrees(self):
        default, explicit = get_ET_2L(), get_ET_2L(alignment_angle=42.5)
        for ifo, expected in zip(default, explicit):
            assert ifo.xarm_azimuth == expected.xarm_azimuth
            assert ifo.yarm_azimuth == expected.yarm_azimuth
        np.testing.assert_allclose(
            np.degrees(default[0].xarm_azimuth - default[1].xarm_azimuth), 42.5
        )

    def test_default_options_and_options_reach_both_detectors(self):
        assert not any(
            ifo.earth_rotation or ifo.finite_arm_length
            for ifo in get_ET_2L(alignment_angle=45.0)
        )
        for ifo in get_ET_2L(
            alignment_angle=45.0, earth_rotation=True, finite_arm_length=True
        ):
            assert ifo.earth_rotation
            assert ifo.finite_arm_length


@pytest.mark.parametrize("getter", [get_ET, get_ET_2L, get_CE])
def test_next_generation_presets_take_keyword_arguments_only(getter):
    with pytest.raises(TypeError, match="positional"):
        getter(True)
