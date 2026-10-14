from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries
from scipy.signal import welch
from scipy.signal.windows import tukey

from jimgw.core.single_event.data import Data, PowerSpectrum


class TestData:
    """Tests for the Data class."""

    def setup_method(self):
        self.f_samp = 2048
        self.duration = 4
        self.start_time = 2.0
        self.name = "Dummy"
        delta_t = 1 / self.f_samp
        n_time = int(self.duration / delta_t)
        self.data = Data.from_td(
            td=jnp.ones(n_time),
            delta_t=delta_t,
            name=self.name,
            start_time=self.start_time,
        )

    def test_basic_attributes(self):
        """Basic attributes are set correctly on construction."""
        assert self.data.name == "Dummy"
        assert self.data.start_time == self.start_time
        assert self.data.end_time == self.start_time + self.duration
        assert self.data.duration == self.duration
        assert self.data.delta_t == 1 / self.f_samp
        assert len(self.data.td) == int(self.f_samp * self.duration)

    def test_default_window(self):
        """The default preserves alpha=0.2 for a four-second segment."""
        alpha = 2 * 0.4 / self.duration
        expected = tukey(len(self.data.td), alpha=alpha)
        assert jnp.allclose(self.data.window, expected)

    def test_end_time_is_derived_from_the_actual_samples_and_start_time(self):
        data = Data.from_td(jnp.ones(9), 0.25, start_time=10.0)
        assert data.end_time == 12.25
        assert data.end_time == float(data.times[-1]) + data.delta_t
        data.start_time = 11.0
        assert data.end_time == 13.25
        assert "end_time" not in data.__dict__

    def test_default_window_scales_alpha_for_long_segments(self):
        """Long segments keep the same roll-off time rather than taper fraction."""
        duration = 128
        sampling_frequency = 16
        data = Data.from_td(
            td=jnp.ones(duration * sampling_frequency),
            delta_t=1 / sampling_frequency,
        )

        expected_alpha = 2 * 0.4 / duration
        expected = tukey(len(data.td), alpha=expected_alpha)
        assert jnp.allclose(data.window, expected)

    def test_set_tukey_window_accepts_explicit_alpha(self):
        """Callers can retain direct control of the Tukey shape parameter."""
        self.data.set_tukey_window(alpha=0.4)
        assert jnp.allclose(self.data.window, tukey(len(self.data.td), alpha=0.4))

    def test_set_tukey_window_accepts_custom_roll_off(self):
        """A custom roll-off is converted to alpha using the data duration."""
        self.data.set_tukey_window(roll_off=0.4)
        expected_alpha = 2 * 0.4 / self.duration
        assert jnp.allclose(
            self.data.window, tukey(len(self.data.td), alpha=expected_alpha)
        )

    def test_set_tukey_window_rejects_alpha_and_roll_off(self):
        """Alpha and roll-off are alternative parameterizations."""
        with pytest.raises(ValueError, match="either alpha or roll_off"):
            self.data.set_tukey_window(alpha=0.2, roll_off=0.2)

    @pytest.mark.parametrize("roll_off", [-0.1, np.nan, np.inf, -np.inf])
    def test_set_tukey_window_rejects_invalid_roll_off(self, roll_off):
        with pytest.raises(ValueError, match="finite and between"):
            self.data.set_tukey_window(roll_off=roll_off)

    def test_set_tukey_window_rejects_roll_off_longer_than_half_duration(self):
        with pytest.raises(ValueError, match="finite and between"):
            self.data.set_tukey_window(roll_off=self.duration / 2 + 0.1)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf, -np.inf])
    def test_set_tukey_window_rejects_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match="finite and between 0 and 1"):
            self.data.set_tukey_window(alpha=alpha)

    def test_set_tukey_window_rejects_changes_after_fd_is_read(self):
        """Changing a window cannot invalidate fixed frequency-domain data."""
        _ = self.data.fd
        with pytest.raises(RuntimeError, match="frequency-domain data have been fixed"):
            self.data.set_tukey_window(alpha=0.4)

    @pytest.mark.parametrize("field", ["td", "window", "delta_t"])
    def test_fixed_fd_rejects_direct_source_changes(self, field):
        fd = self.data.fd
        previous = getattr(self.data, field)
        changed = 0.5 if field == "delta_t" else jnp.zeros_like(previous)
        with pytest.raises(RuntimeError, match="frequency-domain data have been fixed"):
            setattr(self.data, field, changed)
        with pytest.raises(RuntimeError, match="frequency-domain data have been fixed"):
            delattr(self.data, field)
        assert getattr(self.data, field) is previous
        assert self.data.fd is fd
        np.testing.assert_allclose(self.data.fft(), fd)

    def test_fixing_fd_detaches_mutable_source_aliases(self):
        td = np.arange(8.0)
        window = np.linspace(0.5, 1.0, 8)
        time_step = np.array(0.25)
        data = Data.from_td(td, time_step, window=window)
        data.td, data.window, data.delta_t = td, window, time_step
        fd = data.fd
        expected_td, expected_window = td.copy(), window.copy()

        td[0], window[0], time_step[()] = 100.0, 0.0, 0.5
        np.testing.assert_array_equal(data.td, expected_td)
        np.testing.assert_array_equal(data.window, expected_window)
        assert data.delta_t == 0.25
        np.testing.assert_allclose(data.fft(), fd)

    def test_fft_only_computes(self):
        """fft() returns the transform and changes nothing, whatever the window."""
        window = self.data.window
        expected = jnp.fft.rfft(self.data.td * window) * self.data.delta_t
        assert jnp.allclose(self.data.fft(), expected)

        ones = jnp.ones_like(self.data.td)
        assert jnp.allclose(
            self.data.fft(window=ones), jnp.fft.rfft(self.data.td) * self.data.delta_t
        )
        assert not self.data.has_fd
        assert self.data.window is window
        self.data.set_tukey_window(alpha=0.4)  # still allowed

        _ = self.data.fd
        assert jnp.allclose(
            self.data.fft(window=ones), jnp.fft.rfft(self.data.td) * self.data.delta_t
        )

    def test_frequency_slice_locks_tukey_window(self):
        """Frequency slices materialize FD data before detectors cache them."""
        self.data.frequency_slice(20, 512)
        with pytest.raises(RuntimeError, match="frequency-domain data have been fixed"):
            self.data.set_tukey_window(alpha=0.4)

    def test_zero_fd_is_fixed(self):
        """An all-zero FD is still a materialized FD representation."""
        data = Data.from_td(td=jnp.zeros_like(self.data.td), delta_t=self.data.delta_t)
        _ = data.fd
        assert data.has_fd
        with pytest.raises(RuntimeError, match="frequency-domain data have been fixed"):
            data.set_tukey_window(alpha=0.4)

    def test_empty_data_cannot_set_a_tukey_window(self):
        """Empty Data instances remain usable as detector placeholders."""
        data = Data()
        assert data.is_empty
        assert data.window.size == 0
        with pytest.raises(ValueError, match="empty data"):
            data.set_tukey_window()

    def test_default_window_rejects_too_short_data(self):
        """A roll-off cannot exceed half of a non-empty data segment."""
        with pytest.raises(ValueError, match="finite and between"):
            Data.from_td(td=jnp.ones(3), delta_t=0.25)

    @pytest.mark.parametrize("delta_t", [0.0, -0.25, np.nan, np.inf, -np.inf])
    def test_nonempty_data_rejects_invalid_delta_t_with_custom_window(self, delta_t):
        """An explicit window cannot bypass time-step validation."""
        with pytest.raises(ValueError, match="delta_t must be finite and positive"):
            Data.from_td(
                td=jnp.ones(4),
                delta_t=delta_t,
                window=jnp.ones(4),
            )

    def test_bool_nonempty(self):
        """bool(data) is True when data are present."""
        assert bool(self.data)

    def test_fd_is_computed_on_first_use(self):
        """Reading fd gives the windowed FFT of the time-domain data, and keeps it."""
        assert not self.data.has_fd
        expected_fd = jnp.fft.rfft(self.data.td * self.data.window) * self.data.delta_t
        assert jnp.allclose(self.data.fd, expected_fd)
        assert self.data.has_fd
        assert self.data.fd is self.data.fd
        fftfreq = jnp.fft.rfftfreq(len(self.data.td), self.data.delta_t)
        assert len(self.data.fd) == len(fftfreq) == self.data.n_freq

    def test_empty_data_have_no_fd(self):
        """There is nothing to transform."""
        with pytest.raises(
            ValueError, match="Empty data have no frequency-domain data"
        ):
            _ = Data().fd

    def test_data_are_created_by_the_named_constructors(self):
        """Data() is empty; data come from from_td, from_fd, from_gwosc or from_file."""
        assert Data().is_empty
        with pytest.raises(TypeError, match="Data.from_td"):
            Data(jnp.ones(8), 0.25)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Data.from_td"):
            Data(td=jnp.ones(8), delta_t=0.25)  # type: ignore[arg-type]

    def test_fd_of_data_built_from_fd_is_the_array_that_was_given(self):
        """fd keeps what was supplied exactly; fft() only recomputes it from td."""
        frequencies = jnp.fft.rfftfreq(8, 0.25)
        fd = jnp.zeros(len(frequencies), dtype=jnp.complex128).at[2].set(1.0)
        data = Data.from_fd(fd, frequencies)
        assert jnp.array_equal(data.fd, fd)
        assert jnp.allclose(data.fft(), fd)

    def test_from_fd_fixes_an_all_zero_frequency_domain_input(self):
        """Imported FD data cannot be re-windowed, even when all values are zero."""
        n_time = 8
        delta_t = 0.25
        frequencies = jnp.fft.rfftfreq(n_time, delta_t)
        data = Data.from_fd(
            jnp.zeros(len(frequencies), dtype=jnp.complex128), frequencies
        )

        assert data.has_fd
        with pytest.raises(RuntimeError, match="frequency-domain data have been fixed"):
            data.set_tukey_window(alpha=0.4)

    def test_trigger_time_defaults_to_two_seconds_before_the_end(self):
        """Without a trigger time, the data place it 2 s before their end."""
        assert self.data.trigger_time == self.start_time + self.duration - 2.0

    def test_trigger_time_is_stored(self):
        """The trigger time given at construction is kept."""
        data = Data.from_td(td=jnp.ones(8), delta_t=0.25, trigger_time=12.5)
        assert data.trigger_time == 12.5

    def test_trigger_time_can_be_set_and_reset(self):
        """Setting the trigger time overrides the default; None restores it."""
        default = self.data.trigger_time
        self.data.trigger_time = 3.0
        assert self.data.trigger_time == 3.0
        self.data.trigger_time = None
        assert self.data.trigger_time == default

    def test_empty_data_have_no_trigger_time(self):
        """There is no segment to place a default in."""
        with pytest.raises(ValueError, match="Empty data have no trigger time"):
            _ = Data().trigger_time
        data = Data()
        data.trigger_time = 12.5
        assert data.trigger_time == 12.5

    def test_from_fd_stores_the_trigger_time(self):
        """Data built from FD data carry the trigger time they are given."""
        frequencies = jnp.fft.rfftfreq(8, 0.25)
        fd = jnp.zeros(len(frequencies), dtype=jnp.complex128)
        default = Data.from_fd(fd, frequencies, start_time=10.0)
        assert default.trigger_time == 10.0 + 2.0 - 2.0
        data = Data.from_fd(fd, frequencies, trigger_time=12.5)
        assert data.trigger_time == 12.5

    def test_frequency_slice_computes_and_keeps_fd(self):
        """Calling frequency_slice computes the FD data and keeps them."""
        fftfreq = jnp.fft.rfftfreq(len(self.data.td), self.data.delta_t)
        expected_fd = jnp.fft.rfft(self.data.td * self.data.window) * self.data.delta_t

        fmin, fmax = 20, 512
        freq_slice, data_slice = self.data.frequency_slice(fmin, fmax)

        freq_mask = (fftfreq >= fmin) & (fftfreq <= fmax)
        assert jnp.allclose(self.data.fd, expected_fd)
        assert jnp.allclose(data_slice, expected_fd[freq_mask])
        assert jnp.allclose(freq_slice, fftfreq[freq_mask])

    def test_fft_matches_frequency_slice(self):
        """fft() gives the same result as the data frequency_slice uses."""
        expected_fd = jnp.fft.rfft(self.data.td * self.data.window) * self.data.delta_t

        data_copy = deepcopy(self.data)
        assert not data_copy.has_fd
        assert jnp.allclose(data_copy.fft(), expected_fd)
        assert not data_copy.has_fd
        assert jnp.allclose(data_copy.fd, expected_fd)

        fmin, fmax = 20, 512
        freq_via_slice, slice_via_slice = self.data.frequency_slice(fmin, fmax)
        freq_via_fft, slice_via_fft = data_copy.frequency_slice(fmin, fmax)
        assert jnp.allclose(slice_via_slice, slice_via_fft)
        assert jnp.allclose(freq_via_slice, freq_via_fft)


class TestDataInputValidation:
    @pytest.mark.parametrize("read_fd", [False, True])
    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("td", np.ones((8, 1)), "td.*one-dimensional real finite"),
            ("td", np.full(8, np.nan), "td.*one-dimensional real finite"),
            ("td", np.ones(8, dtype=complex), "td.*one-dimensional real finite"),
            ("window", np.ones((8, 1)), "window"),
            ("window", np.ones(1), "window"),
            ("delta_t", "0.25", "delta_t.*finite and positive"),
            ("delta_t", np.array([0.25]), "delta_t.*finite and positive"),
            ("delta_t", np.nan, "delta_t.*finite and positive"),
            ("delta_t", 0.0, "delta_t.*finite and positive"),
        ],
    )
    def test_cold_fft_validates_replaced_source_fields(
        self, read_fd, field, value, match
    ):
        data = Data.from_td(jnp.ones(8), 0.25)
        setattr(data, field, value)
        with pytest.raises(ValueError, match=match):
            if read_fd:
                _ = data.fd
            else:
                data.fft()
        assert not data.has_fd

    @pytest.mark.parametrize("n_time", [0, 8])
    @pytest.mark.parametrize(
        "delta_t", [np.nan, np.inf, -np.inf, -0.25, "0.25", [0.25], 0.25 + 0j]
    )
    def test_malformed_time_steps_are_rejected_even_for_empty_data(
        self, n_time, delta_t
    ):
        with pytest.raises(ValueError, match="delta_t must be finite and positive"):
            Data.from_td(jnp.ones(n_time), delta_t, window=jnp.ones(n_time))

    def test_empty_data_can_have_a_zero_time_step(self):
        data = Data.from_td(jnp.array([]), 0.0)
        assert data.is_empty and data.delta_t == 0.0

    @pytest.mark.parametrize("sample,window", [(1e308, 1.0), (1e200, 1e200)])
    def test_fft_overflow_raises_without_fixing_frequency_data(self, sample, window):
        data = Data.from_td(jnp.full(8, sample), 0.25, window=jnp.full(8, window))
        with pytest.raises(ValueError, match="FFT.*finite"):
            _ = data.fd
        assert not data.has_fd

    @pytest.mark.parametrize(
        "td",
        [
            jnp.array(1.0),
            jnp.ones((8, 2)),
            jnp.ones(8, dtype=complex),
            jnp.array([0.0, np.nan]),
            jnp.array([0.0, np.inf]),
        ],
    )
    def test_invalid_samples_are_rejected(self, td):
        with pytest.raises(ValueError, match="td.*one-dimensional real finite"):
            Data.from_td(td, 0.25)

    @pytest.mark.parametrize(
        "window",
        [
            jnp.array(1.0),
            jnp.ones((8, 1)),
            jnp.ones(8, dtype=complex),
            jnp.full(8, np.nan),
            jnp.full(8, np.inf),
            jnp.ones(1),
            jnp.ones(7),
        ],
    )
    def test_invalid_windows_are_rejected_by_constructor_and_fft(self, window):
        with pytest.raises(ValueError, match="window"):
            Data.from_td(jnp.ones(8), 0.25, window=window)
        data = Data.from_td(jnp.ones(8), 0.25)
        with pytest.raises(ValueError, match="window"):
            data.fft(window=window)
        assert not data.has_fd

    @pytest.mark.parametrize("field", ["start_time", "trigger_time"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "12", 1 + 0j, [12.0]])
    def test_invalid_times_are_rejected_at_construction_and_assignment(
        self, field, value
    ):
        with pytest.raises(ValueError, match=f"{field}.*finite real scalar"):
            Data.from_td(jnp.ones(8), 0.25, **{field: value})
        data = Data.from_td(jnp.ones(8), 0.25)
        previous = getattr(data, field)
        with pytest.raises(ValueError, match=f"{field}.*finite real scalar"):
            setattr(data, field, value)
        assert getattr(data, field) == previous

    def test_numpy_and_jax_scalar_metadata_are_stored_as_python_floats(self):
        data = Data.from_td(
            np.ones(8),
            np.array(0.25),
            start_time=np.int64(10),
            trigger_time=jnp.array(11.5),
            window=np.ones(8),
        )
        assert (data.start_time, data.trigger_time, data.delta_t) == (10.0, 11.5, 0.25)
        assert isinstance(data.start_time, float)
        assert isinstance(data.trigger_time, float)


class TestFromFdValidation:
    @pytest.mark.parametrize(
        ("frequencies", "match"),
        [
            ([], "at least two"),
            ([1.0], "at least two"),
            ([0.0, np.nan, 1.0], "real finite"),
            ([0.0, np.inf], "real finite"),
            ([-0.5, 0.0, 0.5], "nonnegative"),
            ([0.0, 0.5, 0.5], "strictly increasing"),
            ([1.0, 0.5, 0.0], "strictly increasing"),
            ([0.0, 0.5, 1.1], "uniformly spaced"),
            ([0.25, 0.75, 1.25], "aligned"),
        ],
    )
    def test_malformed_frequency_grids_raise_value_errors(self, frequencies, match):
        frequencies = jnp.asarray(frequencies)
        with pytest.raises(ValueError, match=match):
            Data.from_fd(jnp.zeros(frequencies.shape, dtype=complex), frequencies)

    @pytest.mark.parametrize("fd", [jnp.ones(4), jnp.ones((5, 1))])
    def test_frequency_data_shape_must_match(self, fd):
        with pytest.raises(ValueError, match="same shape"):
            Data.from_fd(fd, jnp.arange(5.0))

    @pytest.mark.parametrize("value", [np.nan, np.inf, 1 + np.inf * 1j])
    def test_frequency_data_must_be_finite(self, value):
        fd = jnp.ones(5, dtype=complex).at[2].set(value)
        with pytest.raises(ValueError, match="fd.*finite"):
            Data.from_fd(fd, jnp.arange(5.0))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_decimal_frequency_subset_is_accepted_with_roundoff(self, dtype):
        frequencies = jnp.arange(3, 18, dtype=dtype) * 0.1
        fd = jnp.linspace(1.0, 2.0, len(frequencies)).astype(complex)
        data = Data.from_fd(fd, frequencies)
        np.testing.assert_allclose(data.frequencies[3:], frequencies, rtol=1e-6)
        np.testing.assert_allclose(data.fd[3:], fd)
        np.testing.assert_allclose(data.fft(), data.fd, atol=1e-14)

    def test_complex_dc_and_nyquist_are_normalized(self):
        fd = jnp.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, 9 + 10j])
        data = Data.from_fd(fd, jnp.arange(5.0) * 0.5)
        expected = fd.at[0].set(1.0).at[-1].set(9.0)
        assert jnp.array_equal(data.fd, expected)
        np.testing.assert_allclose(data.fft(), expected, atol=1e-14)


class TestFrequencyBounds:
    """The frequency range of the data is stored in the data and checked when set."""

    def setup_method(self):
        # 4 s at 2048 Hz: the frequencies run from 0 Hz to the Nyquist frequency, 1024 Hz.
        self.data = Data.from_td(td=jnp.zeros(8192), delta_t=1 / 2048.0)

    def test_the_default_does_not_restrict_the_data(self):
        assert self.data.frequency_bounds == (0.0, jnp.inf)
        assert bool(jnp.all(self.data.frequency_mask))
        assert Data().frequency_bounds == (0.0, jnp.inf)

    def test_the_range_is_stored_and_selects_the_frequencies(self):
        self.data.frequency_bounds = (20.0, 512.0)
        assert self.data.frequency_bounds == (20.0, 512.0)
        selected = self.data.frequencies[self.data.frequency_mask]
        assert float(selected[0]) == 20.0
        assert float(selected[-1]) == 512.0

    @pytest.mark.parametrize(
        ("bounds", "match"),
        [
            ((jnp.nan, 100.0), "NaN"),
            ((20.0, jnp.nan), "NaN"),
            ((-1.0, 100.0), "negative"),
            ((100.0, 100.0), "below"),
            ((200.0, 100.0), "below"),
            ((20.0, 5000.0), "Nyquist"),
            ((20.0, 20.1), "at least two"),
            (20.0, "pair"),
            ((20.0,), "pair"),
            ((20.0, 100.0, 200.0), "pair"),
            (("low", "high"), "pair"),
            (None, "pair"),
        ],
    )
    def test_an_invalid_range_raises_and_changes_nothing(self, bounds, match):
        self.data.frequency_bounds = (20.0, 512.0)
        with pytest.raises(ValueError, match=match):
            self.data.frequency_bounds = bounds
        assert self.data.frequency_bounds == (20.0, 512.0)

    @pytest.mark.parametrize("sampling_frequency", [501.0, 744.0, 773.0, 2048.0])
    def test_f_max_at_the_nyquist_frequency_is_accepted_for_any_sampling_rate(
        self, sampling_frequency
    ):
        # 1 / (1 / fs) is not exactly fs for many integer rates, such as 744 and 773.
        data = Data.from_td(
            td=jnp.zeros(int(4 * sampling_frequency)), delta_t=1 / sampling_frequency
        )
        data.frequency_bounds = (20.0, sampling_frequency / 2)
        assert data.frequency_bounds == (20.0, sampling_frequency / 2)

    def test_empty_data_have_no_nyquist_frequency_to_check(self):
        data = Data()
        data.frequency_bounds = (20.0, 5000.0)
        assert data.frequency_bounds == (20.0, 5000.0)

    def test_an_infinite_f_max_is_accepted(self):
        self.data.frequency_bounds = (20.0, jnp.inf)
        assert self.data.frequency_bounds == (20.0, jnp.inf)

    def test_setting_the_range_does_not_fix_the_data(self):
        # Setting the range must not Fourier transform the data, which would fix the window.
        self.data.frequency_bounds = (20.0, 512.0)
        assert not self.data.has_fd
        self.data.set_tukey_window(alpha=0.2)  # still allowed


class TestAddFd:
    """``add_fd`` returns a copy of the data with frequency-domain content added."""

    def setup_method(self):
        self.data = Data.from_td(
            td=jnp.ones(8), delta_t=0.25, start_time=10.0, trigger_time=11.0, name="x"
        )
        self.data.frequency_bounds = (0.5, 1.5)
        self.series = jnp.arange(5.0).astype(jnp.complex128) * (1 + 1j)

    def test_the_content_is_added(self):
        new = self.data.add_fd(self.series)
        expected = self.data.fd + self.series.at[-1].set(self.series[-1].real)
        assert jnp.array_equal(new.fd, expected)
        assert jnp.allclose(new.td, jnp.fft.irfft(new.fd) / 0.25)
        np.testing.assert_allclose(new.fft(), new.fd, atol=1e-14)

    def test_the_copy_has_no_window_and_keeps_everything_else(self):
        new = self.data.add_fd(self.series)
        assert jnp.array_equal(new.window, jnp.ones(8))
        assert new.has_fd
        assert (new.name, new.start_time, new.trigger_time, new.delta_t) == (
            "x",
            10.0,
            11.0,
            0.25,
        )
        assert new.frequency_bounds == (0.5, 1.5)

    def test_the_data_are_not_changed(self):
        fd, td = self.data.fd, self.data.td
        self.data.add_fd(self.series)
        assert self.data.fd is fd
        assert self.data.td is td

    def test_an_odd_number_of_samples_keeps_its_grid(self):
        data = Data.from_td(td=jnp.ones(9), delta_t=0.25)
        series = jnp.zeros(5, dtype=complex).at[0].set(1 + 2j).at[-1].set(3 + 4j)
        new = data.add_fd(series)
        assert (new.n_time, new.delta_t) == (9, 0.25)
        assert jnp.array_equal(new.frequencies, data.frequencies)
        np.testing.assert_allclose(new.fd[-1], data.fd[-1] + 3 + 4j)
        assert new.fd[0].imag == 0
        np.testing.assert_allclose(new.fft(), new.fd, atol=1e-14)

    def test_addition_preserves_the_original_lazy_fft_and_window(self):
        window = self.data.window
        new = self.data.add_fd(self.series)
        assert new.has_fd
        assert not self.data.has_fd
        assert self.data.window is window
        self.data.set_tukey_window(alpha=0.5)

    @pytest.mark.parametrize("value", [np.nan, np.inf, complex(0, np.inf)])
    def test_nonfinite_additions_raise_without_freezing_the_original_window(
        self, value
    ):
        with pytest.raises(ValueError, match="finite"):
            self.data.add_fd(self.series.at[2].set(value))
        assert not self.data.has_fd

    def test_a_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            self.data.add_fd(jnp.zeros(4, dtype=jnp.complex128))
        assert not self.data.has_fd

    def test_empty_data_cannot_take_fd(self):
        with pytest.raises(ValueError, match="no frequency-domain data"):
            Data().add_fd(jnp.zeros(1, dtype=jnp.complex128))


class TestFrozenData:
    @pytest.mark.parametrize(
        "attribute",
        [
            "td",
            "window",
            "_fd",
            "delta_t",
            "start_time",
            "trigger_time",
            "frequency_bounds",
            "name",
            "_frozen",
        ],
    )
    def test_frozen_data_reject_assignment_and_deletion(self, attribute):
        data = Data.from_td(jnp.ones(8), 0.25, window=jnp.ones(8))
        data._freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            setattr(data, attribute, getattr(data, attribute))
        with pytest.raises(RuntimeError, match="frozen"):
            delattr(data, attribute)

    def test_freezing_materializes_fd_and_repeated_freezing_keeps_array_identity(self):
        data = Data.from_td(jnp.arange(8.0), 0.25)
        td, window = data.td, data.window
        assert not data.is_frozen
        assert not data.has_fd
        data._freeze()
        fd = data.fd
        assert data.is_frozen
        assert data.has_fd
        assert data.td is td and data.window is window
        data._freeze()
        assert data.td is td and data.window is window and data.fd is fd
        np.testing.assert_allclose(jax.jit(lambda: data.fd)(), fd)

    @pytest.mark.parametrize("n_time", [8, 9])
    def test_frequency_grid_follows_the_time_step_until_freezing(self, n_time):
        data = Data.from_td(jnp.arange(float(n_time)), 0.25)
        before = data.frequencies
        data.delta_t = 0.5
        np.testing.assert_array_equal(data.frequencies, before / 2)
        data._freeze()
        frequencies = data.frequencies
        np.testing.assert_array_equal(frequencies, jnp.fft.rfftfreq(n_time, 0.5))
        np.testing.assert_array_equal(jax.jit(lambda: data.frequencies)(), frequencies)

    @pytest.mark.parametrize("supplied_fd", [False, True])
    def test_freezing_detaches_numpy_payloads_and_scalar_time_step(self, supplied_fd):
        td = np.arange(8.0)
        window = np.linspace(0.5, 1.0, 8)
        delta_t = np.array(0.25)
        fd = np.fft.rfft(td * window) * delta_t
        data = Data.from_td(td, delta_t, window=window)
        # Public fields can hold mutable arrays before likelihood initialization.
        data.td, data.window, data.delta_t = td, window, delta_t
        if supplied_fd:
            data._fd = fd
        data._freeze()
        owned_td, owned_window, owned_fd = (
            np.array(value) for value in (data.td, data.window, data.fd)
        )
        td[0], window[0], fd[1], delta_t[()] = 100.0, 0.0, 100 + 100j, 0.5
        np.testing.assert_array_equal(data.td, owned_td)
        np.testing.assert_array_equal(data.window, owned_window)
        np.testing.assert_array_equal(data.fd, owned_fd)
        assert data.delta_t == 0.25
        assert isinstance(data.delta_t, float)
        for value in (data.td, data.window, data.fd):
            assert isinstance(value, jax.Array)
            with pytest.raises(TypeError):
                value[0] = 0.0

    def test_is_frozen_is_read_only_and_a_frozen_window_cannot_be_reset(self):
        data = Data.from_td(jnp.ones(8), 0.25)
        with pytest.raises(AttributeError):
            data.is_frozen = True
        data._freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            data.is_frozen = False
        with pytest.raises(RuntimeError, match="frozen"):
            data.set_tukey_window(alpha=0.2)

    @pytest.mark.parametrize("trigger_time", [None, 11.5])
    def test_add_fd_to_frozen_data_returns_mutable_same_class_data(self, trigger_time):
        class CustomData(Data):
            pass

        data = CustomData.from_td(
            jnp.ones(9),
            0.25,
            start_time=10.0,
            trigger_time=trigger_time,
            name="original",
        )
        data.frequency_bounds = (0.5, 1.5)
        data._freeze()
        original_fd = data.fd
        added = data.add_fd(jnp.arange(data.n_freq) * (1 + 2j))
        assert type(added) is CustomData
        assert not added.is_frozen
        assert added.has_fd
        assert added._trigger_time == data._trigger_time
        assert added.start_time == data.start_time
        assert added.frequency_bounds == data.frequency_bounds
        np.testing.assert_array_equal(added.window, jnp.ones(9))
        np.testing.assert_allclose(added.fft(), added.fd, atol=1e-14)
        added.name = "copy"
        added.trigger_time = 12.0
        assert data.name == "original"
        assert data.fd is original_fd
        assert data.is_frozen

    def test_frozen_data_can_be_read_sliced_and_transformed(self):
        data = Data.from_td(jnp.arange(8.0), 0.25)
        data._freeze()
        fd = data.fd
        frequencies, strain = data.frequency_slice(0.5, 1.5)
        np.testing.assert_array_equal(frequencies, [0.5, 1.0, 1.5])
        np.testing.assert_array_equal(strain, fd[1:4])
        np.testing.assert_allclose(data.fft(), fd)
        psd = data.to_psd(nperseg=4)
        assert not psd.is_frozen
        assert data.fd is fd


class TestFrozenPowerSpectrum:
    @pytest.mark.parametrize("attribute", ["values", "frequencies", "name", "_frozen"])
    def test_frozen_psd_rejects_assignment_and_deletion(self, attribute):
        psd = PowerSpectrum(jnp.ones(5), jnp.arange(5.0), "original")
        psd._freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            setattr(psd, attribute, getattr(psd, attribute))
        with pytest.raises(RuntimeError, match="frozen"):
            delattr(psd, attribute)

    def test_freezing_detaches_numpy_aliases(self):
        values, frequencies = np.arange(1.0, 6.0), np.arange(5.0)
        psd = PowerSpectrum(values, frequencies)
        psd._freeze()
        owned_values, owned_frequencies = (
            np.array(psd.values),
            np.array(psd.frequencies),
        )
        values[0], frequencies[0] = 100.0, 100.0
        np.testing.assert_array_equal(psd.values, owned_values)
        np.testing.assert_array_equal(psd.frequencies, owned_frequencies)
        for value in (psd.values, psd.frequencies):
            assert isinstance(value, jax.Array)
            with pytest.raises(TypeError):
                value[0] = 0.0

    def test_repeated_freezing_retains_existing_jax_array_identity(self):
        values, frequencies = jnp.ones(5), jnp.arange(5.0)
        psd = PowerSpectrum(values, frequencies)
        assert not psd.is_frozen
        psd._freeze()
        psd._freeze()
        assert psd.is_frozen
        assert psd.values is values and psd.frequencies is frequencies

    def test_interpolation_from_frozen_psd_returns_a_mutable_psd(self):
        psd = PowerSpectrum(jnp.arange(1.0, 6.0), jnp.arange(5.0))
        psd._freeze()
        values = psd.values
        result = psd.interpolate(jnp.linspace(0.0, 4.0, 9))
        assert not result.is_frozen
        np.testing.assert_allclose(result.values, jnp.linspace(1.0, 5.0, 9))
        result.name = "interpolated"
        result.values = jnp.ones(9)
        assert psd.values is values
        assert psd.is_frozen


class TestPowerSpectrumInputValidation:
    @pytest.mark.parametrize("field", ["values", "frequencies"])
    @pytest.mark.parametrize(
        "value", [np.array(1.0), np.ones((5, 1)), np.ones(5, dtype=complex), ["1"] * 5]
    )
    def test_non_real_or_non_vector_arrays_are_rejected(self, field, value):
        arrays = {"values": np.ones(5), "frequencies": np.arange(5.0)}
        arrays[field] = value
        with pytest.raises(
            ValueError, match=f"PSD {field}.*one-dimensional real numeric"
        ):
            PowerSpectrum(**arrays)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            PowerSpectrum(np.ones(4), np.arange(5.0))

    @pytest.mark.parametrize(
        "frequencies",
        [[0.0, 1.0, np.nan], [0.0, 1.0, np.inf], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    )
    def test_invalid_frequency_coordinates_are_rejected(self, frequencies):
        with pytest.raises(ValueError, match="finite, nonnegative and unique"):
            PowerSpectrum(np.ones(3), frequencies)

    def test_distinct_unsorted_frequencies_and_invalid_values_outside_the_slice_are_preserved(
        self,
    ):
        frequencies = jnp.array([2.0, 0.0, 4.0, 1.0, 3.0])
        values = jnp.array([1.0, jnp.inf, 0.0, 1.0, jnp.nan])
        psd = PowerSpectrum(values, frequencies)
        assert psd.values is values and psd.frequencies is frequencies
        selected, selected_values = psd.frequency_slice(1.0, 2.0)
        np.testing.assert_array_equal(selected, [2.0, 1.0])
        np.testing.assert_array_equal(selected_values, [1.0, 1.0])

    def test_interpolation_extends_values_at_frequency_extrema_on_an_unsorted_grid(
        self,
    ):
        psd = PowerSpectrum(jnp.array([20.0, 10.0, 30.0]), jnp.array([2.0, 1.0, 3.0]))
        target = jnp.array([0.0, 1.5, 4.0])
        interpolated = psd.interpolate(target)
        np.testing.assert_array_equal(interpolated.values, [10.0, 15.0, 30.0])

    @pytest.mark.parametrize("n_freq", [0, 1])
    def test_interpolation_needs_two_frequency_coordinates(self, n_freq):
        psd = PowerSpectrum(jnp.ones(n_freq), jnp.arange(float(n_freq)))
        with pytest.raises(ValueError, match="at least two frequencies"):
            psd.interpolate(jnp.arange(3.0))

    def test_uniform_unsorted_grid_has_correct_segment_metadata_and_noise(self):
        psd = PowerSpectrum(jnp.ones(4), jnp.array([2.0, 4.0, 1.0, 3.0]))
        assert float(psd.delta_f) == 1.0
        assert float(psd.duration) == 1.0
        assert float(psd.sampling_frequency) == 8.0
        assert float(psd.delta_t) == 0.125
        noise = psd.simulate_data(jax.random.key(0))
        assert noise.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(noise)))
        np.testing.assert_allclose(
            jax.jit(psd.simulate_data)(jax.random.key(0)), noise, rtol=1e-12
        )
        assert float(jax.jit(lambda: psd.sampling_frequency)()) == 8.0

    @pytest.mark.parametrize("operation", ["delta_f", "duration", "simulate_data"])
    def test_nonuniform_grid_cannot_supply_frequency_spacing(self, operation):
        psd = PowerSpectrum(jnp.ones(3), jnp.array([0.0, 1.0, 3.0]))
        assert float(psd.sampling_frequency) == 6.0
        with pytest.raises(ValueError, match="uniform frequency grid"):
            if operation == "simulate_data":
                psd.simulate_data(jax.random.key(0))
            else:
                getattr(psd, operation)

    @pytest.mark.parametrize("n_freq", [0, 1])
    @pytest.mark.parametrize("operation", ["delta_f", "duration", "simulate_data"])
    def test_spacing_operations_need_at_least_two_frequencies(self, n_freq, operation):
        psd = PowerSpectrum(jnp.ones(n_freq), jnp.arange(float(n_freq)))
        with pytest.raises(ValueError, match="at least two frequencies"):
            if operation == "simulate_data":
                psd.simulate_data(jax.random.key(0))
            else:
                getattr(psd, operation)

    @pytest.mark.parametrize(
        "frequencies", [jnp.array([]), jnp.array([0.0]), jnp.array([1e308])]
    )
    def test_sampling_frequency_must_be_defined_and_finite(self, frequencies):
        psd = PowerSpectrum(jnp.ones(len(frequencies)), frequencies)
        with pytest.raises(ValueError, match="sampling frequency"):
            _ = psd.sampling_frequency

    @pytest.mark.parametrize("value", [0.0, -1.0, jnp.nan, jnp.inf])
    def test_noise_simulation_checks_every_psd_value(self, value):
        psd = PowerSpectrum(jnp.array([1.0, value, 1.0]), jnp.arange(3.0))
        with pytest.raises(ValueError, match="finite and positive PSD values"):
            psd.simulate_data(jax.random.key(0))

    def test_noise_simulation_rejects_overflowing_variance(self):
        psd = PowerSpectrum(jnp.full(3, 1e308), jnp.array([0.0, 1e-308, 2e-308]))
        with pytest.raises(
            ValueError, match="Noise variance must be finite and positive"
        ):
            psd.simulate_data(jax.random.key(0))


class TestPowerSpectrum:
    """Tests for the PowerSpectrum class."""

    def setup_method(self):
        self.f_samp = 2048
        self.duration = 4
        self.name = "Dummy"
        delta_t = 1 / self.f_samp
        n_time = int(self.duration / delta_t)
        self.data = Data.from_td(
            td=jnp.ones(n_time), delta_t=delta_t, name=self.name, start_time=0.0
        )

        delta_f = 1 / self.duration
        self.psd_range = (20, 512)
        psd_min, psd_max = self.psd_range
        freqs = jnp.arange(int(psd_max / delta_f)) * delta_f
        freqs_psd = freqs[freqs >= psd_min]
        self.psd = PowerSpectrum(
            jnp.ones_like(freqs_psd), frequencies=freqs_psd, name=self.name
        )

    def test_basic_attributes(self):
        """Basic attributes are set correctly on construction."""
        assert self.psd.name == "Dummy"
        assert self.psd.n_freq == len(self.psd.frequencies)
        assert jnp.all(self.psd.frequencies >= self.psd_range[0])
        assert jnp.all(self.psd.frequencies <= self.psd_range[1])

    def test_the_arrays_of_the_caller_are_not_shared(self):
        """Changing the caller's numpy arrays afterwards does not change the PSD."""
        values, frequencies = np.ones(5), np.arange(5.0)
        psd = PowerSpectrum(values, frequencies)
        values[0], frequencies[1] = 100.0, 50.0
        assert float(psd.values[0]) == 1.0
        assert float(psd.frequencies[1]) == 1.0

    def test_bool_nonempty(self):
        """bool(psd) is True when data are present."""
        assert bool(self.psd)

    def test_frequency_slice(self):
        """Slicing the PSD to its own frequency range returns the full array."""
        freq_slice, sliced_psd = self.psd.frequency_slice(*self.psd_range)
        assert jnp.allclose(sliced_psd, self.psd.values)
        assert jnp.allclose(freq_slice, self.psd.frequencies)

    def test_welch_psd_from_data(self):
        """PSD estimated from data via Welch's method matches scipy."""
        # The constant data of the fixture have an all-zero PSD, which any estimate matches.
        noise = np.random.default_rng(0).normal(size=self.data.n_time)
        data = Data.from_td(noise, 1 / self.f_samp, name=self.name)
        nperseg = data.n_time // 2
        psd_auto = data.to_psd(nperseg=nperseg)
        freq_manual, psd_manual = welch(noise, fs=self.f_samp, nperseg=nperseg)
        assert float(jnp.max(psd_manual)) > 0
        np.testing.assert_allclose(psd_auto.frequencies, freq_manual, rtol=1e-12)
        np.testing.assert_allclose(psd_auto.values, psd_manual, rtol=1e-12)

    def test_welch_psd_does_not_fix_the_tukey_window(self):
        """Welch estimation operates directly on time-domain data."""
        self.data.to_psd(nperseg=self.data.n_time // 2)
        assert not self.data.has_fd
        self.data.set_tukey_window(alpha=0.4)

    def test_interpolate_returns_power_spectrum(self):
        """Interpolating the PSD to a new frequency grid returns a PowerSpectrum."""
        psd_interp = self.psd.interpolate(self.data.frequencies)
        assert isinstance(psd_interp, PowerSpectrum)

    def test_simulate_data_variance(self):
        """Simulated FD noise has the expected variance."""
        fd_data = self.psd.simulate_data(jax.random.key(0))

        target_var = self.psd.values / (4 * self.psd.delta_f)
        assert jnp.allclose(jnp.var(fd_data.real), target_var, rtol=1e-1)
        assert jnp.allclose(jnp.var(fd_data.imag), target_var, rtol=1e-1)

    def test_simulate_data_whitened_unit_variance(self):
        """Whitened time-domain noise from simulated data has unit variance."""
        fd_data = self.psd.simulate_data(jax.random.key(0))

        fd_data_white = fd_data / jnp.sqrt(self.psd.values / 2 / self.psd.delta_t)
        td_data_white = jnp.fft.irfft(fd_data_white) / self.psd.delta_t
        assert jnp.allclose(jnp.var(td_data_white), 1, rtol=1e-1)


class TestPowerSpectrumFromFile:
    """Tests for PowerSpectrum.from_file across all supported formats."""

    _FREQS = np.array([10.0, 20.0, 30.0])
    _PSD = np.array([1e-46, 4e-46, 9e-46])
    _ASD = np.sqrt(_PSD)

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def test_npz_roundtrip(self, tmp_path: Path):
        """.npz file is loaded with the correct values and frequencies."""
        path = str(tmp_path / "psd.npz")
        np.savez(path, values=self._PSD, frequencies=self._FREQS, name="H1")
        psd = PowerSpectrum.from_file(path)
        np.testing.assert_allclose(psd.values, self._PSD, rtol=1e-12, atol=0)
        assert jnp.allclose(psd.frequencies, jnp.array(self._FREQS))
        assert psd.name == "H1"

    def test_npz_asd_squared(self, tmp_path: Path):
        """is_asd=True squares the values of an .npz archive, as for text files."""
        path = str(tmp_path / "asd.npz")
        np.savez(path, values=self._ASD, frequencies=self._FREQS)
        psd = PowerSpectrum.from_file(path, is_asd=True)
        np.testing.assert_allclose(psd.values, self._PSD, rtol=1e-12, atol=0)

    @pytest.mark.parametrize("ext", [".npz", ".txt", ".csv"])
    def test_negative_asd_values_are_rejected_before_squaring(self, tmp_path, ext):
        path = str(tmp_path / f"negative{ext}")
        values, frequencies = np.array([1.0, -1.0, 1.0]), np.arange(3.0)
        if ext == ".npz":
            np.savez(path, values=values, frequencies=frequencies)
        else:
            np.savetxt(
                path,
                np.column_stack([frequencies, values]),
                delimiter="," if ext == ".csv" else " ",
            )
        with pytest.raises(ValueError, match="ASD values must be real and nonnegative"):
            PowerSpectrum.from_file(path, is_asd=True)

    def test_npz_missing_keys_raises(self, tmp_path: Path):
        path = str(tmp_path / "bad.npz")
        np.savez(path, values=self._PSD)  # missing 'frequencies'
        with pytest.raises(ValueError, match="must contain"):
            PowerSpectrum.from_file(path)

    # ------------------------------------------------------------------
    # TXT / DAT
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ext", [".txt", ".dat"])
    def test_text_psd_file(self, tmp_path: Path, ext: str):
        """Two-column text file loads correctly as PSD (is_asd=False)."""
        path = str(tmp_path / f"psd{ext}")
        np.savetxt(path, np.column_stack([self._FREQS, self._PSD]))
        psd = PowerSpectrum.from_file(path, is_asd=False)
        np.testing.assert_allclose(psd.values, self._PSD, rtol=1e-12, atol=0)
        assert jnp.allclose(psd.frequencies, jnp.array(self._FREQS))

    @pytest.mark.parametrize("ext", [".txt", ".dat"])
    def test_text_asd_file_squared(self, tmp_path: Path, ext: str):
        """is_asd=True squares the loaded values to give a PSD."""
        path = str(tmp_path / f"asd{ext}")
        np.savetxt(path, np.column_stack([self._FREQS, self._ASD]))
        psd = PowerSpectrum.from_file(path, is_asd=True)
        np.testing.assert_allclose(psd.values, self._PSD, rtol=1e-12, atol=0)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def test_csv_psd_file(self, tmp_path: Path):
        """Comma-separated two-column CSV loads as PSD."""
        path = str(tmp_path / "psd.csv")
        np.savetxt(path, np.column_stack([self._FREQS, self._PSD]), delimiter=",")
        psd = PowerSpectrum.from_file(path)
        np.testing.assert_allclose(psd.values, self._PSD, rtol=1e-12, atol=0)

    def test_csv_asd_file_squared(self, tmp_path: Path):
        """CSV with ASD values is squared correctly when is_asd=True."""
        path = str(tmp_path / "asd.csv")
        np.savetxt(path, np.column_stack([self._FREQS, self._ASD]), delimiter=",")
        psd = PowerSpectrum.from_file(path, is_asd=True)
        np.testing.assert_allclose(psd.values, self._PSD, rtol=1e-12, atol=0)

    # ------------------------------------------------------------------
    # Unsupported
    # ------------------------------------------------------------------

    def test_unsupported_extension_raises(self, tmp_path: Path):
        path = str(tmp_path / "psd.xyz")
        with pytest.raises(ValueError, match="Unsupported file format"):
            PowerSpectrum.from_file(path)


class TestPowerSpectrumToFile:
    """Tests for PowerSpectrum.to_file (npz only)."""

    _FREQS = np.array([10.0, 20.0, 30.0])
    _PSD = np.array([1e-46, 4e-46, 9e-46])
    _NAME = "H1"

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def test_npz_roundtrip(self, tmp_path: Path):
        """PowerSpectrum saved as .npz reloads to identical arrays."""
        path = str(tmp_path / "psd.npz")
        ps = PowerSpectrum(jnp.array(self._PSD), jnp.array(self._FREQS), self._NAME)
        ps.to_file(path)
        ps2 = PowerSpectrum.from_file(path)
        # a save/load round-trip must recover the arrays exactly (npz is lossless);
        # default allclose atol=1e-8 would pass on PSD values ~1e-46 no matter what.
        np.testing.assert_array_equal(np.asarray(ps2.values), np.asarray(ps.values))
        np.testing.assert_array_equal(
            np.asarray(ps2.frequencies), np.asarray(ps.frequencies)
        )
        assert ps2.name == self._NAME


# ----------------------------------------------------------------------
# Helpers shared across file-loading tests
# ----------------------------------------------------------------------


def _make_mock_timeseries(n=8192, dt=1 / 2048.0, epoch=0.0):
    """Return a MagicMock that looks like a gwpy TimeSeries."""
    ts = MagicMock()
    ts.value = np.zeros(n)
    ts.dt.value = dt
    ts.epoch.value = epoch
    return ts


# ----------------------------------------------------------------------
class TestDataFromGwosc:
    @pytest.mark.parametrize("trigger_time", [None, 1126259461.0, 0.0])
    def test_actual_interval_matches_the_returned_gwpy_series(self, trigger_time):
        requested_start = 1126259460.413
        requested_end = requested_start + 4.0
        rate = 2048
        actual_start = np.ceil(requested_start * rate) / rate
        fetched = TimeSeries(np.zeros(4 * rate - 3), t0=actual_start, dt=1 / rate)
        with patch(
            "jimgw.core.single_event.data.TimeSeries.fetch_open_data",
            return_value=fetched,
        ) as fetch:
            data = Data.from_gwosc(
                "H1", requested_start, requested_end, trigger_time=trigger_time
            )

        assert data.start_time == fetched.span[0]
        assert data.end_time == fetched.span[1]
        assert data.start_time != requested_start
        assert data.end_time != requested_end
        expected_trigger = requested_end - 2.0 if trigger_time is None else trigger_time
        assert data.trigger_time == expected_trigger
        fetch.assert_called_once_with("H1", requested_start, requested_end, cache=True)

    @pytest.mark.parametrize("second_rate", [2048, 4096])
    def test_requested_interval_sets_a_shared_trigger_despite_sample_rounding(
        self, second_rate
    ):
        requested_start = 1126259460.413
        requested_end = requested_start + 4.0
        rates = (2048, second_rate)
        epochs = (
            np.floor(requested_start * rates[0]) / rates[0],
            np.ceil(requested_start * rates[1]) / rates[1],
        )
        counts = (4 * rates[0] - 2, 4 * rates[1] + 3)
        series = [
            _make_mock_timeseries(n=count, dt=1 / rate, epoch=epoch)
            for count, rate, epoch in zip(counts, rates, epochs, strict=True)
        ]
        with patch(
            "jimgw.core.single_event.data.TimeSeries.fetch_open_data",
            side_effect=series,
        ) as fetch:
            data = [
                Data.from_gwosc(
                    name, requested_start, requested_end, cache=False, sample_rate=rate
                )
                for name, rate in zip(("H1", "L1"), rates, strict=True)
            ]

        assert data[0].trigger_time == data[1].trigger_time == requested_end - 2.0
        for strain, epoch, rate, count in zip(data, epochs, rates, counts, strict=True):
            assert strain.start_time == epoch
            assert strain.delta_t == 1 / rate
            assert strain.n_time == count
            assert strain.duration == count / rate
            assert strain.end_time == epoch + count / rate
            assert strain.start_time != requested_start
            assert strain.end_time != requested_end
            assert strain.trigger_time != strain.end_time - 2.0
        assert [strain.name for strain in data] == ["H1", "L1"]
        assert fetch.call_count == 2
        fetch.assert_any_call(
            "H1", requested_start, requested_end, cache=False, sample_rate=rates[0]
        )
        fetch.assert_any_call(
            "L1", requested_start, requested_end, cache=False, sample_rate=rates[1]
        )

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (np.nan, 14.0),
            (10.0, np.nan),
            (np.inf, 14.0),
            (10.0, np.inf),
            (-np.inf, 14.0),
            (10.0, -np.inf),
            ([10.0], 14.0),
            (10.0, [14.0]),
            ("10", 14.0),
            (10.0, "14"),
            (14.0, 14.0),
            (15.0, 14.0),
        ],
    )
    def test_invalid_requested_intervals_are_rejected_before_fetching(self, start, end):
        with (
            patch("jimgw.core.single_event.data.TimeSeries.fetch_open_data") as fetch,
            pytest.raises(ValueError),
        ):
            Data.from_gwosc("H1", start, end)
        fetch.assert_not_called()

    @pytest.mark.parametrize(
        "trigger_time", [jnp.nan, jnp.inf, -jnp.inf, [12.0], "12", 12 + 1j]
    )
    def test_invalid_trigger_times_are_rejected_before_fetching(self, trigger_time):
        with (
            patch("jimgw.core.single_event.data.TimeSeries.fetch_open_data") as fetch,
            pytest.raises(
                ValueError, match="trigger_time must be a finite real scalar"
            ),
        ):
            Data.from_gwosc("H1", 10.0, 14.0, trigger_time=trigger_time)
        fetch.assert_not_called()

    def test_numpy_integer_epoch_is_a_valid_scalar(self):
        fetched = _make_mock_timeseries(n=8, dt=0.25, epoch=np.int64(10))
        with patch(
            "jimgw.core.single_event.data.TimeSeries.fetch_open_data",
            return_value=fetched,
        ):
            data = Data.from_gwosc("H1", 10.0, 12.0)
        assert data.start_time == 10.0
        assert isinstance(data.start_time, float)


class TestDataFromFile:
    """Tests for Data.from_file and Data._from_gwf."""

    @pytest.mark.parametrize("source", ["gwosc", "gwf", "hdf5"])
    @pytest.mark.parametrize("field", ["dt", "epoch"])
    @pytest.mark.parametrize(
        "value", [np.nan, np.inf, "0.25", np.array([0.25]), 0.25 + 0j]
    )
    def test_gwpy_metadata_are_validated_without_scalar_coercion(
        self, source, field, value
    ):
        series = _make_mock_timeseries()
        getattr(series, field).value = value
        with (
            patch("jimgw.core.single_event.data.TimeSeries.read", return_value=series),
            patch(
                "jimgw.core.single_event.data.TimeSeries.fetch_open_data",
                return_value=series,
            ),
            pytest.raises(ValueError, match="delta_t|start_time"),
        ):
            if source == "gwosc":
                Data.from_gwosc("H1", 0.0, 4.0)
            else:
                Data.from_file(f"strain.{source}", channel="H1:GDS-CALIB_STRAIN")

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def test_from_file_npz_roundtrip(self, tmp_path: Path):
        """from_file with .npz produces Data with correct attributes."""
        td = np.ones(8192)
        dt = 1 / 2048.0
        start = 100.0
        path = str(tmp_path / "strain.npz")
        np.savez(path, td=td, dt=dt, start_time=start, name="H1")

        data = Data.from_file(path)

        assert data.name == "H1"
        assert data.start_time == pytest.approx(start)
        assert data.delta_t == pytest.approx(dt)
        assert len(data.td) == len(td)
        # A file with only the required keys gets the defaults.
        assert data.trigger_time == pytest.approx(start + 4.0 - 2.0)
        assert data.frequency_bounds == (0.0, jnp.inf)
        assert float(data.window[0]) == 0.0  # the default Tukey window

    def test_from_file_npz_missing_keys_raises(self, tmp_path: Path):
        """from_file raises ValueError for an .npz missing required keys."""
        path = str(tmp_path / "bad.npz")
        np.savez(path, td=np.zeros(4))  # missing 'dt' and 'start_time'

        with pytest.raises(ValueError, match="must contain"):
            Data.from_file(path)

    # ------------------------------------------------------------------
    # Unsupported extension
    # ------------------------------------------------------------------

    def test_from_file_unsupported_extension_raises(self, tmp_path: Path):
        """from_file raises ValueError for an unrecognised file extension."""
        path = str(tmp_path / "data.xyz")
        with pytest.raises(ValueError, match="Unsupported file format"):
            Data.from_file(path)

    # ------------------------------------------------------------------
    # GWF
    # ------------------------------------------------------------------

    def test_from_file_gwf_delegates_to_from_gwf(self):
        """from_file with .gwf extension calls _from_gwf with the right args."""
        sentinel = object()
        with patch.object(Data, "_from_gwf", return_value=sentinel) as mock_gwf:
            result = Data.from_file(
                "data.gwf",
                channel="H1:GDS-CALIB_STRAIN",
                start_time=10.0,
                end_time=14.0,
            )
        mock_gwf.assert_called_once_with(
            "data.gwf",
            channel="H1:GDS-CALIB_STRAIN",
            start_time=10.0,
            end_time=14.0,
        )
        assert result is sentinel

    def test_from_gwf_explicit_channel(self):
        """_from_gwf with an explicit channel calls TimeSeries.read correctly."""
        mock_ts = _make_mock_timeseries()
        with patch(
            "jimgw.core.single_event.data.TimeSeries.read", return_value=mock_ts
        ) as mock_read:
            data = Data._from_gwf("strain.gwf", channel="H1:GDS-CALIB_STRAIN")

        mock_read.assert_called_once_with(
            source="strain.gwf", channel="H1:GDS-CALIB_STRAIN"
        )
        assert data.name == "H1"
        assert data.delta_t == pytest.approx(mock_ts.dt.value)

    def test_from_gwf_explicit_channel_with_time_bounds(self):
        """_from_gwf passes start/end to TimeSeries.read when provided."""
        mock_ts = _make_mock_timeseries()
        with patch(
            "jimgw.core.single_event.data.TimeSeries.read", return_value=mock_ts
        ) as mock_read:
            Data._from_gwf(
                "strain.gwf",
                channel="L1:GDS-CALIB_STRAIN",
                start_time=0.0,
                end_time=4.0,
            )

        mock_read.assert_called_once_with(
            source="strain.gwf",
            channel="L1:GDS-CALIB_STRAIN",
            start=0.0,
            end=4.0,
        )

    def test_from_gwf_explicit_channel_not_found_raises(self):
        """_from_gwf re-raises as ValueError when the named channel is missing."""
        with (
            patch(
                "jimgw.core.single_event.data.TimeSeries.read",
                side_effect=RuntimeError("no channel"),
            ),
            pytest.raises(ValueError, match="Could not read channel"),
        ):
            Data._from_gwf("strain.gwf", channel="H1:BAD_CHANNEL")

    def test_from_gwf_auto_channel_fallback(self):
        """_from_gwf tries presets and succeeds on a later candidate."""
        mock_ts = _make_mock_timeseries()

        tried = []

        def _side_effect(**kwargs):
            tried.append(kwargs["channel"])
            # Fail on the first two attempts, succeed on the third
            if len(tried) < 3:
                raise RuntimeError("channel not found")
            return mock_ts

        with patch(
            "jimgw.core.single_event.data.TimeSeries.read",
            side_effect=_side_effect,
        ):
            data = Data._from_gwf("strain.gwf")

        assert len(tried) == 3
        assert len(set(tried)) == 3
        assert data.name == tried[-1].split(":")[0]

    def test_from_gwf_auto_channel_fallback_moves_on_to_the_next_detector(self):
        """When no channel of the first detector exists, the next detector is tried."""
        mock_ts = _make_mock_timeseries()

        def _side_effect(**kwargs):
            if kwargs["channel"].startswith("H1:"):
                raise RuntimeError("channel not found")
            return mock_ts

        with patch(
            "jimgw.core.single_event.data.TimeSeries.read",
            side_effect=_side_effect,
        ):
            data = Data._from_gwf("strain.gwf")

        assert data.name == "L1"

    def test_from_gwf_no_channel_all_fail_raises(self):
        """_from_gwf raises ValueError when no preset channel works."""
        with (
            patch(
                "jimgw.core.single_event.data.TimeSeries.read",
                side_effect=RuntimeError("channel not found"),
            ),
            pytest.raises(ValueError, match="Could not load any data"),
        ):
            Data._from_gwf("strain.gwf")

    # ------------------------------------------------------------------
    # HDF5 / CSV
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("ext", "expected"),
        [
            (".hdf5", {"path": "H1:GDS-CALIB_STRAIN"}),
            (".h5", {"path": "H1:GDS-CALIB_STRAIN"}),
            (".hdf", {"path": "H1:GDS-CALIB_STRAIN"}),
            (".csv", {}),
        ],
    )
    def test_from_file_gwpy_formats(self, ext: str, expected: dict):
        """In HDF5 the channel names the dataset; a CSV file only takes its name."""
        mock_ts = _make_mock_timeseries()
        with patch(
            "jimgw.core.single_event.data.TimeSeries.read", return_value=mock_ts
        ) as mock_read:
            data = Data.from_file(
                f"data{ext}", channel="H1:GDS-CALIB_STRAIN", start_time=0.0
            )

        mock_read.assert_called_once_with(source=f"data{ext}", start=0.0, **expected)
        assert data.name == "H1"

    @pytest.mark.parametrize("ext", [".hdf5", ".h5", ".hdf", ".csv"])
    def test_from_file_gwpy_no_channel(self, ext: str):
        """from_file for HDF5/CSV passes no channel kwarg when channel is None."""
        mock_ts = _make_mock_timeseries()
        with patch(
            "jimgw.core.single_event.data.TimeSeries.read", return_value=mock_ts
        ) as mock_read:
            data = Data.from_file(f"data{ext}")

        mock_read.assert_called_once_with(source=f"data{ext}")
        assert data.name == ""


class TestDataToFile:
    """Tests for Data.to_file across all supported formats."""

    _N = 4096
    _DT = 1 / 2048.0
    _T0 = 1126259462.0
    _NAME = "H1"

    def _make_data(self) -> Data:
        td = np.random.default_rng(0).standard_normal(self._N)
        return Data.from_td(
            jnp.array(td), self._DT, start_time=self._T0, name=self._NAME
        )

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def test_npz_roundtrip(self, tmp_path: Path):
        """Data saved as .npz reloads to identical arrays."""
        path = str(tmp_path / "strain.npz")
        d = self._make_data()
        d.to_file(path)
        d2 = Data.from_file(path)
        assert jnp.allclose(d2.td, d.td)
        assert d2.delta_t == pytest.approx(self._DT)
        assert d2.start_time == pytest.approx(self._T0)
        assert d2.name == self._NAME

    def test_npz_roundtrip_keeps_the_window_trigger_time_and_frequency_range(
        self, tmp_path: Path
    ):
        """Reloading gives back the data that were saved, not a re-windowed copy."""
        path = str(tmp_path / "strain.npz")
        d = self._make_data()
        d.set_tukey_window(roll_off=0.2)  # not the default
        d.trigger_time = self._T0 + 1.0
        d.frequency_bounds = (20.0, 500.0)
        d.to_file(path)
        d2 = Data.from_file(path)
        assert jnp.array_equal(d2.window, d.window)
        assert d2.trigger_time == d.trigger_time
        assert d2.frequency_bounds == d.frequency_bounds
        np.testing.assert_allclose(d2.fd, d.fd, rtol=1e-12)

    @pytest.mark.parametrize("n_time", [8, 9])
    def test_npz_roundtrip_keeps_materialized_frequency_data_exactly(
        self, tmp_path, n_time
    ):
        path = str(tmp_path / "injected.npz")
        data = Data.from_td(jnp.ones(n_time), 0.25, start_time=10.0)
        injected = data.add_fd(jnp.arange(data.n_freq) * (1 + 2j))
        injected.to_file(path)
        reloaded = Data.from_file(path)
        assert reloaded.has_fd
        assert jnp.array_equal(reloaded.fd, injected.fd)
        assert jnp.array_equal(reloaded.window, injected.window)
        assert reloaded.trigger_time == injected.trigger_time
        assert reloaded._trigger_time is None
        np.testing.assert_allclose(reloaded.fft(), reloaded.fd, atol=1e-14)
        with pytest.raises(RuntimeError, match="fixed"):
            reloaded.set_tukey_window(alpha=0.5)

    def test_saving_lazy_data_keeps_the_fft_lazy_and_trigger_unset(self, tmp_path):
        path = str(tmp_path / "lazy.npz")
        data = self._make_data()
        data.to_file(path)
        assert not data.has_fd
        reloaded = Data.from_file(path)
        assert not reloaded.has_fd
        assert reloaded._trigger_time is None

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_npz_additive_roundtrip_respects_stored_precision(self, tmp_path, dtype):
        path = str(tmp_path / "sum.npz")
        sample = jnp.arange(2048, dtype=dtype)
        td = jnp.sin(sample * 0.1)
        data = Data.from_td(td, 1 / 512, window=jnp.ones_like(td))
        injected = data.add_fd(jnp.fft.rfft(jnp.cos(sample * 0.14)) / 512)
        injected.to_file(path)
        reloaded = Data.from_file(path)
        assert reloaded.has_fd
        assert jnp.array_equal(reloaded.fd, injected.fd)
        assert jnp.array_equal(reloaded.td, injected.td)

        # A genuine change in cached strain must still fail the coherence check.
        with np.load(path) as saved:
            corrupted = dict(saved)
        amplitude = np.max(np.abs(corrupted["fd"]))
        corrupted["fd"][17] += 0.01 * amplitude
        np.savez(path, **corrupted)
        with pytest.raises(ValueError, match="Stored fd must agree"):
            Data.from_file(path)

    @pytest.mark.parametrize(
        "fd",
        [np.ones(4), np.full(5, np.nan), np.ones(5) * (1 + 1j), np.ones(5) * 2],
    )
    def test_inconsistent_cached_frequency_data_in_npz_are_rejected(self, tmp_path, fd):
        path = str(tmp_path / "invalid.npz")
        np.savez(
            path, td=np.ones(8), dt=0.25, start_time=10.0, window=np.ones(8), fd=fd
        )
        with pytest.raises(ValueError, match="fd|frequency_series"):
            Data.from_file(path)

    @pytest.mark.parametrize("huge_td", [False, True])
    def test_overflowing_complex_magnitude_cannot_bypass_npz_coherence(
        self, tmp_path, huge_td
    ):
        path = str(tmp_path / "incoherent.npz")
        if huge_td:
            td = np.array([0.7e308, -0.7e308, -0.7e308, 0.7e308])
            dt = 1.0
            fd = np.array([0, 0.7e308 + 0.7e308j, 0])
        else:
            td, dt = np.zeros(8), 0.25
            fd = np.zeros(5, dtype=complex)
            fd[2] = 1.4e308 + 1.4e308j
        assert np.all(np.isfinite(fd))
        np.savez(path, td=td, dt=dt, start_time=10.0, window=np.ones_like(td), fd=fd)
        with pytest.raises(ValueError, match="Stored fd must agree"):
            Data.from_file(path)

    @pytest.mark.parametrize("field", ["td", "window", "dt"])
    def test_corrupted_npz_fft_inputs_cannot_keep_a_stale_cache(self, tmp_path, field):
        path = str(tmp_path / "changed.npz")
        data = Data.from_td(jnp.ones(8), 0.25, window=jnp.ones(8))
        _ = data.fd
        data.to_file(path)
        with np.load(path) as saved:
            archive = dict(saved)
        archive[field] = archive[field] * 2
        np.savez(path, **archive)
        with pytest.raises(ValueError, match="Stored fd must agree"):
            Data.from_file(path)

    # ------------------------------------------------------------------
    # Text / dat
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ext", [".txt", ".dat"])
    def test_text_roundtrip(self, tmp_path: Path, ext: str):
        """Frequency-domain strain saved as text reloads with matching columns."""
        path = str(tmp_path / f"strain{ext}")
        d = self._make_data()
        d.to_file(path)
        loaded = np.loadtxt(path)
        f_expected = np.array(d.frequencies)
        fd_expected = np.array(d.fd)
        assert np.allclose(loaded[:, 0], f_expected)
        assert np.allclose(loaded[:, 1], fd_expected.real)
        assert np.allclose(loaded[:, 2], fd_expected.imag)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def test_csv_roundtrip(self, tmp_path: Path):
        """CSV file has comma delimiter and correct frequency-domain content."""
        path = str(tmp_path / "strain.csv")
        d = self._make_data()
        d.to_file(path)
        loaded = np.loadtxt(path, delimiter=",")
        assert np.allclose(loaded[:, 0], np.array(d.frequencies))
        assert np.allclose(loaded[:, 1], np.array(d.fd).real)

    # ------------------------------------------------------------------
    # GWF
    # ------------------------------------------------------------------

    def test_gwf_calls_timeseries_write(self, tmp_path: Path):
        """to_file for .gwf constructs a TimeSeries and calls write."""
        path = str(tmp_path / "strain.gwf")
        d = self._make_data()
        mock_ts = MagicMock()
        with patch(
            "jimgw.core.single_event.data.TimeSeries", return_value=mock_ts
        ) as mock_cls:
            d.to_file(path)
        mock_cls.assert_called_once()
        mock_ts.write.assert_called_once_with(path)

    # ------------------------------------------------------------------
    # HDF5
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ext", [".hdf5", ".h5"])
    def test_hdf5_roundtrip(self, tmp_path: Path, ext: str):
        """Data written as HDF5 reload through gwpy with the same samples and name."""
        path = str(tmp_path / f"strain{ext}")
        d = self._make_data()
        d.to_file(path)
        d2 = Data.from_file(path, channel="H1:STRAIN")
        np.testing.assert_array_equal(d2.td, d.td)
        assert d2.delta_t == d.delta_t
        assert d2.start_time == d.start_time
        assert d2.name == self._NAME
        # A file with a single dataset also loads without naming it.
        assert Data.from_file(path).name == ""

    # ------------------------------------------------------------------
    # Channel name
    # ------------------------------------------------------------------

    def test_gwf_channel_name_with_colon(self, tmp_path: Path):
        """If name contains ':', it is used as-is for the channel."""
        path = str(tmp_path / "s.gwf")
        d = Data.from_td(
            jnp.array(np.zeros(self._N)),
            self._DT,
            start_time=self._T0,
            name="H1:GDS-CALIB_STRAIN",
        )
        mock_ts = MagicMock()
        with patch(
            "jimgw.core.single_event.data.TimeSeries", return_value=mock_ts
        ) as mock_cls:
            d.to_file(path)
        _, kwargs = mock_cls.call_args
        assert kwargs["channel"] == "H1:GDS-CALIB_STRAIN"

    def test_gwf_channel_name_without_colon(self, tmp_path: Path):
        """If name has no ':', channel is set to '{name}:STRAIN'."""
        path = str(tmp_path / "s.gwf")
        d = self._make_data()
        mock_ts = MagicMock()
        with patch(
            "jimgw.core.single_event.data.TimeSeries", return_value=mock_ts
        ) as mock_cls:
            d.to_file(path)
        _, kwargs = mock_cls.call_args
        assert kwargs["channel"] == "H1:STRAIN"

    # ------------------------------------------------------------------
    # Unsupported
    # ------------------------------------------------------------------

    def test_unsupported_extension_raises(self, tmp_path: Path):
        path = str(tmp_path / "strain.xyz")
        d = self._make_data()
        with pytest.raises(ValueError, match="Unsupported file format"):
            d.to_file(path)
