import jax

import jax.numpy as jnp
from copy import deepcopy
from scipy.signal import welch
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
        self.data = Data(
            td=jnp.ones(n_time),
            delta_t=delta_t,
            name=self.name,
            start_time=self.start_time,
        )

    def test_basic_attributes(self):
        """Basic attributes are set correctly on construction."""
        assert self.data.name == "Dummy"
        assert self.data.start_time == self.start_time
        assert self.data.duration == self.duration
        assert self.data.delta_t == 1 / self.f_samp
        assert len(self.data.td) == int(self.f_samp * self.duration)

    def test_default_window(self):
        """A Tukey window matching the data length is created by default."""
        assert len(self.data.window) == len(self.data.td)

    def test_bool_nonempty(self):
        """bool(data) is True when data are present."""
        assert bool(self.data)

    def test_fd_initially_zero(self):
        """FD array starts as zeros but has the correct length."""
        assert not self.data.has_fd
        assert jnp.all(self.data.fd == 0)
        fftfreq = jnp.fft.rfftfreq(len(self.data.td), self.data.delta_t)
        assert len(self.data.fd) == len(fftfreq)
        assert self.data.n_freq == len(fftfreq)

    def test_frequency_slice_triggers_fft(self):
        """Calling frequency_slice computes and caches the FFT."""
        fftfreq = jnp.fft.rfftfreq(len(self.data.td), self.data.delta_t)
        expected_fd = jnp.fft.rfft(self.data.td * self.data.window) * self.data.delta_t

        fmin, fmax = 20, 512
        data_slice, freq_slice = self.data.frequency_slice(fmin, fmax)

        freq_mask = (fftfreq >= fmin) & (fftfreq <= fmax)
        assert jnp.allclose(self.data.fd, expected_fd)
        assert jnp.allclose(data_slice, expected_fd[freq_mask])
        assert jnp.allclose(freq_slice, fftfreq[freq_mask])

    def test_explicit_fft_matches_frequency_slice(self):
        """Explicitly calling fft() gives the same result as frequency_slice."""
        expected_fd = jnp.fft.rfft(self.data.td * self.data.window) * self.data.delta_t

        data_copy = deepcopy(self.data)
        assert not data_copy.has_fd
        data_copy.fft()
        assert jnp.allclose(data_copy.fd, expected_fd)

        fmin, fmax = 20, 512
        slice_via_slice, freq_via_slice = self.data.frequency_slice(fmin, fmax)
        slice_via_fft, freq_via_fft = data_copy.frequency_slice(fmin, fmax)
        assert jnp.allclose(slice_via_slice, slice_via_fft)
        assert jnp.allclose(freq_via_slice, freq_via_fft)


class TestPowerSpectrum:
    """Tests for the PowerSpectrum class."""

    def setup_method(self):
        self.f_samp = 2048
        self.duration = 4
        self.name = "Dummy"
        delta_t = 1 / self.f_samp
        n_time = int(self.duration / delta_t)
        self.data = Data(
            td=jnp.ones(n_time), delta_t=delta_t, name=self.name, start_time=0.0
        )

        delta_f = 1 / self.duration
        self.psd_band = (20, 512)
        psd_min, psd_max = self.psd_band
        freqs = jnp.arange(int(psd_max / delta_f)) * delta_f
        freqs_psd = freqs[freqs >= psd_min]
        self.psd = PowerSpectrum(
            jnp.ones_like(freqs_psd), frequencies=freqs_psd, name=self.name
        )

    def test_basic_attributes(self):
        """Basic attributes are set correctly on construction."""
        assert self.psd.name == "Dummy"
        assert self.psd.n_freq == len(self.psd.frequencies)
        assert jnp.all(self.psd.frequencies >= self.psd_band[0])
        assert jnp.all(self.psd.frequencies <= self.psd_band[1])

    def test_frequency_slice(self):
        """Slicing the PSD to its own band returns the full array."""
        sliced_psd, freq_slice = self.psd.frequency_slice(*self.psd_band)
        assert jnp.allclose(sliced_psd, self.psd.values)
        assert jnp.allclose(freq_slice, self.psd.frequencies)

    def test_welch_psd_from_data(self):
        """PSD estimated from data via Welch's method matches scipy."""
        nperseg = self.data.n_time // 2
        psd_auto = self.data.to_psd(nperseg=nperseg)
        freq_manual, psd_manual = welch(self.data.td, fs=self.f_samp, nperseg=nperseg)
        assert jnp.allclose(psd_auto.frequencies, freq_manual)
        assert jnp.allclose(psd_auto.values, psd_manual)

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
