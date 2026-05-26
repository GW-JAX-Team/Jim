import jax

import jax.numpy as jnp
import numpy as np
import pytest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch
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


class TestPowerSpectrumFromFile:
    """Tests for PowerSpectrum.from_file across all supported formats."""

    _FREQS = np.array([10.0, 20.0, 30.0])
    _PSD = np.array([1e-46, 4e-46, 9e-46])
    _ASD = np.sqrt(_PSD)

    # -- NPZ ------------------------------------------------------------------

    def test_npz_roundtrip(self, tmp_path: Path):
        """.npz file is loaded with the correct values and frequencies."""
        path = str(tmp_path / "psd.npz")
        np.savez(path, values=self._PSD, frequencies=self._FREQS, name="H1")
        psd = PowerSpectrum.from_file(path)
        assert jnp.allclose(psd.values, jnp.array(self._PSD))
        assert jnp.allclose(psd.frequencies, jnp.array(self._FREQS))
        assert psd.name == "H1"

    def test_npz_missing_keys_raises(self, tmp_path: Path):
        path = str(tmp_path / "bad.npz")
        np.savez(path, values=self._PSD)  # missing 'frequencies'
        with pytest.raises(ValueError, match="must contain"):
            PowerSpectrum.from_file(path)

    # -- TXT / DAT -----------------------------------------------------------

    @pytest.mark.parametrize("ext", [".txt", ".dat"])
    def test_text_psd_file(self, tmp_path: Path, ext: str):
        """Two-column text file loads correctly as PSD (is_asd=False)."""
        path = str(tmp_path / f"psd{ext}")
        np.savetxt(path, np.column_stack([self._FREQS, self._PSD]))
        psd = PowerSpectrum.from_file(path, is_asd=False)
        assert jnp.allclose(psd.values, jnp.array(self._PSD))
        assert jnp.allclose(psd.frequencies, jnp.array(self._FREQS))

    @pytest.mark.parametrize("ext", [".txt", ".dat"])
    def test_text_asd_file_squared(self, tmp_path: Path, ext: str):
        """is_asd=True squares the loaded values to give a PSD."""
        path = str(tmp_path / f"asd{ext}")
        np.savetxt(path, np.column_stack([self._FREQS, self._ASD]))
        psd = PowerSpectrum.from_file(path, is_asd=True)
        assert jnp.allclose(psd.values, jnp.array(self._PSD), rtol=1e-6)

    # -- CSV ------------------------------------------------------------------

    def test_csv_psd_file(self, tmp_path: Path):
        """Comma-separated two-column CSV loads as PSD."""
        path = str(tmp_path / "psd.csv")
        np.savetxt(path, np.column_stack([self._FREQS, self._PSD]), delimiter=",")
        psd = PowerSpectrum.from_file(path)
        assert jnp.allclose(psd.values, jnp.array(self._PSD))

    def test_csv_asd_file_squared(self, tmp_path: Path):
        """CSV with ASD values is squared correctly when is_asd=True."""
        path = str(tmp_path / "asd.csv")
        np.savetxt(path, np.column_stack([self._FREQS, self._ASD]), delimiter=",")
        psd = PowerSpectrum.from_file(path, is_asd=True)
        assert jnp.allclose(psd.values, jnp.array(self._PSD), rtol=1e-6)

    # -- Unsupported ----------------------------------------------------------

    def test_unsupported_extension_raises(self, tmp_path: Path):
        path = str(tmp_path / "psd.xyz")
        with pytest.raises(ValueError, match="Unsupported file format"):
            PowerSpectrum.from_file(path)


# ---------------------------------------------------------------------------
# Helpers shared across file-loading tests
# ---------------------------------------------------------------------------


def _make_mock_timeseries(n=8192, dt=1 / 2048.0, epoch=0.0):
    """Return a MagicMock that looks like a gwpy TimeSeries."""
    ts = MagicMock()
    ts.value = np.zeros(n)
    ts.dt.value = dt
    ts.epoch.value = epoch
    return ts


# ---------------------------------------------------------------------------


class TestDataFromFile:
    """Tests for Data.from_file and Data._from_gwf."""

    # -- NPZ ------------------------------------------------------------------

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

    def test_from_file_npz_missing_keys_raises(self, tmp_path: Path):
        """from_file raises ValueError for an .npz missing required keys."""
        path = str(tmp_path / "bad.npz")
        np.savez(path, td=np.zeros(4))  # missing 'dt' and 'start_time'

        with pytest.raises(ValueError, match="must contain"):
            Data.from_file(path)

    # -- Unsupported extension ------------------------------------------------

    def test_from_file_unsupported_extension_raises(self, tmp_path: Path):
        """from_file raises ValueError for an unrecognised file extension."""
        path = str(tmp_path / "data.xyz")
        with pytest.raises(ValueError, match="Unsupported file format"):
            Data.from_file(path)

    # -- GWF ------------------------------------------------------------------

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
        with patch(
            "jimgw.core.single_event.data.TimeSeries.read",
            side_effect=RuntimeError("no channel"),
        ):
            with pytest.raises(ValueError, match="Could not read channel"):
                Data._from_gwf("strain.gwf", channel="H1:BAD_CHANNEL")

    def test_from_gwf_auto_channel_fallback(self):
        """_from_gwf tries presets and succeeds on a later candidate."""
        mock_ts = _make_mock_timeseries()

        call_count = 0

        def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on the first two attempts, succeed on the third
            if call_count < 3:
                raise RuntimeError("channel not found")
            return mock_ts

        with patch(
            "jimgw.core.single_event.data.TimeSeries.read",
            side_effect=_side_effect,
        ):
            data = Data._from_gwf("strain.gwf")

        assert call_count == 3
        assert data is not None

    def test_from_gwf_no_channel_all_fail_raises(self):
        """_from_gwf raises ValueError when no preset channel works."""
        with patch(
            "jimgw.core.single_event.data.TimeSeries.read",
            side_effect=RuntimeError("channel not found"),
        ):
            with pytest.raises(ValueError, match="Could not load any data"):
                Data._from_gwf("strain.gwf")

    # -- HDF5 / CSV -----------------------------------------------------------

    @pytest.mark.parametrize("ext", [".hdf5", ".h5", ".hdf", ".csv"])
    def test_from_file_gwpy_formats(self, ext: str):
        """from_file for HDF5/CSV formats calls TimeSeries.read."""
        mock_ts = _make_mock_timeseries()
        with patch(
            "jimgw.core.single_event.data.TimeSeries.read", return_value=mock_ts
        ) as mock_read:
            data = Data.from_file(
                f"data{ext}", channel="H1:GDS-CALIB_STRAIN", start_time=0.0
            )

        mock_read.assert_called_once_with(
            source=f"data{ext}",
            channel="H1:GDS-CALIB_STRAIN",
            start=0.0,
        )
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
