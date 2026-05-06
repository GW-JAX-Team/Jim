from abc import ABC
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex, Key

from gwpy.timeseries import TimeSeries
from typing import Optional, Self
from scipy.signal import welch
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# TODO: Need to expand this list. Currently it is only O3.
asd_file_dict = {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",
}


class Data(ABC):
    """Base class for all data.

    The time domain data are considered the primary entity; the Fourier domain
    data are derived from an FFT after applying a window. The structure is set up
    so that td and fd are always Fourier conjugates of each other: the one-sided
    Fourier series is complete up to the Nyquist frequency.

    Attributes:
        name: Name of the data instance.
        td: Time domain data array.
        fd: Frequency domain data array.
        start_time: GPS start time of the data segment in seconds.
        delta_t: Time step between samples.
        window: Window function applied to data.
    """

    name: str

    td: Float[Array, "n_time"]
    fd: Complex[Array, "n_time // 2 + 1"]

    start_time: Float
    delta_t: Float

    window: Float[Array, "n_time"]

    def __len__(self) -> int:
        """Returns the length of the time-domain data.

        Returns:
            int: Length of time domain data array.
        """
        return len(self.td)

    def __iter__(self):
        """Iterator over the time-domain data.

        Returns:
            iterator: Iterator over time domain data.
        """
        return iter(self.td)

    @property
    def n_time(self) -> int:
        """Gets number of time samples.

        Returns:
            int: Number of time domain samples.
        """
        return len(self.td)

    @property
    def n_freq(self) -> int:
        """Gets number of frequency samples.

        Returns:
            int: Number of frequency domain samples.
        """
        return self.n_time // 2 + 1

    @property
    def is_empty(self) -> bool:
        """Checks if the data is empty.

        Returns:
            bool: True if data is empty, False otherwise.
        """
        return self.n_time == 0

    @property
    def duration(self) -> float:
        """Gets duration of the data in seconds.

        Returns:
            float: Duration in seconds.
        """
        return self.n_time * self.delta_t

    @property
    def sampling_frequency(self) -> float:
        """Gets sampling frequency of the data.

        Returns:
            float: Sampling frequency in Hz.
        """
        return 1 / self.delta_t

    @property
    def times(self) -> Float[Array, "n_time"]:
        """Gets time points of the data.

        Returns:
            Array: Array of time points in seconds.
        """
        return jnp.arange(self.n_time) * self.delta_t + self.start_time

    @property
    def frequencies(self) -> Float[Array, "n_time // 2 + 1"]:
        """Gets frequencies of the data.

        Returns:
            Array: Array of frequencies in Hz.
        """
        return jnp.fft.rfftfreq(self.n_time, self.delta_t)

    @property
    def has_fd(self) -> bool:
        """Checks if Fourier domain data exists.

        Returns:
            bool: True if Fourier domain data exists, False otherwise.
        """
        return bool(jnp.any(self.fd))

    def __init__(
        self,
        td: Float[Array, "n_time"] = jnp.array([]),
        delta_t: Float = 0.0,
        start_time: Float = 0.0,
        name: str = "",
        window: Optional[Float[Array, "n_time"]] = None,
    ) -> None:
        """Initialize the data class.

        Args:
            td: Time domain data array.
            delta_t: Time step of the data in seconds.
            start_time: GPS start time of the segment in seconds (default: 0).
            name: Name of the data (default: '').
            window: Window function to apply to the data before FFT (default: None).
        """
        self.name = name or ""
        self.td = td
        self.fd = jnp.zeros(self.n_freq, dtype="complex128")
        self.delta_t = delta_t
        self.start_time = start_time
        if window is None:
            self.set_tukey_window()
        else:
            self.window = window

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            + f"delta_t={self.delta_t}, start_time={self.start_time})"
        )

    def __bool__(self) -> bool:
        """Check if the data is empty."""
        return len(self.td) > 0

    def set_tukey_window(self, alpha: float = 0.2) -> None:
        """Create a Tukey window on the data; the window is stored in the
        window attribute and only applied when FFTing the data.

        Args:
            alpha: Shape parameter of the Tukey window (default: 0.2); this is
                the fraction of the segment that is tapered on each side.
        """
        logger.info(f"Setting Tukey window to {self.name} data")
        self.window = jnp.array(tukey(self.n_time, alpha))

    def fft(
        self, window: Optional[Float[Array, "n_time"]] = None
    ) -> Complex[Array, "n_freq"]:
        """Compute the Fourier transform of the data and store it
        in the fd attribute.

        Args:
            window: Window function to apply to the data before FFT (default: None).
        """
        if self.n_time > 0:
            assert self.delta_t > 0, "Delta t must be positive"
        if self.has_fd and (window is None or window == self.window):
            # Perhaps one needs to also check self.td and self.delta_t are the same.
            logger.debug(f"{self.name} has FD data, skipping FFT.")
            return self.fd
        if window is None:
            window = self.window

        logger.info(f"Computing FFT of {self.name} data")
        self.fd = jnp.fft.rfft(self.td * window) * self.delta_t
        self.window = window
        return self.fd

    def frequency_slice(
        self, f_min: Float, f_max: Float, auto_fft: bool = True
    ) -> tuple[Complex[Array, " n_sample"], Float[Array, " n_sample"]]:
        """Slice the data in the frequency domain.
        This is the main function which interacts with the likelihood.

        Args:
            f_min: Minimum frequency of the slice in Hz.
            f_max: Maximum frequency of the slice in Hz.
            auto_fft: Whether to automatically compute FFT if not already done.

        Returns:
            tuple: Sliced data in the frequency domain and corresponding frequencies.
        """
        if auto_fft:
            self.fft()
        mask = (self.frequencies >= f_min) * (self.frequencies <= f_max)
        return self.fd[mask], self.frequencies[mask]

    def to_psd(self, **kws) -> "PowerSpectrum":
        """Compute a Welch estimate of the power spectral density of the data.

        Args:
            **kws: Keyword arguments for `scipy.signal.welch`.

        Returns:
            PowerSpectrum: Power spectral density of the data.
        """
        if not self.has_fd:
            self.fft()
        freq, psd = welch(self.td, fs=self.sampling_frequency, **kws)
        return PowerSpectrum(jnp.asarray(psd), jnp.asarray(freq), self.name)

    @classmethod
    def from_gwosc(
        cls,
        ifo: str,
        gps_start_time: Float,
        gps_end_time: Float,
        cache: bool = True,
        **kws,
    ) -> Self:
        """Pull data from GWOSC.

        Args:
            ifo: Interferometer name.
            gps_start_time: GPS start time of the data.
            gps_end_time: GPS end time of the data.
            cache: Whether to cache the data (default: True).
            **kws: Keyword arguments for `gwpy.timeseries.TimeSeries.fetch_open_data`.

        Returns:
            Data: Data object with the fetched time domain data.
        """
        duration = gps_end_time - gps_start_time
        logger.info(
            f"Fetching {duration} s of {ifo} data from GWOSC "
            f"[{gps_start_time}, {gps_end_time}]"
        )

        data_td = TimeSeries.fetch_open_data(
            ifo, gps_start_time, gps_end_time, cache=cache, **kws
        )
        return cls(data_td.value, data_td.dt.value, data_td.epoch.value, ifo)  # type: ignore[union-attr]

    @classmethod
    def from_fd(
        cls,
        fd_strain: Complex[Array, "n_freq"],
        frequencies: Float[Array, "n_freq"],
        start_time: float = 0.0,
        name: str = "",
    ) -> Self:
        """Create a Data object starting from (potentially incomplete)
        Fourier domain data.

        Args:
            fd_strain: Fourier domain data array.
            frequencies: Frequencies of the data in Hz.
            start_time: GPS start time of the segment in seconds (default: 0).
            name: Name of the data (default: '').

        Returns:
            Data: Data object with the Fourier and time domain data.
        """
        assert fd_strain.shape == frequencies.shape, (
            "Frequency and data arrays must have the same length"
        )
        f_nyq = frequencies[-1]
        sampling_rate = 2 * f_nyq
        duration = 1 / (frequencies[1] - frequencies[0])
        n_samples = int(jnp.round(sampling_rate * duration))

        # Ensure time-domain samples will be even
        if (n_samples % 2) != 0:
            raise ValueError(
                "The number of time-domain samples will not be even. "
                + "Please check your frequency array."
            )

        # Construct the full frequency array
        n_frequencies = int(jnp.round(n_samples / 2) + 1)
        freqs = jnp.arange(n_frequencies) / duration
        # Fill in the full data array
        start_idx = jnp.searchsorted(freqs, frequencies[0])
        data_fd_full = jax.lax.dynamic_update_slice(
            jnp.zeros_like(freqs, dtype=fd_strain.dtype), fd_strain, (start_idx,)
        )
        # IFFT into time domain
        delta_t = 1 / sampling_rate
        data_td_full = jnp.fft.irfft(data_fd_full) / delta_t
        # Check frequencies
        assert jnp.array_equal(freqs, jnp.fft.rfftfreq(len(data_td_full), delta_t)), (
            "Generated frequencies do not match the input frequencies"
        )
        # Create a Data object
        data = cls(data_td_full, delta_t, start_time=start_time, name=name)
        data.fd = data_fd_full

        # Ensures the newly constructed Data in FD faithfully
        # represents the input FD data.
        d_new, f_new = data.frequency_slice(frequencies[0], frequencies[-1])
        assert jnp.array_equal(d_new, fd_strain), (
            "Fourier domain data do not match after slicing"
        )
        assert jnp.array_equal(f_new, frequencies), (
            "Frequencies do not match after slicing"
        )
        return data

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Load data from a file. This assumes the data to be in .npz format.
        It should at least contain the keys 'td', 'dt', and 'start_time' (GPS start time).

        Args:
            path (str): Path to the .npz file containing the data.
        """
        with np.load(path) as data:
            if "td" not in data or "dt" not in data or "start_time" not in data:
                raise ValueError(
                    "The file must contain 'td', 'dt', and 'start_time' keys."
                )
            td = jnp.array(data["td"])
            dt = float(data["dt"])
            start_time = float(data["start_time"])
            name = str(data.get("name", ""))
        return cls(td, dt, start_time, name)

    def to_file(self, path: str):
        """Save the data to a file in .npz format.

        Args:
            path (str): Path to save the .npz file.
        """
        jnp.savez(
            path,
            td=self.td,
            dt=self.delta_t,
            start_time=self.start_time,
            name=self.name,
        )


class PowerSpectrum(ABC):
    """Class representing a power spectral density.

    Attributes:
        name: Name of the power spectrum.
        values: Array of PSD values.
        frequencies: Array of frequencies corresponding to PSD values.
    """

    name: str
    values: Float[Array, "n_freq"]
    frequencies: Float[Array, "n_freq"]

    @property
    def n_freq(self) -> int:
        """Gets number of frequency samples.

        Returns:
            int: Number of frequency samples.
        """
        return len(self.values)

    @property
    def is_empty(self) -> bool:
        """Checks if the data is empty.

        Returns:
            bool: True if data is empty, False otherwise.
        """
        return self.n_freq == 0

    @property
    def delta_f(self) -> Float:
        """Gets frequency resolution.

        Returns:
            float: Frequency resolution in Hz.
        """
        return self.frequencies[1] - self.frequencies[0]

    @property
    def delta_t(self) -> Float:
        """Gets time resolution.

        Returns:
            float: Time resolution in seconds.
        """
        return 1 / self.sampling_frequency

    @property
    def duration(self) -> Float:
        """Gets duration of the data.

        Returns:
            float: Duration in seconds.
        """
        return 1 / self.delta_f

    @property
    def sampling_frequency(self) -> Float:
        """Gets sampling frequency.

        Returns:
            float: Sampling frequency in Hz.
        """
        return self.frequencies[-1] * 2

    def __init__(
        self,
        values: Float[Array, "n_freq"] = jnp.array([]),
        frequencies: Float[Array, "n_freq"] = jnp.array([]),
        name: Optional[str] = None,
    ) -> None:
        """Initialize PowerSpectrum.

        Args:
            values: Array of PSD values. Defaults to empty array.
            frequencies: Array of frequencies in Hz. Defaults to empty array.
            name: Name of the power spectrum. Defaults to None.
        """
        # NOTE: Are we sure the values and frequencies start from 0?
        self.values = values
        self.frequencies = frequencies
        assert self.n_freq == len(self.frequencies), (
            "Values and frequencies must have the same length"
        )
        self.name = name or ""

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            + f"frequencies={self.frequencies})"
        )

    def __bool__(self) -> bool:
        """Check if the power spectrum is empty.

        Returns:
            bool: True if power spectrum contains data, False otherwise.
        """
        return self.n_freq > 0

    def frequency_slice(
        self, f_min: float, f_max: float
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """Slice the power spectrum to a frequency range.

        Args:
            f_min: Minimum frequency of the slice in Hz.
            f_max: Maximum frequency of the slice in Hz.

        Returns:
            tuple: Contains:
                - values: Sliced PSD values
                - frequencies: Frequencies corresponding to sliced values
        """
        mask = (self.frequencies >= f_min) & (self.frequencies <= f_max)
        return self.values[mask], self.frequencies[mask]

    def interpolate(
        self, frequencies: Float[Array, " n_sample"], kind: str = "linear", **kws
    ) -> "PowerSpectrum":
        """Interpolate the power spectrum to new frequencies.

        Args:
            f: Frequencies to interpolate to.
            kind: Interpolation method. Defaults to 'linear'.
            **kws: Additional keyword arguments for scipy.interpolate.interp1d.

        Returns:
            PowerSpectrum: New power spectrum with interpolated values.
        """
        interp = interp1d(
            self.frequencies,
            self.values,
            kind=kind,
            fill_value=(self.values[0], self.values[-1]),  # type: ignore[arg-type]  # scipy stubs
            bounds_error=False,
            **kws,
        )
        return PowerSpectrum(interp(frequencies), frequencies, self.name)

    def simulate_data(
        self,
        key: Key,
    ) -> Complex[Array, " n_sample"]:
        """Simulate noise data based on the power spectrum.

        Args:
            key: JAX PRNG key for random number generation.

        Returns:
            Complex frequency series of simulated noise data.
        """
        key, subkey = jax.random.split(key, 2)
        var = self.values / (4 * self.delta_f)
        noise_real = jax.random.normal(key, shape=var.shape) * jnp.sqrt(var)
        noise_imag = jax.random.normal(subkey, shape=var.shape) * jnp.sqrt(var)
        return noise_real + 1j * noise_imag

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Load power spectrum from a file. This assumes the data to be in .npz format.
        It should at least contains the keys 'values', 'frequencies', and 'name'.
        `values` is the PSD values, `frequencies` is the frequencies of the PSD.

        Args:
            path (str): Path to the .npz file containing the data.
        """
        with np.load(path) as data:
            if "values" not in data or "frequencies" not in data:
                raise ValueError(
                    "The file must contain 'values' and 'frequencies' keys."
                )
            values = jnp.array(data["values"])
            frequencies = jnp.array(data["frequencies"])
            name = str(data.get("name", ""))
        return cls(values, frequencies, name)

    def to_file(self, path: str):
        """Save the power spectrum to a file in .npz format.

        Args:
            path (str): Path to save the .npz file.
        """
        jnp.savez(
            path,
            values=self.values,
            frequencies=self.frequencies,
            name=self.name,
        )
