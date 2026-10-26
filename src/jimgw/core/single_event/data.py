import logging
from abc import ABC
from typing import Any, Never, Optional, Self

import jax
import jax.numpy as jnp
import numpy as np
from gwpy.timeseries import TimeSeries
from jaxtyping import Array, Bool, Complex, Float, Key
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.signal.windows import tukey

from jimgw.core.single_event.mixins import _Freezable
from jimgw.typing import FloatLike, FloatScalar

logger = logging.getLogger(__name__)

DEFAULT_TUKEY_ROLL_OFF = 0.4
DEFAULT_POST_TRIGGER_DURATION = 2.0
_EMPTY_ARRAY = jnp.array([])


class Data(_Freezable, ABC):
    """Base class for all data.

    The time domain data are considered the primary entity; the frequency domain
    data are derived from an FFT after applying a window. The structure is set up
    so that td and fd are always Fourier conjugates of each other: the one-sided
    Fourier series is complete up to the Nyquist frequency.

    Attributes:
        name: Name of the data instance.
        td: Time domain data array.
        fd: Frequency domain data array.
        start_time: GPS start time of the data segment in seconds.
        end_time: Exclusive GPS end of the actual samples, derived from their start,
            count and time step.
        trigger_time: GPS time of the event trigger in seconds, the reference time
            of ``t_c``. Defaults to 2 s before the end of the data.
        delta_t: Time step between samples.
        window: Window function applied to data.
    """

    name: str

    td: Float[Array, "n_time"]
    _fd: Optional[Complex[Array, "n_time // 2 + 1"]]

    _start_time: float
    _trigger_time: Optional[float]
    delta_t: FloatLike

    window: Float[Array, "n_time"]
    _frequency_bounds: tuple[float, float]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, *args: Never, **kwargs: Never) -> None:
        """Initialize empty data, the placeholder of a detector without data.

        It takes no arguments: create data with ``from_td``, ``from_fd``,
        ``from_gwosc`` or ``from_file``.

        Raises:
            TypeError: If any argument is given.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if len(args) + len(kwargs) > 0:
            raise TypeError(
                "Data() takes no arguments and is empty. Create data with "
                "Data.from_td(), Data.from_fd(), Data.from_gwosc() or "
                "Data.from_file()."
            )
        self.name = ""
        self.td = _EMPTY_ARRAY
        self._fd = None
        self.delta_t = 0.0
        self.start_time = 0.0
        self._trigger_time = None
        self.window = jnp.array([])
        self._frequency_bounds = (0.0, jnp.inf)

    @classmethod
    def from_td(
        cls,
        td: Float[Array, "n_time"],
        delta_t: FloatLike,
        *,
        start_time: float = 0.0,
        trigger_time: Optional[float] = None,
        name: str = "",
        window: Optional[Float[Array, "n_time"]] = None,
    ) -> Self:
        """Create data from time domain samples.

        Args:
            td: Time domain data array.
            delta_t: Time step of the data in seconds.
            start_time: GPS start time of the segment in seconds (default: 0).
            trigger_time: GPS time of the event trigger in seconds (default: 2 s
                before the end of the data).
            name: Name of the data (default: '').
            window: Window function to apply to the data before FFT (default: None).

        Returns:
            Data: Data object with the time domain data.

        Raises:
            ValueError: If samples or the window are not one-dimensional, real
                and finite, if the window shape differs from the data, if timing
                metadata are not finite scalars, or if non-empty data have a
                non-positive time step.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        data = cls()
        data.name = name or ""
        data.td = jnp.asarray(td)
        if (
            data.td.ndim != 1
            or jnp.iscomplexobj(data.td)
            or not jnp.all(jnp.isfinite(data.td))
        ):
            raise ValueError("td must be a one-dimensional real finite array")
        time_step = np.asarray(delta_t)
        if (
            time_step.ndim != 0
            or not np.issubdtype(time_step.dtype, np.number)
            or np.iscomplexobj(time_step)
            or not np.isfinite(float(time_step))
            or float(time_step) < 0
            or (float(time_step) == 0 and not data.is_empty)
        ):
            raise ValueError(
                "delta_t must be finite and positive for non-empty data; "
                "empty data allow zero"
            )
        data.delta_t = float(time_step)
        data.start_time = start_time
        data.trigger_time = trigger_time
        if window is None:
            if not data.is_empty:
                data.set_tukey_window()
        else:
            data.window = jnp.asarray(window)
            if (
                data.window.shape != data.td.shape
                or jnp.iscomplexobj(data.window)
                or not jnp.all(jnp.isfinite(data.window))
            ):
                raise ValueError(
                    "window must be a real finite array with the same shape as td"
                )
        return data

    @classmethod
    def from_fd(
        cls,
        fd: Complex[Array, "n_freq"],
        frequencies: Float[Array, "n_freq"],
        *,
        start_time: float = 0.0,
        trigger_time: Optional[float] = None,
        name: str = "",
    ) -> Self:
        """Create a Data object starting from (potentially incomplete)
        frequency domain data. The frequencies must be consecutive points of a
        real-FFT grid, ending at the Nyquist frequency of an even-length segment.
        Imaginary DC and Nyquist components are discarded because real samples
        cannot represent them.

        Args:
            fd: Frequency domain data array.
            frequencies: Frequencies of the data in Hz.
            start_time: GPS start time of the segment in seconds (default: 0).
            trigger_time: GPS time of the event trigger in seconds (default: 2 s
                before the end of the data).
            name: Name of the data (default: '').

        Returns:
            Data: Data object with the frequency and time domain data.

        Raises:
            ValueError: If arrays differ in shape, contain non-finite values, or
                frequencies are not at least two nonnegative, strictly increasing,
                uniformly spaced points aligned with a real-FFT grid.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        frequencies = jnp.asarray(frequencies)
        if (
            frequencies.ndim != 1
            or jnp.iscomplexobj(frequencies)
            or not jnp.all(jnp.isfinite(frequencies))
        ):
            raise ValueError("frequencies must be a one-dimensional real finite array")
        try:
            fd = jnp.asarray(fd)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "fd must be a one-dimensional finite numeric array"
            ) from error
        if fd.ndim != 1 or fd.shape != frequencies.shape:
            raise ValueError("Frequency and data arrays must have the same shape")
        if not bool(jnp.all(jnp.isfinite(fd))):
            raise ValueError("fd must contain only finite values")
        if len(frequencies) < 2:
            raise ValueError("frequencies must contain at least two points")
        frequencies_np = np.asarray(frequencies, dtype=float)
        if frequencies_np[0] < 0 or not np.all(np.diff(frequencies_np) > 0):
            raise ValueError("frequencies must be nonnegative and strictly increasing")

        delta_f = (frequencies_np[-1] - frequencies_np[0]) / (len(frequencies) - 1)
        dtype = np.asarray(frequencies).dtype
        epsilon = np.finfo(dtype).eps if np.issubdtype(dtype, np.inexact) else 0.0
        tolerance = max(8 * epsilon, 1e-12)
        start_idx = round(frequencies_np[0] / delta_f)
        last_idx = start_idx + len(frequencies) - 1
        expected = np.arange(start_idx, last_idx + 1) * delta_f
        if not np.allclose(
            frequencies_np, expected, rtol=tolerance, atol=tolerance * delta_f
        ):
            raise ValueError(
                "frequencies must be uniformly spaced and aligned with a real-FFT grid"
            )

        n_samples = 2 * last_idx
        delta_t = 1 / (n_samples * delta_f)
        data_fd_full = jnp.zeros(last_idx + 1, dtype=jnp.result_type(fd, 1j))
        data_fd_full = data_fd_full.at[start_idx:].set(fd)
        data_fd_full = data_fd_full.at[0].set(jnp.real(data_fd_full[0]))
        data_fd_full = data_fd_full.at[-1].set(jnp.real(data_fd_full[-1]))
        data_td_full = jnp.fft.irfft(data_fd_full, n=n_samples) / delta_t
        # Create a Data object
        data = cls.from_td(
            data_td_full,
            delta_t,
            start_time=start_time,
            trigger_time=trigger_time,
            name=name,
            window=jnp.ones_like(data_td_full),
        )
        data._fd = data_fd_full

        return data

    @classmethod
    def from_gwosc(
        cls,
        ifo: str,
        gps_start_time: float,
        gps_end_time: float,
        cache: bool = True,
        *,
        trigger_time: Optional[float] = None,
        **kws,
    ) -> Self:
        """Pull data from GWOSC.

        The reference time of ``t_c`` defaults to two seconds before the requested
        end time. Detectors fetched for the same interval therefore share a
        reference, even if their sample timestamps are rounded differently.
        ``start_time`` and ``end_time`` describe the returned samples, using their
        epoch, time step and sample count rather than the requested boundaries.

        Args:
            ifo: Detector name.
            gps_start_time: Requested GPS start time of the data.
            gps_end_time: Requested GPS end time of the data.
            cache: Whether to cache the data (default: True).
            trigger_time: GPS reference time of ``t_c``. Defaults to two seconds
                before the requested end time.
            **kws: Keyword arguments for `gwpy.timeseries.TimeSeries.fetch_open_data`.

        Returns:
            Data: Data object with the fetched time domain data.

        Raises:
            ValueError: If requested times or an explicit trigger time are not
                finite real scalars, or the end time is not later than the start.
        """
        time_arguments = [
            ("gps_start_time", gps_start_time),
            ("gps_end_time", gps_end_time),
        ]
        if trigger_time is not None:
            time_arguments.append(("trigger_time", trigger_time))
        # --------------------------------------------------------------
        # Timing validation
        # --------------------------------------------------------------

        for name, value in time_arguments:
            time = np.asarray(value)
            if (
                time.ndim != 0
                or not np.issubdtype(time.dtype, np.number)
                or np.iscomplexobj(time)
                or not np.isfinite(float(time))
            ):
                raise ValueError(f"{name} must be a finite real scalar")
        gps_start_time = float(gps_start_time)
        gps_end_time = float(gps_end_time)
        if gps_end_time <= gps_start_time:
            raise ValueError("gps_end_time must be later than gps_start_time")
        duration = gps_end_time - gps_start_time
        logger.info(
            f"Fetching {duration} s of {ifo} data from GWOSC "
            f"[{gps_start_time}, {gps_end_time}]"
        )

        data_td = TimeSeries.fetch_open_data(
            ifo, gps_start_time, gps_end_time, cache=cache, **kws
        )
        if data_td.epoch is None:
            raise ValueError("GWOSC data have no epoch")
        return cls.from_td(
            data_td.value,
            data_td.dt.value,
            # Astropy types Time.value broadly; from_td validates the raw scalar.
            start_time=data_td.epoch.value,
            trigger_time=(
                gps_end_time - DEFAULT_POST_TRIGGER_DURATION
                if trigger_time is None
                else trigger_time
            ),
            name=ifo,
        )

    @classmethod
    def _from_gwf(
        cls,
        path: str,
        channel: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Self:
        """Load data from a GWF (Gravitational Wave Frame) file using gwpy.

        Args:
            path: Path to the .gwf file.
            channel: Channel name (e.g., ``'H1:GDS-CALIB_STRAIN'``). If ``None``,
                common preset channel names are tried automatically.
            start_time: GPS start time to read (optional).
            end_time: GPS end time to read (optional).

        Returns:
            Data: Data object with the loaded time domain data.
        """
        kwargs: dict = {}
        if start_time is not None:
            kwargs["start"] = start_time
        if end_time is not None:
            kwargs["end"] = end_time

        strain = None

        if channel is not None:
            try:
                strain = TimeSeries.read(source=path, channel=channel, **kwargs)
                logger.info(f"Successfully loaded channel {channel} from {path}")
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"Could not read channel '{channel}' from {path}: {exc}"
                ) from exc
        else:
            _ligo_channels = [
                "GDS-CALIB_STRAIN",
                "DCS-CALIB_STRAIN_C01",
                "DCS-CALIB_STRAIN_C02",
                "DCH-CLEAN_STRAIN_C02",
                "GWOSC-16KHZ_R1_STRAIN",
                "GWOSC-4KHZ_R1_STRAIN",
            ]
            _virgo_channels = [
                "Hrec_hoft_V1O2Repro2A_16384Hz",
                "FAKE_h_16384Hz_4R",
                "GWOSC-16KHZ_R1_STRAIN",
                "GWOSC-4KHZ_R1_STRAIN",
            ]
            _preset_channels: dict[str, list[str]] = {
                "H1": _ligo_channels,
                "L1": _ligo_channels,
                "V1": _virgo_channels,
            }
            for det, ch_types in _preset_channels.items():
                if strain is not None:
                    break
                for ch_type in ch_types:
                    ch = f"{det}:{ch_type}"
                    try:
                        strain = TimeSeries.read(source=path, channel=ch, **kwargs)
                        logger.info(f"Successfully loaded channel {ch} from {path}")
                        channel = ch
                        break
                    except (RuntimeError, ValueError):
                        pass

            if strain is None:
                raise ValueError(
                    f"Could not load any data from '{path}'. "
                    "Please specify the channel name explicitly via the 'channel' argument."
                )

        name = channel.split(":")[0] if channel and ":" in channel else ""
        return cls.from_td(
            jnp.array(strain.value),
            strain.dt.value,
            start_time=strain.epoch.value,
            name=name,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        channel: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Self:
        """Load data from a file.

        Supported formats:

        * ``.npz`` — NumPy archive containing ``td`` (time-domain strain),
          ``dt`` (time step in seconds), and ``start_time`` (GPS start time). It may
          also hold ``window``, ``trigger_time``, ``frequency_bounds`` and ``fd``, which
          ``to_file`` writes; without them the data get the default Tukey window,
          trigger time and frequency range of the data.
        * ``.gwf`` / ``.gwf.gz`` — LIGO/Virgo Gravitational Wave Frame file.
          Requires `gwpy`. Specify *channel* (e.g. ``'H1:GDS-CALIB_STRAIN'``);
          if omitted, common preset channel names are tried automatically.
        * ``.hdf5`` / ``.h5`` / ``.hdf`` — HDF5 file readable by gwpy.
          *channel* names the dataset; it is required when the file contains several.
        * ``.csv`` — Two-column CSV time-series file (gwpy format). *channel* only
          names the data.

        Args:
            path: Path to the file.
            channel: Channel name for frame or HDF5 files
                (e.g. ``'H1:GDS-CALIB_STRAIN'``). Ignored for ``.npz``.
            start_time: GPS start time to read (optional; for gwpy-backed formats).
            end_time: GPS end time to read (optional; for gwpy-backed formats).

        Returns:
            Data: Data object with the loaded time domain data.
        """
        path_lower = path.lower()
        if path_lower.endswith(".npz"):
            with np.load(path) as npz:
                if "td" not in npz or "dt" not in npz or "start_time" not in npz:
                    raise ValueError(
                        "The file must contain 'td', 'dt', and 'start_time' keys."
                    )
                td = jnp.array(npz["td"])
                dt = npz["dt"]
                t0 = npz["start_time"]
                name = str(npz.get("name", ""))
                window = jnp.array(npz["window"]) if "window" in npz else None
                trigger_time = npz.get("trigger_time")
                bounds = npz.get("frequency_bounds")
                stored_fd = npz.get("fd")
            data = cls.from_td(
                td,
                dt,
                start_time=t0,
                trigger_time=trigger_time,
                name=name,
                window=window,
            )
            if bounds is not None:
                data.frequency_bounds = bounds
            if stored_fd is not None:
                # ------------------------------------------------------
                # Stored Fourier data validation
                # ------------------------------------------------------

                normalized_fd = data.normalize_fd(stored_fd)
                if not bool(jnp.array_equal(normalized_fd, stored_fd)):
                    raise ValueError(
                        "Stored fd must have real DC and Nyquist components"
                    )
                calculated_fd = data.fft()
                # Complex magnitudes can overflow even when both components are
                # finite. An infinite scale would accept any incoherent cache.
                scale = float(
                    jnp.maximum(
                        jnp.max(jnp.abs(normalized_fd.real)),
                        jnp.max(jnp.abs(normalized_fd.imag)),
                    )
                )
                # A saved additive FD series has undergone an inverse FFT before
                # this forward FFT. Allow that roundoff at the stored precision,
                # including near-zero bins where relative error is uninformative.
                precision = max(
                    np.finfo(array.dtype).eps
                    for array in (calculated_fd.real, normalized_fd.real)
                )
                relative_tolerance = max(32 * precision, 1e-10)
                absolute_tolerance = scale * max(32 * precision, 1e-12)
                # Compare components: allclose on complex arrays also computes
                # magnitudes, which can overflow for otherwise finite samples.
                if not bool(
                    jnp.allclose(
                        normalized_fd.real,
                        calculated_fd.real,
                        rtol=relative_tolerance,
                        atol=absolute_tolerance,
                    )
                    & jnp.allclose(
                        normalized_fd.imag,
                        calculated_fd.imag,
                        rtol=relative_tolerance,
                        atol=absolute_tolerance,
                    )
                ):
                    raise ValueError(
                        "Stored fd must agree with the windowed time-domain data"
                    )
                data._fd = normalized_fd
            return data
        elif path_lower.endswith((".gwf", ".gwf.gz")):
            return cls._from_gwf(
                path, channel=channel, start_time=start_time, end_time=end_time
            )
        elif path_lower.endswith((".hdf5", ".h5", ".hdf", ".csv")):
            kwargs: dict = {}
            if channel is not None and not path_lower.endswith(".csv"):
                kwargs["path"] = channel
            if start_time is not None:
                kwargs["start"] = start_time
            if end_time is not None:
                kwargs["end"] = end_time
            strain = TimeSeries.read(source=path, **kwargs)
            name = channel.split(":")[0] if channel and ":" in channel else ""
            return cls.from_td(
                jnp.array(strain.value),
                strain.dt.value,
                start_time=strain.epoch.value,
                name=name,
            )
        else:
            raise ValueError(
                f"Unsupported file format for '{path}'. "
                "Supported formats: .npz, .gwf, .gwf.gz, .hdf5, .h5, .hdf, .csv"
            )

    # ------------------------------------------------------------------
    # Freezing and special methods
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in ("td", "window", "delta_t")
            and getattr(self, "_fd", None) is not None
        ):
            self._check_mutable()
            raise RuntimeError(
                f"Cannot change {name} after frequency-domain data have been fixed"
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if (
            name in ("td", "window", "delta_t")
            and getattr(self, "_fd", None) is not None
        ):
            self._check_mutable()
            raise RuntimeError(
                f"Cannot delete {name} after frequency-domain data have been fixed"
            )
        super().__delattr__(name)

    def _freeze(self) -> None:
        """Own immutable payloads and compute the FFT before freezing this object."""
        if self.is_frozen:
            return
        for name in ("td", "window", "_fd"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, jax.Array):
                # Detaching an alias preserves the already-fixed FD representation.
                object.__setattr__(self, name, jnp.array(value, copy=True))
        if not self.is_empty:
            _ = self.fd
        object.__setattr__(self, "delta_t", float(self.delta_t))
        super()._freeze()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            + f"delta_t={self.delta_t}, start_time={self.start_time})"
        )

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

    def __bool__(self) -> bool:
        """Check if the data is empty."""
        return len(self.td) > 0

    # ------------------------------------------------------------------
    # Segment properties
    # ------------------------------------------------------------------

    @property
    def start_time(self) -> float:
        """GPS start time of the segment, as a finite real scalar."""
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        # --------------------------------------------------------------
        # Timing validation
        # --------------------------------------------------------------

        time = np.asarray(value)
        if (
            time.ndim != 0
            or not np.issubdtype(time.dtype, np.number)
            or np.iscomplexobj(time)
            or not np.isfinite(float(time))
        ):
            raise ValueError("start_time must be a finite real scalar")
        self._start_time = float(time)

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
    def duration(self) -> FloatLike:
        """Gets duration of the data in seconds.

        Returns:
            float: Duration in seconds.
        """
        return self.n_time * self.delta_t

    @property
    def end_time(self) -> float:
        """Exclusive GPS end of the segment, one time step after its last sample."""
        return self.start_time + float(self.duration)

    @property
    def sampling_frequency(self) -> FloatLike:
        """Gets sampling frequency of the data.

        Returns:
            float: Sampling frequency in Hz.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

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
    def trigger_time(self) -> float:
        """Gets the GPS time of the event trigger, the reference time of ``t_c``.

        Returns:
            float: The trigger time in seconds; unless it was set, 2 s before the
                end of the data.

        Raises:
            ValueError: If the data are empty and no trigger time was set.
        """
        if self._trigger_time is not None:
            return self._trigger_time
        if self.is_empty:
            raise ValueError("Empty data have no trigger time.")
        return self.end_time - DEFAULT_POST_TRIGGER_DURATION

    @trigger_time.setter
    def trigger_time(self, value: Optional[float]) -> None:
        """Sets a finite scalar trigger time; None restores the default."""
        if value is None:
            self._trigger_time = None
            return
        # --------------------------------------------------------------
        # Timing validation
        # --------------------------------------------------------------

        time = np.asarray(value)
        if (
            time.ndim != 0
            or not np.issubdtype(time.dtype, np.number)
            or np.iscomplexobj(time)
            or not np.isfinite(float(time))
        ):
            raise ValueError("trigger_time must be a finite real scalar")
        self._trigger_time = float(time)

    @property
    def frequency_bounds(self) -> tuple[float, float]:
        """Gets the frequency range of the data.

        Returns:
            tuple: The minimum and maximum frequency in Hz. The default, ``(0, inf)``,
                does not restrict the data.
        """
        return self._frequency_bounds

    @frequency_bounds.setter
    def frequency_bounds(self, bounds: tuple[float, float]) -> None:
        """Sets the frequency range of the data.

        Raises:
            ValueError: If ``bounds`` is not a pair of numbers, if ``f_min`` or
                ``f_max`` is NaN, if ``f_min`` is negative or not below ``f_max``, if
                ``f_max`` is above the Nyquist frequency of the data, or if the range
                holds fewer than two frequencies of the data.
        """
        # --------------------------------------------------------------
        # Frequency range validation
        # --------------------------------------------------------------

        try:
            low, high = (float(f) for f in bounds)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"The frequency range of the data must be a pair (f_min, f_max) of "
                f"numbers, got {bounds!r}."
            ) from error
        if jnp.isnan(low) or jnp.isnan(high):
            raise ValueError(
                f"f_min and f_max of the data must not be NaN, got ({low}, {high})."
            )
        if low < 0:
            raise ValueError(f"f_min of the data must not be negative, got {low} Hz.")
        if not low < high:
            raise ValueError(
                f"f_min of the data must be below f_max, got ({low}, {high}) Hz."
            )
        if not self.is_empty and jnp.isfinite(high):
            nyquist = float(self.sampling_frequency) / 2
            # The Nyquist frequency is 1 / delta_t / 2, which is off by a rounding error
            # for some sampling rates (744 Hz, for example): do not reject f_max for that.
            if high > nyquist and not jnp.isclose(high, nyquist, rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"f_max of the data, {high} Hz, is above their Nyquist "
                    f"frequency, {nyquist} Hz."
                )
        if not self.is_empty:
            n_inside = int(
                jnp.sum((self.frequencies >= low) & (self.frequencies <= high))
            )
            if n_inside < 2:
                raise ValueError(
                    f"The frequency range of the data ({low}, {high}) Hz holds {n_inside} "
                    f"frequencies of the data, which are {1 / self.duration} Hz "
                    "apart. It needs at least two."
                )
        self._frequency_bounds = (low, high)

    @property
    def frequency_mask(self) -> Bool[Array, "n_freq"]:
        """Gets the mask of the frequencies inside the frequency range of the data.

        Returns:
            Array: True for the frequencies from ``f_min`` to ``f_max``.
        """
        f_min, f_max = self.frequency_bounds
        return (f_min <= self.frequencies) & (self.frequencies <= f_max)

    # ------------------------------------------------------------------
    # Frequency-domain data
    # ------------------------------------------------------------------

    def set_tukey_window(
        self,
        alpha: Optional[float] = None,
        roll_off: Optional[float] = None,
    ) -> None:
        """Configure the Tukey window used when FFTing time-domain data.

        This must be called before frequency-domain data are computed or supplied.
        Only one of ``alpha`` and ``roll_off`` may be provided.

        Args:
            alpha: Shape parameter passed directly to the Tukey window. If
                supplied, it must be finite and between 0 and 1.
            roll_off: Duration in seconds of the taper on each side. Defaults
                to 0.4 seconds. It must be finite and between 0 and half the
                data duration. The corresponding shape parameter is
                ``2 * roll_off / duration``.

        Raises:
            ValueError: If the data are empty, both parameterizations are
                supplied, or a parameter is outside its valid range.
            RuntimeError: If frequency-domain data have already been fixed.
        """
        self._check_mutable()
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if self.is_empty:
            raise ValueError("Cannot set a Tukey window on empty data")
        if self.has_fd:
            raise RuntimeError(
                "Cannot change the Tukey window after frequency-domain data "
                "have been fixed"
            )
        if alpha is not None and roll_off is not None:
            raise ValueError("Specify either alpha or roll_off, not both")
        duration = float(self.duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("data duration must be finite and positive")

        if alpha is None:
            roll_off = DEFAULT_TUKEY_ROLL_OFF if roll_off is None else roll_off
            roll_off = float(roll_off)
            if not np.isfinite(roll_off) or not 0 <= roll_off <= duration / 2:
                raise ValueError(
                    "roll_off must be finite and between 0 and half the data duration"
                )
            resolved_alpha = 2 * roll_off / duration
        else:
            alpha = float(alpha)
            if not np.isfinite(alpha) or not 0 <= alpha <= 1:
                raise ValueError("alpha must be finite and between 0 and 1")
            resolved_alpha = alpha

        logger.debug(f"Setting Tukey window on {self.name or '(unnamed)'}")
        self.window = jnp.array(tukey(self.n_time, resolved_alpha))

    def fft(
        self, window: Optional[Float[Array, "n_time"]] = None
    ) -> Complex[Array, "n_freq"]:
        """Compute the Fourier transform of the windowed time-domain data.

        Args:
            window: Window function to apply to the data before the transform
                (default: the window of the data).

        Returns:
            Array: The Fourier transform, scaled by the time step.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        td = jnp.asarray(self.td)
        if td.ndim != 1 or jnp.iscomplexobj(td) or not bool(jnp.all(jnp.isfinite(td))):
            raise ValueError("td must be a one-dimensional real finite array")
        if td.size == 0:
            raise ValueError("Empty data have no frequency-domain data.")
        time_step = np.asarray(self.delta_t)
        if (
            time_step.ndim != 0
            or not np.issubdtype(time_step.dtype, np.number)
            or np.iscomplexobj(time_step)
            or not np.isfinite(float(time_step))
            or float(time_step) <= 0
        ):
            raise ValueError("delta_t must be finite and positive for non-empty data")
        if window is None:
            window = self.window
        window = jnp.asarray(window)
        if (
            window.shape != td.shape
            or jnp.iscomplexobj(window)
            or not jnp.all(jnp.isfinite(window))
        ):
            raise ValueError(
                "window must be a real finite array with the same shape as td"
            )
        transformed = jnp.fft.rfft(td * window) * float(time_step)
        # --------------------------------------------------------------
        # FFT result validation
        # --------------------------------------------------------------

        if not bool(jnp.all(jnp.isfinite(transformed))):
            raise ValueError(
                "FFT of the windowed time-domain data must contain only finite values"
            )
        return transformed

    @property
    def has_fd(self) -> bool:
        """Checks whether frequency-domain data have been computed or supplied.

        Returns:
            bool: True if frequency-domain data are fixed, False otherwise.
        """
        return self._fd is not None

    @property
    def fd(self) -> Complex[Array, "n_time // 2 + 1"]:
        """Gets the frequency-domain data.

        Data built from frequency-domain arrays hold them with real DC and Nyquist
        components. Otherwise they are ``fft()`` of the time-domain data, computed
        on first use and kept, so the window can no longer change.

        Returns:
            Array: frequency-domain data.

        Raises:
            ValueError: If the data are empty.
        """
        if self._fd is None:
            if self.is_empty:
                raise ValueError("Empty data have no frequency-domain data.")
            # Own the inputs as well as their FFT: mutable aliases must not leave
            # the fixed frequency-domain cache inconsistent with its source.
            for name in ("td", "window"):
                value = getattr(self, name)
                if not isinstance(value, jax.Array):
                    setattr(self, name, jnp.array(value, copy=True))
            logger.info(f"Computing FFT of {self.name} data")
            transformed = self.fft()
            self.delta_t = float(self.delta_t)
            self._fd = transformed
        return self._fd

    def normalize_fd(
        self, frequency_series: Complex[Array, "n_freq"]
    ) -> Complex[Array, "n_freq"]:
        """Validate a finite series on this grid and make real-FFT endpoints real.

        DC is real; the final bin is also real for an even number of time samples.
        With an odd number of samples, the final bin keeps its complex phase.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if self.is_empty:
            raise ValueError("Empty data have no frequency-domain data.")
        try:
            series = jnp.asarray(frequency_series)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "frequency_series must be a finite numeric array"
            ) from error
        if series.shape != (self.n_freq,):
            raise ValueError(
                f"frequency_series has shape {series.shape}, but fd has shape "
                f"{(self.n_freq,)}."
            )
        if not bool(jnp.all(jnp.isfinite(series))):
            raise ValueError("frequency_series must contain only finite values")
        series = series.astype(jnp.result_type(series, 1j))
        series = series.at[0].set(jnp.real(series[0]))
        if self.n_time % 2 == 0:
            series = series.at[-1].set(jnp.real(series[-1]))
        return series

    def add_fd(self, frequency_series: Complex[Array, "n_freq"]) -> Self:
        """Return a copy of the data with a frequency series added to ``fd``.

        The copy keeps every other property of the data, such as the trigger time and
        the frequency range of the data, and has no window. Imaginary DC and even-length
        Nyquist components are discarded so the TD and FD arrays stay conjugate.
        The original data, including their FFT cache, are unchanged. The returned
        data are mutable even if the original data are frozen.

        Args:
            frequency_series: Array to add, on the frequencies of the data.

        Raises:
            ValueError: If ``frequency_series`` does not have the shape of ``fd`` or
                contains non-finite values.
        """
        frequency_series = self.normalize_fd(frequency_series)
        source_fd = self._fd if self._fd is not None else self.fft()
        added_fd = self.normalize_fd(source_fd + frequency_series)
        td = jnp.fft.irfft(added_fd, n=self.n_time) / self.delta_t
        data = type(self).from_td(
            td,
            self.delta_t,
            start_time=self.start_time,
            trigger_time=self._trigger_time,
            name=self.name,
            window=jnp.ones_like(td),
        )
        data._fd = added_fd
        data._frequency_bounds = self.frequency_bounds
        return data

    def frequency_slice(
        self,
        f_min: FloatLike,
        f_max: FloatLike,
    ) -> tuple[Float[Array, " n_sample"], Complex[Array, " n_sample"]]:
        """Slice the data in the frequency domain.
        This is the main function which interacts with the likelihood.

        Args:
            f_min: Minimum frequency of the slice in Hz.
            f_max: Maximum frequency of the slice in Hz.

        Returns:
            tuple: The frequencies in the slice and the sliced data in the frequency
                domain.
        """
        mask = (self.frequencies >= f_min) * (self.frequencies <= f_max)
        return self.frequencies[mask], self.fd[mask]

    # ------------------------------------------------------------------
    # Conversion and saving
    # ------------------------------------------------------------------

    def to_psd(self, **kws) -> "PowerSpectrum":
        """Compute a Welch estimate of the power spectral density of the data.

        Args:
            **kws: Keyword arguments for `scipy.signal.welch`.

        Returns:
            PowerSpectrum: Power spectral density of the data.
        """
        freq, psd = welch(self.td, fs=float(self.sampling_frequency), **kws)
        return PowerSpectrum(jnp.asarray(psd), jnp.asarray(freq), self.name)

    def to_file(self, path: str) -> None:
        """Save the data to a file.

        Supported formats:

        * ``.npz`` — NumPy archive with keys ``td`` (time-domain strain),
          ``dt`` (time step in seconds), ``start_time`` (GPS), ``name``, ``window``,
          ``frequency_bounds`` and, if they were set, ``trigger_time`` and ``fd``. It is the only
          format that keeps the window, the trigger time and the frequency range of the data.
        * ``.txt`` / ``.dat`` — Three-column whitespace-separated text file
          containing ``[f, real(h(f)), imag(h(f))]`` (frequency-domain strain).
        * ``.csv`` — Same as ``.txt`` but comma-separated.
        * ``.gwf`` — LIGO/Virgo Gravitational Wave Frame file (time-domain).
          Requires ``gwpy``.
        * ``.hdf5`` / ``.h5`` — HDF5 file (time-domain). Requires ``gwpy``.

        Args:
            path: Path to the output file (extension determines format).

        Raises:
            ValueError: If the file extension is not supported.
        """
        path_lower = path.lower()
        if path_lower.endswith(".npz"):
            fields: dict[str, Any] = {
                "td": np.array(self.td),
                "dt": self.delta_t,
                "start_time": self.start_time,
                "name": self.name,
                "window": np.array(self.window),
                "frequency_bounds": np.array(self.frequency_bounds),
            }
            if self._trigger_time is not None:
                fields["trigger_time"] = self._trigger_time
            if self._fd is not None:
                fields["fd"] = np.array(self._fd)
            np.savez(path, **fields)
        elif path_lower.endswith((".txt", ".dat", ".csv")):
            fd = np.array(self.fd)
            data = np.column_stack(
                [
                    np.array(self.frequencies),
                    np.real(fd),
                    np.imag(fd),
                ]
            )
            if path_lower.endswith(".csv"):
                np.savetxt(
                    path,
                    data,
                    delimiter=",",
                    header="f,real_h(f),imag_h(f)",
                    comments="# ",
                )
            else:
                np.savetxt(path, data, header="f real_h(f) imag_h(f)")
        elif path_lower.endswith((".gwf", ".hdf5", ".h5")):
            channel = (
                self.name
                if ":" in self.name
                else f"{self.name}:STRAIN"
                if self.name
                else "STRAIN"
            )
            ts = TimeSeries(
                np.array(self.td),
                t0=self.start_time,
                dt=float(self.delta_t),
                channel=channel,
            )
            if path_lower.endswith(".gwf"):
                ts.write(path)
            else:
                # In an HDF5 file the channel names the dataset.
                ts.write(path, path=channel)
        else:
            raise ValueError(
                f"Unsupported file format for '{path}'. "
                "Supported formats: .npz, .txt, .dat, .csv, .gwf, .hdf5, .h5"
            )


class PowerSpectrum(_Freezable, ABC):
    """Class representing a power spectral density.

    Attributes:
        name: Name of the power spectrum.
        values: Array of PSD values.
        frequencies: Array of frequencies corresponding to PSD values.
    """

    name: str
    values: Float[Array, "n_freq"]
    frequencies: Float[Array, "n_freq"]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        values: Float[Array, "n_freq"] = _EMPTY_ARRAY,
        frequencies: Float[Array, "n_freq"] = _EMPTY_ARRAY,
        name: Optional[str] = None,
    ) -> None:
        """Initialize PowerSpectrum.

        Args:
            values: Array of PSD values. Defaults to empty array.
            frequencies: Array of frequencies in Hz. Defaults to empty array.
            name: Name of the power spectrum. Defaults to None.

        Raises:
            ValueError: If arrays are not one-dimensional real numeric arrays of
                the same shape, or frequencies are not finite, nonnegative and
                unique. Frequency ordering need not be monotonic.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        values_np, frequencies_np = np.asarray(values), np.asarray(frequencies)
        for label, array in (("values", values_np), ("frequencies", frequencies_np)):
            if (
                array.ndim != 1
                or not np.issubdtype(array.dtype, np.number)
                or np.iscomplexobj(array)
            ):
                raise ValueError(
                    f"PSD {label} must be a one-dimensional real numeric array"
                )
        if values_np.shape != frequencies_np.shape:
            raise ValueError("PSD values and frequencies must have the same shape")
        if (
            not np.all(np.isfinite(frequencies_np))
            or np.any(frequencies_np < 0)
            or np.unique(frequencies_np).size != frequencies_np.size
        ):
            raise ValueError("PSD frequencies must be finite, nonnegative and unique")
        self.values = jnp.asarray(values)
        self.frequencies = jnp.asarray(frequencies)
        self.name = name or ""

    def _freeze(self) -> None:
        """Detach mutable input buffers before freezing this power spectrum."""
        if self.is_frozen:
            return
        for name in ("values", "frequencies"):
            value = getattr(self, name)
            if not isinstance(value, jax.Array):
                setattr(self, name, jnp.array(value, copy=True))
        super()._freeze()

    @classmethod
    def from_file(cls, path: str, is_asd: bool = False) -> Self:
        """Load a power spectrum from a file.

        Supported formats:

        * ``.npz`` — NumPy archive containing ``values`` and ``frequencies``
          arrays (``to_file`` writes PSD values).
        * ``.txt`` / ``.dat`` — two-column whitespace-separated text file
          ``(frequency, value)``.
        * ``.csv`` — same two-column format as ``.txt``/``.dat`` but
          comma-separated.

        In every format the values are a PSD in $\\mathrm{Hz}^{-1}$, unless
        *is_asd* is set.

        Args:
            path: Path to the PSD file.
            is_asd: If ``True``, the values in the file are an amplitude spectral
                density in $\\mathrm{Hz}^{-1/2}$ and are squared to obtain the PSD.
                Applies to every format. Defaults to ``False``.

        Returns:
            PowerSpectrum: Loaded power spectrum.
        """
        path_lower = path.lower()
        if path_lower.endswith(".npz"):
            with np.load(path) as data:
                if "values" not in data or "frequencies" not in data:
                    raise ValueError(
                        "The file must contain 'values' and 'frequencies' keys."
                    )
                values = jnp.array(data["values"])
                frequencies = jnp.array(data["frequencies"])
                name = str(data.get("name", ""))
            if is_asd:
                if jnp.iscomplexobj(values) or bool(jnp.any(values < 0)):
                    raise ValueError("ASD values must be real and nonnegative")
                values = values**2
            return cls(values, frequencies, name)
        elif path_lower.endswith((".txt", ".dat", ".csv")):
            delimiter = "," if path_lower.endswith(".csv") else None
            frequencies_np, values_np = np.genfromtxt(
                path, delimiter=delimiter, unpack=True
            )
            if is_asd:
                if np.iscomplexobj(values_np) or np.any(values_np < 0):
                    raise ValueError("ASD values must be real and nonnegative")
                values_np = values_np**2
            return cls(jnp.array(values_np), jnp.array(frequencies_np))
        else:
            raise ValueError(
                f"Unsupported file format for '{path}'. "
                "Supported formats: .npz, .txt, .dat, .csv"
            )

    # ------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    def delta_f(self) -> FloatScalar:
        """Get the positive spacing of a uniform frequency grid, in any order.

        Returns:
            float: Frequency resolution in Hz.

        Raises:
            ValueError: If fewer than two frequencies are present or the grid
                is not uniformly spaced.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if self.n_freq < 2:
            raise ValueError("PSD frequency spacing requires at least two frequencies")
        frequencies = np.sort(np.asarray(self.frequencies, dtype=float))
        spacing = (frequencies[-1] - frequencies[0]) / (self.n_freq - 1)
        dtype = np.asarray(self.frequencies).dtype
        epsilon = np.finfo(dtype).eps if np.issubdtype(dtype, np.inexact) else 0.0
        tolerance = max(8 * epsilon, 1e-12)
        expected = frequencies[0] + np.arange(self.n_freq) * spacing
        if (
            not np.isfinite(spacing)
            or spacing <= 0
            or not np.all(np.diff(frequencies) > 0)
            or not np.allclose(
                frequencies, expected, rtol=tolerance, atol=tolerance * spacing
            )
        ):
            raise ValueError("PSD frequency spacing requires a uniform frequency grid")
        return jnp.asarray(spacing)

    @property
    def delta_t(self) -> FloatScalar:
        """Gets time resolution.

        Returns:
            float: Time resolution in seconds.
        """
        return 1 / self.sampling_frequency

    @property
    def duration(self) -> FloatScalar:
        """Gets duration of the data.

        Returns:
            float: Duration in seconds.
        """
        return 1 / self.delta_f

    @property
    def sampling_frequency(self) -> FloatScalar:
        """Gets sampling frequency.

        Returns:
            float: Sampling frequency in Hz.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if self.is_empty:
            raise ValueError("Empty PSD has no sampling frequency")
        with jax.ensure_compile_time_eval():
            frequency = 2 * jnp.max(jnp.asarray(self.frequencies))
            if not bool(jnp.isfinite(frequency) & (frequency > 0)):
                raise ValueError("PSD sampling frequency must be finite and positive")
        return frequency

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def frequency_slice(
        self, f_min: float, f_max: float
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """Slice the power spectrum to a frequency range.

        Args:
            f_min: Minimum frequency of the slice in Hz.
            f_max: Maximum frequency of the slice in Hz.

        Returns:
            tuple: Contains:
                - frequencies: Frequencies in the slice
                - values: Sliced PSD values
        """
        mask = (self.frequencies >= f_min) & (self.frequencies <= f_max)
        return self.frequencies[mask], self.values[mask]

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
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if self.n_freq < 2:
            raise ValueError("PSD interpolation requires at least two frequencies")
        frequencies_np = np.asarray(self.frequencies)
        interp = interp1d(
            self.frequencies,
            self.values,
            kind=kind,
            fill_value=(
                self.values[int(np.argmin(frequencies_np))],
                self.values[int(np.argmax(frequencies_np))],
            ),
            bounds_error=False,
            **kws,
        )
        return PowerSpectrum(interp(frequencies), frequencies, self.name)

    def simulate_data(
        self,
        key: Key[Array, ""],
    ) -> Complex[Array, " n_sample"]:
        """Simulate noise data based on the power spectrum.

        Args:
            key: JAX PRNG key for random number generation.

        Returns:
            Complex frequency series of simulated noise data.

        Raises:
            ValueError: If the frequencies do not form a uniform grid, any PSD
                value is not finite and positive, or the resulting variance
                cannot be represented as a finite positive number.
        """
        # The PSD is concrete; only the PRNG key varies when this method is JITed.
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        with jax.ensure_compile_time_eval():
            if jnp.iscomplexobj(self.values) or not bool(
                jnp.all(jnp.isfinite(self.values) & (self.values > 0))
            ):
                raise ValueError(
                    "Noise simulation requires finite and positive PSD values"
                )
            var = (self.values / 4) / self.delta_f
            if not bool(jnp.all(jnp.isfinite(var) & (var > 0))):
                raise ValueError("Noise variance must be finite and positive")
        key, subkey = jax.random.split(key, 2)
        noise_real = jax.random.normal(key, shape=var.shape) * jnp.sqrt(var)
        noise_imag = jax.random.normal(subkey, shape=var.shape) * jnp.sqrt(var)
        return noise_real + 1j * noise_imag

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

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
