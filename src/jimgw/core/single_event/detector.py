import copy
import logging
import os
import tempfile
import zlib
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import requests
from beartype import beartype as typechecker
from jaxtyping import Array, Complex, Float, Key, jaxtyped

from jimgw.core.constants import (
    C_SI,
    DEG_TO_RAD,
    EARTH_RADIUS_LIGHT_S,
    EARTH_SEMI_MAJOR_AXIS,
    EARTH_SEMI_MINOR_AXIS,
    MTSUN,
)
from jimgw.core.single_event.data import (
    DEFAULT_POST_TRIGGER_DURATION,
    Data,
    PowerSpectrum,
)
from jimgw.core.single_event.mixins import _Freezable
from jimgw.core.single_event.polarization import Polarization
from jimgw.core.single_event.time_utils import cached_greenwich_mean_sidereal_time
from jimgw.core.single_event.utils import complex_inner_product, inner_product
from jimgw.typing import ComplexScalar, FloatScalar

logger = logging.getLogger(__name__)

# TODO: Need to expand this list. Currently it is only O3.
asd_file_dict = {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",
}


class Detector(_Freezable, ABC):
    """Base class for all detectors.

    Attributes:
        name (str): Name of the detector.
        data (Data): Detector data object.
        psd (PowerSpectrum): Power spectral density object.
    """

    name: str

    # NOTE: for some detectors (e.g. LISA, ET) data could be a list of Data
    # objects so this might be worth revisiting
    data: Data
    psd: PowerSpectrum

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self):
        if not jax.config.read("jax_enable_x64"):
            raise RuntimeError(
                "Detector requires JAX to run in 64-bit (float64) mode, "
                "but jax_enable_x64 is currently False.\n\n"
                "Please enable float64 before creating any Detector by putting at the very top of your script:\n"
                "    import jax\n"
                "    jax.config.update('jax_enable_x64', True)\n"
                "and then re-run."
            )

    def _validate_geometry(self) -> None:
        """Validate detector-specific geometry before likelihood ownership freezes."""

    def _freeze(self) -> None:
        """Freeze the detector and its attached data and PSD in place."""
        if self.is_frozen:
            return
        self.data._freeze()
        self.psd._freeze()
        super()._freeze()

    # ------------------------------------------------------------------
    # Detector response
    # ------------------------------------------------------------------

    @abstractmethod
    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
    ) -> Complex[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the frequency domain.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.
            h_sky (dict[str, Float[Array, "n_sample"]]): Dictionary mapping polarization names
                to frequency-domain waveforms. The keys are polarization names (e.g., 'plus', 'cross')
                and values are complex strain arrays.
            params (dict): Dictionary of source parameters including:
                - ra (Float): Right ascension in radians
                - dec (Float): Declination in radians
                - psi (Float): Polarization angle in radians
                - t_c (Float): The difference between peak time and trigger time in sec

        The trigger and segment start are read from the attached data.

        Returns:
            Complex[Array, "n_sample"]: Complex strain measured by the detector in frequency domain.
        """

    @abstractmethod
    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
    ) -> Float[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the time domain.

        Args:
            time: Array of time samples.
            h_sky: Dictionary mapping polarization names to time-domain waveforms.
            params: Dictionary of source parameters.

        Returns:
            Array of detector response in time domain.
        """

    # ------------------------------------------------------------------
    # Data and PSD
    # ------------------------------------------------------------------

    def clear_data_and_psd(self) -> None:
        """Clear the data and PSD of the detector."""
        self._check_mutable()
        self.data = Data()
        self.psd = PowerSpectrum()

    def slice_to_frequency_range(
        self, f_min: float, f_max: float
    ) -> tuple[
        Float[Array, " n_freq"], Complex[Array, " n_freq"], Float[Array, " n_freq"]
    ]:
        """Slice the data and the PSD to the given frequency range.

        Args:
            f_min: Minimum frequency.
            f_max: Maximum frequency.

        Returns:
            tuple: The frequencies, the frequency-domain data and the PSD in the range.

        Raises:
            ValueError: If the detector has no data or no PSD, if the range is wider
                than the frequency range of the data, if the PSD is not finite and
                positive in it, or if the range holds fewer than two frequencies.
        """
        return self._slice_data_to_frequency_range(self.data, f_min, f_max)

    def _slice_data_to_frequency_range(
        self, data: Data, f_min: float, f_max: float
    ) -> tuple[
        Float[Array, " n_freq"], Complex[Array, " n_freq"], Float[Array, " n_freq"]
    ]:
        """Slice a supplied segment, so injections can validate a copy atomically."""
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if data.is_empty or self.psd.is_empty:
            raise ValueError(
                f"Detector {self.name} needs both data and a PSD before it can be "
                "sliced to a frequency range."
            )
        data_f_min, data_f_max = data.frequency_bounds
        if f_min < data_f_min or f_max > data_f_max:
            raise ValueError(
                f"The frequency range [{f_min}, {f_max}] Hz is wider than the "
                f"frequency range of the data of detector {self.name}, "
                f"[{data_f_min}, {data_f_max}] Hz. Choose f_min and f_max inside it."
            )
        nyquist = float(data.sampling_frequency) / 2
        self._check_psd_in_range(self.psd, f_min, min(f_max, nyquist))
        freqs_data, fd_data = data.frequency_slice(f_min, f_max)
        freqs_psd, psd = self.psd.frequency_slice(f_min, f_max)
        if not jnp.array_equal(freqs_data, freqs_psd):
            raise ValueError(
                f"The {self.name} data and PSD must have the same frequencies."
            )
        if len(freqs_data) < 2:
            raise ValueError(
                f"The frequency range [{f_min}, {f_max}] Hz holds {len(freqs_data)} "
                f"frequencies of the data of detector {self.name}. It needs at "
                "least two."
            )
        return freqs_data, fd_data, psd

    def _matched_psd(self, data: Data, psd: PowerSpectrum) -> PowerSpectrum:
        """Return ``psd`` on the frequencies of ``data``, interpolated if they differ.

        Nothing is changed in the detector.
        """
        if psd.is_empty or data.is_empty:
            return psd
        if psd.n_freq == data.n_freq and bool(
            (psd.frequencies == data.frequencies).all()
        ):
            return psd
        return psd.interpolate(data.frequencies)

    def _check_psd_in_range(
        self, psd: PowerSpectrum, f_min: float, f_max: float
    ) -> None:
        """Require finite, positive PSD values and finite weights between f_min and f_max."""
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        frequencies, values = psd.frequency_slice(f_min, f_max)
        values = jnp.asarray(values)
        invalid = ~(jnp.isfinite(values) & (values > 0) & jnp.isfinite(1 / values))
        if bool(jnp.any(invalid)):
            raise ValueError(
                f"The PSD of detector {self.name} is not finite and positive with "
                "a finite reciprocal at "
                f"{int(jnp.sum(invalid))} frequencies in [{f_min}, {f_max}] Hz, the "
                f"first at {float(frequencies[invalid][0])} Hz."
            )


class GroundBased2G(Detector):
    """Object representing a ground-based detector.

    Contains information about the location and orientation of the detector on Earth,
    as well as actual strain data and the PSD of the associated noise.

    Attributes:
        name (str): Name of the detector.
        latitude (Float): Latitude of the detector in radians.
        longitude (Float): Longitude of the detector in radians.
        xarm_azimuth (Float): Azimuth of the x-arm in radians.
        yarm_azimuth (Float): Azimuth of the y-arm in radians.
        xarm_tilt (Float): Tilt of the x-arm in radians.
        yarm_tilt (Float): Tilt of the y-arm in radians.
        elevation (Float): Elevation of the detector in meters.
        polarization_mode (list[Polarization]): List of polarization modes (`pc` for plus and cross) to be used in
            computing antenna patterns; in the future, this could be expanded to
            include non-GR modes.
        data (Data): Array of frequency-domain strain data.
        psd (PowerSpectrum): Power spectral density object.
    """

    polarization_mode: list[Polarization] | tuple[Polarization, ...]
    data: Data
    psd: PowerSpectrum

    latitude: float = 0
    longitude: float = 0
    xarm_azimuth: float = 0
    yarm_azimuth: float = 0
    xarm_tilt: float = 0
    yarm_tilt: float = 0
    elevation: float = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        name: str,
        latitude: float = 0,
        longitude: float = 0,
        elevation: float = 0,
        xarm_azimuth: float = 0,
        yarm_azimuth: float = 0,
        xarm_tilt: float = 0,
        yarm_tilt: float = 0,
        modes: str = "pc",
    ):
        """Initialize a ground-based detector.

        Args:
            name (str): Name of the detector.
            latitude (float, optional): Latitude of the detector in radians. Defaults to 0.
            longitude (float, optional): Longitude of the detector in radians. Defaults to 0.
            elevation (float, optional): Elevation of the detector in meters. Defaults to 0.
            xarm_azimuth (float, optional): Azimuth of the x-arm in radians. Defaults to 0.
            yarm_azimuth (float, optional): Azimuth of the y-arm in radians. Defaults to 0.
            xarm_tilt (float, optional): Tilt of the x-arm in radians. Defaults to 0.
            yarm_tilt (float, optional): Tilt of the y-arm in radians. Defaults to 0.
            modes (str, optional): Polarization modes. Defaults to "pc".
        """
        super().__init__()

        self.name = name

        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth
        self.yarm_azimuth = yarm_azimuth
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt

        self.polarization_mode = [Polarization(m) for m in modes]
        self.data = Data()
        self.psd = PowerSpectrum()
        self._validate_geometry()

    # ------------------------------------------------------------------
    # Geometry validation
    # ------------------------------------------------------------------

    def _validate_geometry(self) -> None:
        """Reject unusable mutable geometry before any detector is frozen."""
        with jax.ensure_compile_time_eval():
            for name in (
                "latitude",
                "longitude",
                "elevation",
                "xarm_azimuth",
                "yarm_azimuth",
                "xarm_tilt",
                "yarm_tilt",
            ):
                try:
                    value = jnp.asarray(getattr(self, name))
                    valid = (
                        value.ndim == 0
                        and jnp.issubdtype(value.dtype, jnp.number)
                        and not jnp.iscomplexobj(value)
                        and bool(jnp.isfinite(value))
                    )
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError(
                        f"Detector {self.name} {name} must be a finite real scalar."
                    )
            if not bool(jnp.all(jnp.isfinite(self.tensor))) or not bool(
                jnp.all(jnp.isfinite(self.vertex))
            ):
                raise ValueError(
                    f"Detector {self.name} geometry must yield finite responses."
                )

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _freeze(self) -> None:
        """Own scalar geometry before freezing."""
        if self.is_frozen:
            return
        self._validate_geometry()
        for name in (
            "latitude",
            "longitude",
            "elevation",
            "xarm_azimuth",
            "yarm_azimuth",
            "xarm_tilt",
            "yarm_tilt",
        ):
            setattr(self, name, float(getattr(self, name)))
        self.polarization_mode = tuple(self.polarization_mode)
        super()._freeze()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _get_arm(
        lat: float, lon: float, tilt: float, azimuth: float
    ) -> Float[Array, "3"]:
        """Construct detector-arm vectors in geocentric Cartesian coordinates.

        Args:
            lat (Float): Vertex latitude in radians.
            lon (Float): Vertex longitude in radians.
            tilt (Float): Arm tilt in radians.
            azimuth (Float): Arm azimuth in radians.

        Returns:
            Float[Array, "3"]: Detector arm vector in geocentric Cartesian coordinates.
        """
        e_lon = jnp.array([-jnp.sin(lon), jnp.cos(lon), 0])
        e_lat = jnp.array(
            [-jnp.sin(lat) * jnp.cos(lon), -jnp.sin(lat) * jnp.sin(lon), jnp.cos(lat)]
        )
        e_h = jnp.array(
            [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)]
        )

        return (
            jnp.cos(tilt) * jnp.cos(azimuth) * e_lon
            + jnp.cos(tilt) * jnp.sin(azimuth) * e_lat
            + jnp.sin(tilt) * e_h
        )

    @property
    def arms(self) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
        """Get the detector arm vectors.

        Returns:
            tuple[Float[Array, "3"], Float[Array, "3"]]: A tuple containing:
                - x: X-arm vector in geocentric Cartesian coordinates
                - y: Y-arm vector in geocentric Cartesian coordinates
        """
        x = self._get_arm(
            self.latitude, self.longitude, self.xarm_tilt, self.xarm_azimuth
        )
        y = self._get_arm(
            self.latitude, self.longitude, self.yarm_tilt, self.yarm_azimuth
        )
        return x, y

    @property
    def tensor(self) -> Float[Array, "3 3"]:
        """Get the detector tensor defining the strain measurement.

        For a 2-arm differential-length detector, this is given by:

        $$
        D_{ij} = \\left(x_i x_j - y_i y_j\\right)/2
        $$

        for unit vectors $x$ and $y$ along the x and y arms.

        Returns:
            Float[Array, "3 3"]: The 3x3 detector tensor in geocentric coordinates.
        """
        # TODO: this could easily be generalized for other detector geometries
        arm1, arm2 = self.arms
        return 0.5 * (
            jnp.einsum("i,j->ij", arm1, arm1) - jnp.einsum("i,j->ij", arm2, arm2)
        )

    @property
    def vertex(self) -> Float[Array, "3"]:
        """Detector vertex coordinates in the reference celestial frame.

        Based on arXiv:gr-qc/0008066 Eqs. (B11-B13) except for a typo in the
        definition of the local radius; see Section 2.1 of LIGO-T980044-10.

        Returns:
            Float[Array, "3"]: Detector vertex coordinates in geocentric Cartesian coordinates.
        """
        # get detector and Earth parameters
        lat = self.latitude
        lon = self.longitude
        h = self.elevation
        major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
        # compute vertex location
        r = major**2 * (
            major**2 * jnp.cos(lat) ** 2 + minor**2 * jnp.sin(lat) ** 2
        ) ** (-0.5)
        x = (r + h) * jnp.cos(lat) * jnp.cos(lon)
        y = (r + h) * jnp.cos(lat) * jnp.sin(lon)
        z = ((minor / major) ** 2 * r + h) * jnp.sin(lat)
        return jnp.array([x, y, z])

    # ------------------------------------------------------------------
    # Detector response
    # ------------------------------------------------------------------

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict[str, Float],
    ) -> Complex[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the frequency domain.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.
            h_sky (dict[str, Float[Array, "n_sample"]]): Dictionary mapping polarization names
                to frequency-domain waveforms. Keys are polarization names (e.g., 'plus', 'cross')
                and values are complex strain arrays.
            params (dict[str, Float]): Dictionary of source parameters containing:
                - ra (Float): Right ascension in radians
                - dec (Float): Declination in radians
                - psi (Float): Polarization angle in radians
                - t_c (Float): The difference between peak time and trigger time in sec

        The attached data supply the trigger and segment start.

        Returns:
            Array: Complex strain measured by the detector in frequency domain, obtained by
                  combining the antenna patterns for each polarization mode.
        """
        ra, dec, psi = params["ra"], params["dec"], params["psi"]
        trigger_time = self.data.trigger_time
        gmst = cached_greenwich_mean_sidereal_time(trigger_time)
        trigger_start_offset = trigger_time - self.data.start_time
        antenna_pattern = self.antenna_pattern(ra, dec, psi, gmst)
        time_shift = self.delay_from_geocenter(ra, dec, gmst)
        time_shift += trigger_start_offset + params["t_c"]

        h_detector = jax.tree_util.tree_map(
            lambda h, antenna: h * antenna,
            h_sky,
            antenna_pattern,
        )
        projected_strain = jnp.sum(
            jnp.stack(jax.tree_util.tree_leaves(h_detector)), axis=0
        )

        phase_shift = jnp.exp(-2j * jnp.pi * frequency * time_shift)
        return projected_strain * phase_shift

    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
    ) -> Array:
        """Modulate the waveform in the sky frame by the detector response in the time domain.

        Args:
            time: Array of time samples.
            h_sky: Dictionary mapping polarization names to time-domain waveforms.
            params: Dictionary of source parameters.

        Returns:
            Array of detector response in time domain.
        """
        raise NotImplementedError

    def delay_from_geocenter(
        self, ra: FloatScalar, dec: FloatScalar, gmst: FloatScalar
    ) -> FloatScalar:
        """Calculate time delay between two detectors in geocentric coordinates.

        Based on XLALArrivaTimeDiff in TimeDelay.c
        https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_delay__h.html

        Args:
            ra (Float): Right ascension of the source in radians.
            dec (Float): Declination of the source in radians.
            gmst (Float): Greenwich mean sidereal time in radians.

        Returns:
            Float: Time delay from Earth center in seconds.
        """
        delta_d = -self.vertex
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec
        omega = jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )
        return jnp.einsum("i...,i->...", omega, delta_d) / C_SI

    def antenna_pattern(
        self,
        ra: FloatScalar,
        dec: FloatScalar,
        psi: FloatScalar,
        gmst: FloatScalar,
    ) -> dict[str, Complex]:
        """Compute antenna patterns for polarizations at specified sky location.

        In the long-wavelength approximation, the antenna pattern for a
        given polarization is the dyadic product between the detector
        tensor and the corresponding polarization tensor.

        Args:
            ra (Float): Source right ascension in radians.
            dec (Float): Source declination in radians.
            psi (Float): Source polarization angle in radians.
            gmst (Float): Greenwich mean sidereal time (GMST) in radians.

        Returns:
            dict[str, Complex]: Dictionary mapping polarization names to their antenna pattern values.
        """
        detector_tensor = self.tensor

        antenna_patterns = {}
        for polarization in self.polarization_mode:
            wave_tensor = polarization.tensor_from_sky(ra, dec, psi, gmst)
            antenna_patterns[polarization.name] = jnp.einsum(
                "ij,ij...->...", detector_tensor, wave_tensor
            )

        return antenna_patterns

    # ------------------------------------------------------------------
    # Data and PSD
    # ------------------------------------------------------------------

    def set_data(self, data: Data | Array, **kws) -> None:
        """Replace the data of the detector.

        The PSD is interpolated onto the frequencies of the new data.

        Args:
            data (Data | Array): Data to be added to the detector, either as a `Data` object
                or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `Data.from_td`.
        """
        self._check_mutable()
        new = data if isinstance(data, Data) else Data.from_td(data, **kws)
        psd = self._matched_psd(new, self.psd)
        self.data, self.psd = new, psd

    def set_psd(self, psd: PowerSpectrum | Array, **kws) -> None:
        """Replace the PSD of the detector.

        It is interpolated onto the frequencies of the data.

        Args:
            psd (PowerSpectrum | Array): PSD to be added to the detector, either as a `PowerSpectrum`
                object or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `PowerSpectrum` constructor.
        """
        self._check_mutable()
        new = psd if isinstance(psd, PowerSpectrum) else PowerSpectrum(psd, **kws)
        self.psd = self._matched_psd(self.data, new)

    @jaxtyped(typechecker=typechecker)
    def load_and_set_psd(self, psd_file: str = "", asd_file: str = "") -> PowerSpectrum:
        """Load power spectral density (PSD) from file or default GWTC-2 catalog,
            and set it to the detector.

        Supported formats: .npz, .txt, .dat, .csv.
        Pass ``asd_file`` (or ``is_asd=True`` via :meth:`PowerSpectrum.from_file`)
        when the file contains amplitude spectral density values
        ($\\mathrm{Hz}^{-1/2}$); they are squared internally to produce a PSD.

        Args:
            psd_file (str, optional): Path to a PSD file ($\\mathrm{Hz}^{-1}$). If empty, uses GWTC-2 ASD.
            asd_file (str, optional): Path to an ASD file ($\\mathrm{Hz}^{-1/2}$). Values are squared.

        Returns:
            PowerSpectrum: The loaded PSD, already set on the detector.

        Raises:
            ValueError: If both ``psd_file`` and ``asd_file`` are given.
        """
        self._check_mutable()
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if psd_file and asd_file:
            raise ValueError(
                "psd_file and asd_file are mutually exclusive; set at most one."
            )
        if psd_file:
            _loaded_psd = PowerSpectrum.from_file(psd_file, is_asd=False)
        elif asd_file:
            _loaded_psd = PowerSpectrum.from_file(asd_file, is_asd=True)
        else:
            logger.info("Grabbing GWTC-2 PSD for " + self.name)
            url = asd_file_dict[self.name]
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            fd, tmp_file_name = tempfile.mkstemp(
                suffix=".txt", prefix=f"jim_asd_{self.name}_"
            )
            try:
                with os.fdopen(fd, "wb") as _fh:
                    _fh.write(response.content)
                _loaded_psd = PowerSpectrum.from_file(tmp_file_name, is_asd=True)
            finally:
                os.unlink(tmp_file_name)
        _loaded_psd.name = f"{self.name}_psd"
        self.set_psd(_loaded_psd)
        return self.psd

    # ------------------------------------------------------------------
    # Simulated data
    # ------------------------------------------------------------------

    def _zero_data(
        self,
        *,
        duration: float,
        sampling_frequency: float,
        trigger_time: float,
        f_min: float,
        f_max: float,
        post_trigger_duration: float,
    ) -> Data:
        """Return a segment of zeros."""
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if not (
            jnp.isfinite(duration)
            and jnp.isfinite(sampling_frequency)
            and duration > 0
            and sampling_frequency > 0
        ):
            raise ValueError(
                "duration and sampling_frequency must be positive and finite, got "
                f"{duration} s and {sampling_frequency} Hz."
            )
        if not 0 <= post_trigger_duration <= duration:
            raise ValueError(
                f"post_trigger_duration must be between 0 and duration ({duration} s) "
                f"so that the trigger lies inside the segment, got {post_trigger_duration}."
            )
        sample_count = duration * sampling_frequency
        if not jnp.isfinite(sample_count):
            raise ValueError("duration times sampling_frequency must be finite.")
        n_times = round(sample_count)
        effective_duration = n_times / sampling_frequency
        if n_times < 2:
            raise ValueError("The segment must contain at least two time samples.")
        if post_trigger_duration > effective_duration:
            raise ValueError(
                "post_trigger_duration must be between 0 and the rounded segment "
                f"duration ({effective_duration} s), got {post_trigger_duration}."
            )
        if not jnp.isfinite(trigger_time):
            raise ValueError("trigger_time must be finite.")
        data = Data.from_td(
            name=f"{self.name}_zero",
            td=jnp.zeros(n_times),
            delta_t=1 / sampling_frequency,
            window=jnp.ones(n_times),
            start_time=trigger_time - effective_duration + post_trigger_duration,
            trigger_time=trigger_time,
        )
        data.frequency_bounds = (f_min, f_max)
        return data

    def set_zero_noise(
        self,
        *,
        duration: float,
        sampling_frequency: float,
        trigger_time: float,
        f_min: float,
        f_max: float,
        post_trigger_duration: float = DEFAULT_POST_TRIGGER_DURATION,
    ) -> None:
        """Replace the data by a segment of zeros.

        The segment ends ``post_trigger_duration`` seconds after the trigger.
        ``[f_min, f_max]`` becomes the frequency range of the data: signals injected
        later are cut to it. The trigger time is stored in the data, where
        ``inject_signal`` reads it.

        Args:
            duration (float): Duration of the data segment in seconds.
            sampling_frequency (float): Sampling frequency in Hz.
            trigger_time (float): GPS time of the event trigger.
            f_min (float): Minimum frequency of the data in Hz.
            f_max (float): Maximum frequency of the data in Hz.
            post_trigger_duration (float, optional): Time in seconds from the
                trigger to the end of the segment. Defaults to 2.0.

        Raises:
            ValueError: If ``duration`` or ``sampling_frequency`` is not positive, if
                ``post_trigger_duration`` puts the trigger outside the segment, or if
                f_min and f_max are not a valid frequency range of the data (see
                ``Data.frequency_bounds``).
                The detector is unchanged then.
        """
        self._check_mutable()
        data = self._zero_data(
            duration=duration,
            sampling_frequency=sampling_frequency,
            trigger_time=trigger_time,
            f_min=f_min,
            f_max=f_max,
            post_trigger_duration=post_trigger_duration,
        )
        psd = self._matched_psd(data, self.psd)
        self.data, self.psd = data, psd

    def simulate_and_set_noise(
        self,
        *,
        duration: float,
        sampling_frequency: float,
        trigger_time: float,
        f_min: float,
        f_max: float,
        rng_key: Key,
        post_trigger_duration: float = DEFAULT_POST_TRIGGER_DURATION,
    ) -> None:
        """Replace the data by noise drawn from the PSD.

        The noise is cut to ``[f_min, f_max]``, which becomes the frequency range of
        the data. It depends only on ``rng_key`` and the name of the detector, so
        every detector gets its own noise from one key. The segment ends
        ``post_trigger_duration`` seconds after the trigger.

        Note: The power spectral density must be set beforehand, and it must be
        finite and positive over ``[f_min, f_max]``.

        Args:
            duration (float): Duration of the data segment in seconds.
            sampling_frequency (float): Sampling frequency in Hz.
            trigger_time (float): GPS time of the event trigger.
            f_min (float): Minimum frequency of the data in Hz.
            f_max (float): Maximum frequency of the data in Hz.
            rng_key (Key): PRNG key for the noise, such as ``jax.random.key(0)``.
            post_trigger_duration (float, optional): Time in seconds from the
                trigger to the end of the segment. Defaults to 2.0.

        Raises:
            ValueError: If no PSD is set, if it is not finite and positive over the
                frequency range of the data, or for any reason given at ``set_zero_noise``. The
                detector is unchanged then.
        """
        self._check_mutable()
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if self.psd.is_empty:
            raise ValueError(
                f"No PSD is set on detector {self.name}. Call set_psd() or "
                "load_and_set_psd() before simulate_and_set_noise."
            )
        data = self._zero_data(
            duration=duration,
            sampling_frequency=sampling_frequency,
            trigger_time=trigger_time,
            f_min=f_min,
            f_max=f_max,
            post_trigger_duration=post_trigger_duration,
        )
        psd = self._matched_psd(data, self.psd)
        self._check_psd_in_range(psd, *data.frequency_bounds)
        # Derive a unique noise key for this detector based on its name
        noise_key = jax.random.fold_in(rng_key, zlib.crc32(self.name.encode("utf-8")))
        noise_psd = PowerSpectrum(
            jnp.where(data.frequency_mask, psd.values, 1.0), psd.frequencies, psd.name
        )
        noise = noise_psd.simulate_data(noise_key)
        noisy = data.add_fd(jnp.where(data.frequency_mask, noise, 0.0 + 0.0j))
        noisy.name = f"{self.name}_noise"
        self.data, self.psd = noisy, psd

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def inject_signal(
        self,
        waveform_model,
        parameters: dict[str, float],
        allow_wraparound: bool = False,
    ) -> tuple[FloatScalar, ComplexScalar]:
        """Add a signal to the data of the detector.

        The signal is cut to the frequency range of the data and added in the
        frequency domain to whatever the detector holds: zero data, noise, real data
        or earlier signals. Call it again to add more signals.

        Note: The detector needs a PSD and data beforehand; ``set_zero_noise`` and
        ``simulate_and_set_noise`` create them. The trigger time is that of the data
        (``Data.trigger_time``), by default 2 s before their end. The frequency range
        of the data must start above 0 Hz: set ``data.frequency_bounds`` for real
        data.

        Args:
            waveform_model: The frequency-domain waveform model to be injected.
            parameters (dict): Likelihood-space source parameters: those of the
                waveform model (``waveform_model.parameter_names``) and ``ra``,
                ``dec``, ``psi`` and ``t_c``.
            allow_wraparound (bool): The waveform has no time axis, so a signal that
                does not fit the segment wraps around in it. By default that is an
                error; with True it is a warning. A conservative inspiral-duration
                estimate and merger margins check compact-binary signals requiring
                ``M_c`` and ``eta``. Other models need an explicit override because
                their duration cannot be estimated this way.
                Defaults to False.

        Returns:
            tuple: The optimal SNR of the signal, and its matched-filter SNR against
                the data including the signal, noise and earlier signals.

        Raises:
            ValueError: If the detector has no PSD or no data, if the PSD is not
                finite and positive over the frequency range of the data, if the
                range does not start above 0 Hz or holds fewer than two frequencies,
                if the trigger time of the data is not inside them, if a parameter is
                missing or ``eta`` is not in (0, 0.25], if the signal has no finite,
                positive signal-to-noise power, if a time-domain model is supplied,
                or if its estimated duration does not fit the segment or cannot be
                estimated. The detector is unchanged then.
        """
        self._check_mutable()
        # Check everything first, so a bad call leaves the detector untouched.
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if getattr(waveform_model, "waveform_metadata", {}).get("domain") == "TD":
            raise ValueError(
                "inject_signal requires a frequency-domain waveform model."
            )
        required = [*waveform_model.parameter_names, "ra", "dec", "psi", "t_c"]
        missing = [key for key in required if key not in parameters]
        if missing:
            raise ValueError(f"parameters is missing {missing}.")
        eta = parameters.get("eta")
        if eta is not None and not 0 < eta <= 0.25 + 1e-9:
            raise ValueError(f"eta must be in (0, 0.25], got {eta}.")
        if self.psd.is_empty:
            raise ValueError(
                f"No PSD is set on detector {self.name}. Call set_psd() or "
                "load_and_set_psd() before inject_signal."
            )
        if self.data.is_empty:
            raise ValueError(
                f"No data is set on detector {self.name}. Call set_zero_noise(), "
                "simulate_and_set_noise() or set_data() before inject_signal."
            )
        parameters = parameters.copy()
        data = copy.copy(self.data)
        f_min, f_max = data.frequency_bounds
        if not f_min > 0:
            raise ValueError(
                f"The frequency range of the data of detector {self.name} starts at "
                f"{f_min} Hz, and waveforms are not defined at 0 Hz. Set "
                "data.frequency_bounds = (f_min, f_max) first."
            )
        trigger_time = data.trigger_time
        start_time = float(data.start_time)
        end_time = data.end_time
        if not start_time <= trigger_time <= end_time:
            raise ValueError(
                f"The trigger time {trigger_time} is outside the data of detector "
                f"'{self.name}', which span [{start_time}, {end_time}]. Check "
                "data.trigger_time and data.start_time."
            )

        # Read the FFT on a copy: failures must not freeze the original data's window.
        frequencies, fd_data, psd = self._slice_data_to_frequency_range(
            data, f_min, f_max
        )

        # 1. Compute the projected strain from parameters
        grid = data.frequencies
        polarisations = waveform_model(grid, parameters)
        projected_strain = self.fd_response(grid, polarisations, parameters)
        mask = data.frequency_mask
        masked_signal = projected_strain[mask]
        invalid = ~jnp.isfinite(masked_signal)
        if bool(jnp.any(invalid)):
            raise ValueError(
                f"The signal is not finite at {int(jnp.sum(invalid))} of "
                f"{len(masked_signal)} frequencies in [{f_min}, {f_max}] Hz, the first "
                f"at {float(grid[mask][invalid][0])} Hz. Check the source parameters."
            )
        projected_strain = data.normalize_fd(
            jnp.where(mask, projected_strain, 0.0 + 0.0j)
        )
        masked_signal = projected_strain[mask]
        df = frequencies[1] - frequencies[0]
        optimal_snr_squared = inner_product(masked_signal, masked_signal, psd, df)
        if not (bool(jnp.isfinite(optimal_snr_squared)) and optimal_snr_squared > 0):
            raise ValueError(
                "The signal must have finite, positive signal-to-noise power."
            )
        self._check_signal_duration(
            waveform_model, parameters, trigger_time, allow_wraparound
        )

        # 2. Add it to the copied data
        injected = data.add_fd(projected_strain)
        injected.name = f"{self.name}_injected"

        # 3. Measure the SNRs of the injected signal in the frequency range of the data
        optimal_snr = optimal_snr_squared**0.5
        match_filtered_snr = (
            complex_inner_product(masked_signal, fd_data + masked_signal, psd, df)
            / optimal_snr
        )
        self.data = injected

        logger.info(f"For detector {self.name}, the injected signal has:")
        logger.info(f"  - Optimal SNR: {optimal_snr:.4f}")
        logger.info(f"  - Match filtered SNR: {match_filtered_snr:.4f}")
        return optimal_snr, match_filtered_snr

    def _check_signal_duration(
        self,
        waveform_model,
        params: dict,
        trigger_time: float,
        allow_wraparound: bool,
    ) -> None:
        """Check a rough, conservative CBC duration without evaluating the waveform.

        Use LAL's positive 2PN chirp-time terms with maximal physical spin, then
        add rounded merger/ringdown bounds, one low-frequency cycle and the maximum
        geocenter delay. This estimates physical duration, not Fourier-cut tails.
        See ``XLALSimInspiralChirpTimeBound`` in LALSimInspiral.c.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if not {"M_c", "eta"}.issubset(waveform_model.parameter_names):
            message = (
                f"Cannot estimate the signal duration for detector {self.name}: "
                "the waveform model does not declare M_c and eta. Check the "
                "duration yourself and pass allow_wraparound=True to inject it."
            )
        else:
            chirp_mass = float(params["M_c"])
            if not (jnp.isfinite(chirp_mass) and chirp_mass > 0):
                raise ValueError("M_c must be finite and positive.")
            eta = float(params["eta"])
            mass_seconds = chirp_mass * MTSUN / eta**0.6
            f_start = self.data.frequency_bounds[0]
            # Current Ripple higher-mode models include harmonics through m=4;
            # their earliest orbital frequency in the frequency range of the data is
            # half that of m=2.
            if type(waveform_model).__name__ in {
                "IMRPhenomHM",
                "IMRPhenomXHM",
                "IMRPhenomXPHM",
            }:
                f_start /= 2
            v = (jnp.pi * mass_seconds * f_start) ** (1 / 3)
            newtonian = 5 * mass_seconds / (256 * eta * v**8)
            chirp_time = newtonian * (
                1
                + (743 / 252 + 11 * eta / 3) * v**2
                + (226 / 15) * v**3
                + (3058673 / 508032 + 5429 * eta / 504 + 617 * eta**2 / 72) * v**4
            )
            padding = 1 / f_start + EARTH_RADIUS_LIGHT_S
            before = chirp_time + 200 * mass_seconds + padding
            after = 1000 * mass_seconds + padding
            peak = trigger_time - self.data.start_time + float(params["t_c"])
            if peak >= before and peak + after <= float(self.data.duration):
                return
            message = (
                f"The estimated signal duration does not fit the segment of detector "
                f"{self.name}: conservatively allow {before:.2f} s before and "
                f"{after:.2f} s after the peak, which is {peak:.2f} s into the "
                f"{float(self.data.duration):g} s segment. Move the signal (t_c or "
                "the trigger time), lengthen the segment, or pass "
                "allow_wraparound=True to inject it anyway."
            )
        if not allow_wraparound:
            raise ValueError(message)
        logger.warning(message)

    # ------------------------------------------------------------------
    # Whitening
    # ------------------------------------------------------------------

    def get_whitened_frequency_domain_strain(
        self, frequency_series: Complex[Array, " n_freq"]
    ) -> Complex[Array, " n_freq"]:
        """Get the whitened frequency-domain strain.
        Args:
            frequency_series (Complex[Array, "n_freq"]): Array of frequency domain data/signal.
        Returns:
            Complex[Array, "n_freq"]: Whitened frequency-domain strain.
        """
        scaled_asd = jnp.sqrt(self.psd.values * self.data.duration / 4)
        return (frequency_series / scaled_asd) * self.data.frequency_mask

    def whitened_frequency_to_time_domain_strain(
        self, whitened_frequency_series: Complex[Array, " n_time // 2 + 1"]
    ) -> Float[Array, " n_time"]:
        """Get the whitened frequency-domain strain.
        Args:
            whitened_frequency_series (Complex[Array, "n_time // 2 + 1"]):
                Array of whitened frequency domain data/signal.
        Returns:
            Float[Array, "n_time"]: Whitened time-domain strain/signal.
        """
        mask = self.data.frequency_mask
        freq_mask_ratio = len(mask) / jnp.sqrt(jnp.sum(mask))
        return (
            jnp.fft.irfft(whitened_frequency_series, n=self.data.n_time)
            * freq_mask_ratio
        )

    @property
    def whitened_frequency_domain_data(self) -> Complex[Array, " n_sample"]:
        """Get the whitened frequency-domain data.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.

        Returns:
            Float[Array, "n_sample"]: Whitened frequency-domain data.
        """

        return self.get_whitened_frequency_domain_strain(self.data.fd)

    @property
    def whitened_time_domain_data(self) -> Float[Array, " n_sample"]:
        """Get the whitened time-domain data.

        Args:
            time (Float[Array, "n_sample"]): Array of time samples.

        Returns:
            Float[Array, "n_sample"]: Whitened time-domain data.
        """
        return self.whitened_frequency_to_time_domain_strain(
            self.whitened_frequency_domain_data
        )


# ----------------------------------------------------------------------
# Detector presets
# ----------------------------------------------------------------------


def get_H1() -> GroundBased2G:
    """Return a [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instance for LIGO Hanford (H1)."""
    return GroundBased2G(
        "H1",
        latitude=(46 + 27.0 / 60 + 18.528 / 3600) * DEG_TO_RAD,
        longitude=-(119 + 24.0 / 60 + 27.5657 / 3600) * DEG_TO_RAD,
        xarm_azimuth=125.9994 * DEG_TO_RAD,
        yarm_azimuth=215.9994 * DEG_TO_RAD,
        xarm_tilt=-6.195e-4,
        yarm_tilt=1.25e-5,
        elevation=142.554,
        modes="pc",
    )


def get_L1() -> GroundBased2G:
    """Return a [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instance for LIGO Livingston (L1)."""
    return GroundBased2G(
        "L1",
        latitude=(30 + 33.0 / 60 + 46.4196 / 3600) * DEG_TO_RAD,
        longitude=-(90 + 46.0 / 60 + 27.2654 / 3600) * DEG_TO_RAD,
        xarm_azimuth=197.7165 * DEG_TO_RAD,
        yarm_azimuth=287.7165 * DEG_TO_RAD,
        xarm_tilt=-3.121e-4,
        yarm_tilt=-6.107e-4,
        elevation=-6.574,
        modes="pc",
    )


def get_V1() -> GroundBased2G:
    """Return a [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instance for Virgo (V1)."""
    return GroundBased2G(
        "V1",
        latitude=(43 + 37.0 / 60 + 53.0921 / 3600) * DEG_TO_RAD,
        longitude=(10 + 30.0 / 60 + 16.1878 / 3600) * DEG_TO_RAD,
        xarm_azimuth=70.5674 * DEG_TO_RAD,
        yarm_azimuth=160.5674 * DEG_TO_RAD,
        xarm_tilt=0,
        yarm_tilt=0,
        elevation=51.884,
        modes="pc",
    )


def get_ET() -> list[GroundBased2G]:
    """Return a list of three [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instances for Einstein Telescope (ET).

    ET is modelled as a triangle of three interferometers at adjacent vertices,
    with arms rotated by 120° relative to each other. Vertex positions are
    propagated using the spherical forward-azimuth (haversine) formula with a
    latitude-dependent Earth radius derived from the WGS-84 ellipsoid.
    """
    name = "ET"
    latitude = (43 + 37.0 / 60 + 53.0921 / 3600) * DEG_TO_RAD
    longitude = (10 + 30.0 / 60 + 16.1878 / 3600) * DEG_TO_RAD
    xarm_azimuth = 70.5674 * DEG_TO_RAD
    yarm_azimuth = 130.5674 * DEG_TO_RAD
    xarm_tilt = 0
    yarm_tilt = 0
    elevation = 51.884
    length: float = 1e4  # arm length in metres

    a = EARTH_SEMI_MAJOR_AXIS / 1e3  # Numerical instability avoidance
    b = EARTH_SEMI_MINOR_AXIS / 1e3
    earth_approx_radius = (
        a
        * b
        / (jnp.sqrt(a**2 * jnp.sin(latitude) ** 2 + b**2 * jnp.cos(latitude) ** 2))
    )
    earth_approx_radius *= 1e3

    # Navigation bearing (clockwise from North) corresponding to xarm_azimuth
    # (counter-clockwise from East): brng = pi/2 - azimuth.
    # Both brng and the arm azimuths are incremented by 240° (4π/3) per vertex.
    brng = jnp.pi / 2 - xarm_azimuth

    ifos = []
    for i in range(3):
        ifos.append(
            GroundBased2G(
                f"{name}{i + 1}",
                latitude=float(latitude),
                longitude=float(longitude),
                xarm_azimuth=float(xarm_azimuth),
                yarm_azimuth=float(yarm_azimuth),
                elevation=elevation,
                xarm_tilt=xarm_tilt,
                yarm_tilt=yarm_tilt,
            )
        )
        # Propagate to next vertex using the spherical forward-azimuth formula.
        # Coordinate update must precede arm rotation (uses current bearing).
        d = length / earth_approx_radius
        phi1 = latitude
        phi2 = jnp.arcsin(
            jnp.sin(phi1) * jnp.cos(d) + jnp.cos(phi1) * jnp.sin(d) * jnp.cos(brng)
        )
        longitude = longitude + jnp.arctan2(
            jnp.sin(brng) * jnp.sin(d) * jnp.cos(phi1),
            jnp.cos(d) - jnp.sin(phi1) * jnp.sin(phi2),
        )
        latitude = phi2
        # Rotate arms and bearing for the next detector vertex (240°, i.e. 4π/3, per vertex)
        xarm_azimuth += (4 / 3) * jnp.pi
        yarm_azimuth += (4 / 3) * jnp.pi
        brng += (4 / 3) * jnp.pi
    return ifos


def get_CE() -> GroundBased2G:
    """Return a [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instance for Cosmic Explorer (CE).

    CE shares the LIGO Hanford site geometry.
    """
    return GroundBased2G(
        "CE",
        latitude=(46 + 27.0 / 60 + 18.528 / 3600) * DEG_TO_RAD,
        longitude=-(119 + 24.0 / 60 + 27.5657 / 3600) * DEG_TO_RAD,
        xarm_azimuth=125.9994 * DEG_TO_RAD,
        yarm_azimuth=215.994 * DEG_TO_RAD,
        xarm_tilt=-6.195e-4,
        yarm_tilt=1.25e-5,
        elevation=142.554,
        modes="pc",
    )


def get_detector_preset() -> dict[str, GroundBased2G | list[GroundBased2G]]:
    """Return a dictionary of pre-configured detector instances.

    Returns:
        dict: Mapping of detector name to detector object(s).
            Keys are ``"H1"``, ``"L1"``, ``"V1"``, ``"CE"`` (single
            [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G]) and ``"ET"`` (list of three).
    """
    return {
        "H1": get_H1(),
        "L1": get_L1(),
        "V1": get_V1(),
        "ET": get_ET(),
        "CE": get_CE(),
    }
