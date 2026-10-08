import logging
import os
import tempfile
import zlib
from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import requests
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Complex, Float, Key, jaxtyped

from jimgw.core.constants import (
    C_SI,
    DEG_TO_RAD,
    EARTH_ROTATION_RATE,
    EARTH_SEMI_MAJOR_AXIS,
    EARTH_SEMI_MINOR_AXIS,
    MTSUN,
)
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.polarization import Polarization
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.utils import complex_inner_product, inner_product
from jimgw.typing import FloatLike, FloatScalar

logger = logging.getLogger(__name__)

# TODO: Need to expand this list. Currently it is only O3.
asd_file_dict = {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",
}


class Detector(ABC):
    """Base class for all detectors.

    Attributes:
        name (str): Name of the detector.
        data (Data): Detector data object.
        psd (PowerSpectrum): Power spectral density object.
        frequency_bounds (tuple[float, float]): Lower and upper frequency bounds.
    """

    name: str

    # NOTE: for some detectors (e.g. LISA, ET) data could be a list of Data
    # objects so this might be worth revisiting
    data: Data
    psd: PowerSpectrum

    frequency_bounds: tuple[float, float] = (0.0, jnp.inf)

    _sliced_frequencies: Float[Array, " n_sample"] = jnp.array([])
    _sliced_fd_data: Float[Array, " n_sample"] = jnp.array([])
    _sliced_psd: Float[Array, " n_sample"] = jnp.array([])

    @property
    def start_time(self) -> float:
        """GPS start time of the data segment."""
        return self.data.start_time

    @property
    def times(self) -> Float[Array, " n_sample"]:
        return self.data.times

    @property
    def frequencies(self) -> Float[Array, " n_sample"]:
        return self.data.frequencies

    @property
    def duration(self) -> FloatLike:
        return self.data.duration

    @property
    def frequency_mask(self) -> Bool[Array, " n_sample"]:
        f_min, f_max = self.frequency_bounds
        return (f_min <= self.frequencies) & (self.frequencies <= f_max)

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
                - trigger_time (Float): The trigger time in sec
                - t_c (Float): The difference between peak time and trigger time in sec
                - gmst (Float): The greenwich mean sidereal time at the trigger time in radian

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

    def set_frequency_bounds(
        self, f_min: Optional[float] = None, f_max: Optional[float] = None
    ) -> None:
        """Set the frequency bounds for the detector.
        This also set the sliced frequencies, data and psd.

        Args:
            f_min: Minimum frequency.
            f_max: Maximum frequency.
        """
        bounds = list(self.frequency_bounds)
        if f_min is not None:
            bounds[0] = f_min
        if f_max is not None:
            bounds[1] = f_max
        self.frequency_bounds = (bounds[0], bounds[1])

        # Compute sliced frequencies, data and psd.
        data, freqs_1 = self.data.frequency_slice(*self.frequency_bounds)
        psd, freqs_2 = self.psd.frequency_slice(*self.frequency_bounds)

        assert jnp.array_equal(freqs_1, freqs_2), (
            f"The {self.name} data and PSD must have same frequencies"
        )

        self._sliced_frequencies = freqs_1
        self._sliced_fd_data = data
        self._sliced_psd = psd

    def clear_data_and_psd(self) -> None:
        """Clear the data and PSD of the detector."""
        self.data = Data()
        self.psd = PowerSpectrum()
        self.frequency_bounds = (0.0, jnp.inf)
        self._sliced_frequencies = jnp.array([])
        self._sliced_fd_data = jnp.array([])
        self._sliced_psd = jnp.array([])
        self.optimal_snr = None
        self.match_filtered_snr = None

    @property
    def sliced_frequencies(self) -> Float[Array, " n_freq"]:
        """Get frequency-domain data slice based on frequency bounds.

        Returns:
            Float[Array, "n_sample"]: Sliced frequency-domain data.
            Float[Array, "n_sample"]: Frequency array.
        """
        return self._sliced_frequencies

    @property
    def sliced_fd_data(self) -> Complex[Array, " n_freq"]:
        """Get frequency-domain data slice based on frequency bounds.

        Returns:
            Complex[Array, "n_freq"]: Sliced frequency-domain data.
        """
        return self._sliced_fd_data

    @property
    def sliced_psd(self) -> Float[Array, " n_freq"]:
        """Get PSD slice based on frequency bounds.

        Returns:
            Float[Array, "n_freq"]: Sliced power spectral density.
        """
        return self._sliced_psd

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
        data (Data): Array of Fourier-domain strain data.
        psd (PowerSpectrum): Power spectral density object.
    """

    polarization_mode: list[Polarization]
    data: Data
    psd: PowerSpectrum

    latitude: float = 0
    longitude: float = 0
    xarm_azimuth: float = 0
    yarm_azimuth: float = 0
    xarm_tilt: float = 0
    yarm_tilt: float = 0
    elevation: float = 0

    optimal_snr: Optional[FloatScalar] = None
    match_filtered_snr: Optional[Complex] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

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
                - trigger_time (Float): The trigger time in sec
                - t_c (Float): The difference between peak time and trigger time in sec
                - gmst (Float): The greenwich mean sidereal time at the trigger time in radian

        Returns:
            Array: Complex strain measured by the detector in frequency domain, obtained by
                  combining the antenna patterns for each polarization mode.
        """
        ra, dec, psi, gmst = params["ra"], params["dec"], params["psi"], params["gmst"]
        antenna_pattern = self.antenna_pattern(ra, dec, psi, gmst)
        time_shift = self.delay_from_geocenter(ra, dec, gmst)
        return self._project(frequency, h_sky, antenna_pattern, time_shift, params)

    def _project(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        antenna_pattern: dict[str, Complex],
        geocenter_delay: Float,
        params: dict[str, Float],
    ) -> Complex[Array, " n_sample"]:
        """Combine the polarizations with the antenna pattern and shift them in time.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.
            h_sky (dict[str, Float[Array, "n_sample"]]): Sky-frame polarizations.
            antenna_pattern (dict[str, Complex]): Antenna pattern per polarization,
                either a scalar or one value per frequency sample.
            geocenter_delay (Float): Delay from the geocenter in seconds, either a
                scalar or one value per frequency sample.
            params (dict[str, Float]): Source parameters containing
                ``trigger_time`` and ``t_c``.

        Returns:
            Complex[Array, "n_sample"]: Complex strain measured by the detector.
        """
        time_shift = geocenter_delay + (
            params["trigger_time"] - self.start_time + params["t_c"]
        )

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
        omega = self._source_direction(ra, dec, gmst)
        return jnp.einsum("i...,i->...", omega, delta_d) / C_SI

    @staticmethod
    def _source_direction(
        ra: FloatScalar, dec: FloatScalar, gmst: FloatScalar
    ) -> Float[Array, "3 ..."]:
        """Unit vector pointing from the geocenter to the source, in Earth-fixed coordinates.

        Args:
            ra (Float): Right ascension of the source in radians.
            dec (Float): Declination of the source in radians.
            gmst (Float): Greenwich mean sidereal time in radians; an array
                gives one direction per element.

        Returns:
            Float[Array, "3 ..."]: Direction to the source.
        """
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi, theta = jnp.broadcast_arrays(ra - gmst, jnp.pi / 2 - dec)
        return jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )

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
        """
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

    def _equal_data_psd_frequencies(self) -> Bool:
        """Check if the frequencies of the data and PSD match.
        A helper function for `set_data` and `set_psd`.

        Return:
            Bool: True if the frequencies match, False otherwise.
        """
        if self.psd.is_empty or self.data.is_empty:
            # In this case, we simply skip the check
            return True
        if self.psd.n_freq != self.data.n_freq:
            # Cannot proceed comparison, needs interpolation
            return False
        return (self.psd.frequencies == self.data.frequencies).all()

    def set_data(self, data: Data | Array, **kws) -> None:
        """Add data to the detector.

        Args:
            data (Data | Array): Data to be added to the detector, either as a `Data` object
                or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `Data` constructor.

        Returns:
            None
        """
        if isinstance(data, Data):
            self.data = data
        else:
            self.data = Data(data, **kws)
        # Assert PSD frequencies agree with data
        if not ((self.psd is None) or self._equal_data_psd_frequencies()):
            self.psd = self.psd.interpolate(self.data.frequencies)

    def set_psd(self, psd: PowerSpectrum | Array, **kws) -> None:
        """Add PSD to the detector.

        Args:
            psd (PowerSpectrum | Array): PSD to be added to the detector, either as a `PowerSpectrum`
                object or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `PowerSpectrum` constructor.

        Returns:
            None
        """
        if isinstance(psd, PowerSpectrum):
            self.psd = psd
        else:
            # not clear if we want to support this
            self.psd = PowerSpectrum(psd, **kws)
        # Assert PSD frequencies agree with data frequencies
        if not ((self.data is None) or self._equal_data_psd_frequencies()):
            self.psd = self.psd.interpolate(self.data.frequencies)

    def inject_signal(
        self,
        duration: float,
        sampling_frequency: float,
        trigger_time: float,
        waveform_model,
        parameters: dict[str, float],
        f_min: float,
        f_max: float,
        start_time: Optional[float] = None,
        zero_noise: bool = False,
        rng_key: Optional[Key] = None,
    ) -> None:
        """Inject a signal into the detector data.

        Note: The power spectral density must be set beforehand.

        Args:
            duration (float): Duration of the data segment in seconds.
            sampling_frequency (float): Sampling frequency in Hz.
            trigger_time (float): GPS time of the event trigger. Used to stamp
                ``trigger_time`` and derive ``gmst`` for the waveform projection,
                mirroring the behavior of ``TransientLikelihoodFD``.
            waveform_model: The waveform model to be injected.
            parameters (dict): Dictionary of likelihood-space source parameters.
            f_min (float): Minimum frequency in Hz. The waveform is zeroed below
                this frequency.
            f_max (float): Maximum frequency in Hz. Should be set to the same
                value used in the likelihood.
            start_time (Optional[float], optional): GPS start time of the
                data buffer in seconds. If None, defaults to
                ``trigger_time - duration + 2.0`` (2 s of data after the trigger).
                Defaults to None.
            zero_noise (bool, optional): If True, inject into zero noise.
                Defaults to False.
            rng_key (Optional[Key], optional): PRNG key for the injected noise.
                Required unless ``zero_noise`` is True.

        Returns:
            None

        Raises:
            ValueError: If no PSD is set on the detector, or if ``zero_noise`` is
                False and ``rng_key`` is None.
        """
        # Check the inputs first, so a bad call leaves the detector untouched.
        if self.psd.is_empty:
            raise ValueError(
                f"No PSD is set on detector {self.name}. Call set_psd() or "
                "load_and_set_psd() before inject_signal."
            )
        noise_key: Optional[Key] = None
        if zero_noise:
            if rng_key is not None:
                logger.warning(
                    "rng_key is ignored because zero_noise=True: no noise is drawn."
                )
        else:
            if rng_key is None:
                raise ValueError(
                    "rng_key is required when zero_noise=False."
                    "Pass zero_noise=True for a noiseless injection."
                )
            # Derive a unique noise key for this detector based on its name
            noise_key = jax.random.fold_in(
                rng_key, zlib.crc32(self.name.encode("utf-8"))
            )

        # Derive start_time if not provided
        if start_time is None:
            start_time = trigger_time - duration + 2.0
            logger.info(
                "start_time not provided. Defaulting to trigger_time - duration + 2.0 = %.3f s.",
                start_time,
            )

        # Make a copy of the parameters to avoid modifying the original dictionary
        params = parameters.copy()

        # Stamp trigger_time and gmst — mirrors TransientLikelihoodFD.evaluate()
        params["trigger_time"] = float(trigger_time)
        params["gmst"] = float(compute_gmst(trigger_time))

        # 1. Set empty data to initialize the detector
        n_times = int(jnp.round(duration * sampling_frequency))
        self.set_data(
            Data(
                name=f"{self.name}_empty",
                td=jnp.zeros(n_times),
                delta_t=1 / sampling_frequency,
                start_time=start_time,
            )
        )

        # Set frequency bounds before evaluating the waveform
        self.set_frequency_bounds(f_min, f_max)

        # 2. Compute the projected strain from parameters
        polarisations = waveform_model(self.frequencies, params)
        projected_strain = self.fd_response(self.frequencies, polarisations, params)

        # 3. Set the new data
        strain_data = jnp.where(self.frequency_mask, projected_strain, 0.0 + 0.0j)
        if noise_key is not None:
            noise = self.psd.simulate_data(noise_key)
            strain_data += jnp.where(self.frequency_mask, noise, 0.0 + 0.0j)

        self.set_data(
            Data.from_fd(
                name=f"{self.name}_injected",
                fd_strain=strain_data,
                frequencies=self.frequencies,
                start_time=self.data.start_time,
            )
        )

        # 4. Update the sliced data and psd with the (potentially) new frequency bounds
        self.set_frequency_bounds()
        masked_signal = projected_strain[self.frequency_mask]

        df = self.sliced_frequencies[1] - self.sliced_frequencies[0]
        _optimal_snr_sq = inner_product(
            masked_signal, masked_signal, self.sliced_psd, df
        )
        optimal_snr = _optimal_snr_sq**0.5
        match_filtered_snr = complex_inner_product(
            masked_signal, self.sliced_fd_data, self.sliced_psd, df
        )
        match_filtered_snr /= optimal_snr

        # Save as attributes
        self.optimal_snr = optimal_snr
        self.match_filtered_snr = match_filtered_snr

        logger.info(f"For detector {self.name}, the injected signal has:")
        logger.info(f"  - Optimal SNR: {optimal_snr:.4f}")
        logger.info(f"  - Match filtered SNR: {match_filtered_snr:.4f}")

    def get_whitened_frequency_domain_strain(
        self, frequency_series: Complex[Array, " n_freq"]
    ) -> Complex[Array, " n_freq"]:
        """Get the whitened frequency-domain strain.
        Args:
            frequency_series (Complex[Array, "n_freq"]): Array of frequency domain data/signal.
        Returns:
            Complex[Array, "n_freq"]: Whitened frequency-domain strain.
        """
        scaled_asd = jnp.sqrt(self.psd.values * self.duration / 4)
        return (frequency_series / scaled_asd) * self.frequency_mask

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
        freq_mask_ratio = len(self.frequency_mask) / jnp.sqrt(
            jnp.sum(self.frequency_mask)
        )
        return jnp.fft.irfft(whitened_frequency_series) * freq_mask_ratio

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


def time_to_merger(
    frequency: Float[Array, " n_sample"],
    M_c: FloatLike,
    eta: FloatLike,
    s1_z: FloatLike = 0.0,
    s2_z: FloatLike = 0.0,
    mode: int = 2,
) -> Float[Array, " n_sample"]:
    """Stationary-phase time to merger at 2PN for an aligned-spin binary.

    Implements Eq. (3.3) of Poisson & Will (1995), arXiv:gr-qc/9502040,

    $$
    \\tau(f) = \\frac{5}{256}\\mathcal{M}(\\pi\\mathcal{M}f)^{-8/3}
    \\left[1 + \\frac{4}{3}\\left(\\frac{743}{336} + \\frac{11}{4}\\eta\\right)x
    - \\frac{8}{5}(4\\pi - \\beta)x^{3/2}
    + 2\\left(\\frac{3058673}{1016064} + \\frac{5429}{1008}\\eta
    + \\frac{617}{144}\\eta^2 - \\sigma\\right)x^2\\right],
    $$

    with $x = (\\pi M f)^{2/3}$, the spin-orbit term
    $\\beta = \\frac{1}{12}\\sum_i \\left[113 (m_i/M)^2 + 75\\eta\\right]\\chi_i$ and the
    spin-spin term $\\sigma = \\frac{474}{48}\\eta\\chi_1\\chi_2$ for spins aligned with
    the orbital angular momentum. A frequency $f$ of the azimuthal mode $m$ is
    emitted when the quadrupole frequency is $2f/|m|$, so
    $\\tau_m(f) = \\tau_{22}(2f/|m|)$.

    Args:
        frequency (Float[Array, "n_sample"]): Gravitational-wave frequency in Hz.
        M_c (Float): Detector-frame chirp mass in solar masses.
        eta (Float): Symmetric mass ratio.
        s1_z (Float, optional): Aligned spin of the primary. Defaults to 0.
        s2_z (Float, optional): Aligned spin of the secondary. Defaults to 0.
        mode (int, optional): Azimuthal mode number $m$. Defaults to 2.

    Returns:
        Float[Array, "n_sample"]: Time to merger in seconds.

    Raises:
        ValueError: If ``mode`` is 0, for which the mapping is undefined.
    """
    if mode == 0:
        raise ValueError(
            "mode must be a non-zero azimuthal number: an m = 0 mode has no "
            "frequency-to-time mapping."
        )
    f_22 = 2 * frequency / abs(mode)
    M = M_c * eta ** (-3.0 / 5.0)
    # Component mass fractions m_i / M.
    delta = jnp.sqrt(1 - 4 * eta)
    x1, x2 = (1 + delta) / 2, (1 - delta) / 2
    M_c_s = M_c * MTSUN
    x = (jnp.pi * M * MTSUN * f_22) ** (2.0 / 3.0)
    beta = ((113 * x1**2 + 75 * eta) * s1_z + (113 * x2**2 + 75 * eta) * s2_z) / 12
    sigma = 474 / 48 * eta * s1_z * s2_z
    tau_newtonian = 5 / 256 * M_c_s * (jnp.pi * M_c_s * f_22) ** (-8.0 / 3.0)
    return tau_newtonian * (
        1
        + 4 / 3 * (743 / 336 + 11 / 4 * eta) * x
        - 8 / 5 * (4 * jnp.pi - beta) * x**1.5
        + 2
        * (3058673 / 1016064 + 5429 / 1008 * eta + 617 / 144 * eta**2 - sigma)
        * x**2
    )


class GroundBased3G(GroundBased2G):
    """Ground-based detector whose response follows the Earth's rotation.

    Signals from next-generation detectors stay in band for hours, during which
    the Earth rotates appreciably. Each frequency sample is therefore projected
    with the antenna pattern and the delay from the geocenter evaluated at the
    sidereal angle of the time it is emitted, using the stationary-phase
    mapping $t(f) = t_c - \\tau(f)$ with $\\tau$ from
    [`time_to_merger`][jimgw.core.single_event.detector.time_to_merger]:

    $$
    \\mathrm{GMST}(f) = \\mathrm{GMST}(t_{\\rm trigger})
    + \\Omega_\\oplus\\left[t_c - \\tau(f)\\right],
    $$

    where $\\Omega_\\oplus$ is the sidereal rotation rate. The sky-frame
    polarizations are mode-summed, so the dominant $m = 2$ mapping is used.

    Optionally, the finite length of the arms is included through the
    single-arm transfer function of Eq. (2.13) of Baral et al. (2023),
    arXiv:2304.09889 (see also Rakhmanov 2008). Its phase is referenced to the
    time the light reaches the end mirror, as in that reference, and it tends to
    one in the long-wavelength limit.

    With both effects switched off, the response is identical to
    [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G].

    The response needs ``M_c``, ``eta``, ``s1_z`` and ``s2_z`` in the parameters
    passed to ``fd_response`` when the Earth's rotation is switched on. Under
    time marginalization the likelihood sets ``t_c`` to zero, so the rotation
    across the ``t_c`` range (about 7e-6 rad for 0.1 s) is neglected.

    Attributes:
        arm_length (float): Arm length in meters, used by the finite-size response.
        earth_rotation (bool): Whether the response follows the Earth's rotation.
        finite_size (bool): Whether the finite-arm-length response is included.
    """

    arm_length: float = 0.0
    earth_rotation: bool = True
    finite_size: bool = False

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
        arm_length: float = 0.0,
        earth_rotation: bool = True,
        finite_size: bool = False,
    ):
        """Initialize a ground-based detector with an Earth-rotating response.

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
            arm_length (float, optional): Arm length in meters. Defaults to 0.
            earth_rotation (bool, optional): Follow the Earth's rotation across
                frequencies. Defaults to True.
            finite_size (bool, optional): Include the finite-arm-length response.
                Defaults to False.

        Raises:
            ValueError: If ``finite_size`` is True and ``arm_length`` is not positive.
        """
        if finite_size and not arm_length > 0:
            raise ValueError(
                f"finite_size=True needs a positive arm_length, got {arm_length}."
            )
        super().__init__(
            name,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            xarm_azimuth=xarm_azimuth,
            yarm_azimuth=yarm_azimuth,
            xarm_tilt=xarm_tilt,
            yarm_tilt=yarm_tilt,
            modes=modes,
        )
        self.arm_length = arm_length
        self.earth_rotation = earth_rotation
        self.finite_size = finite_size

    @classmethod
    def from_detector(
        cls,
        detector: GroundBased2G,
        arm_length: float = 0.0,
        earth_rotation: bool = True,
        finite_size: bool = False,
    ) -> "GroundBased3G":
        """Build a detector with the geometry of an existing ground-based detector.

        Only the geometry and polarization modes are copied, not the data or PSD.

        Args:
            detector (GroundBased2G): Detector whose geometry is copied, e.g. one
                of the [`get_ET`][jimgw.core.single_event.detector.get_ET] detectors.
            arm_length (float, optional): Arm length in meters. Defaults to 0.
            earth_rotation (bool, optional): Follow the Earth's rotation across
                frequencies. Defaults to True.
            finite_size (bool, optional): Include the finite-arm-length response.
                Defaults to False.

        Returns:
            GroundBased3G: The new detector.
        """
        return cls(
            detector.name,
            latitude=detector.latitude,
            longitude=detector.longitude,
            elevation=detector.elevation,
            xarm_azimuth=detector.xarm_azimuth,
            yarm_azimuth=detector.yarm_azimuth,
            xarm_tilt=detector.xarm_tilt,
            yarm_tilt=detector.yarm_tilt,
            modes="".join(p.name for p in detector.polarization_mode),
            arm_length=arm_length,
            earth_rotation=earth_rotation,
            finite_size=finite_size,
        )

    def gmst_at_frequency(
        self, frequency: Float[Array, " n_sample"], params: dict[str, Float]
    ) -> Float[Array, " n_sample"]:
        """Sidereal angle at the time each frequency is emitted.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.
            params (dict[str, Float]): Source parameters containing ``gmst`` (at
                the trigger time), ``t_c``, ``M_c``, ``eta``, ``s1_z`` and ``s2_z``.

        Returns:
            Float[Array, "n_sample"]: GMST in radians, equal to the wrapped GMST
                at the trigger time plus the rotation since then. Non-positive frequencies, which have no emission time, are
                assigned the merger time.
        """
        positive = frequency > 0
        # Evaluate at a safe frequency where masked, so neither the value nor
        # its gradient is poisoned by tau(0) = inf.
        tau = time_to_merger(
            jnp.where(positive, frequency, 1.0),
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
        )
        tau = jnp.where(positive, tau, 0.0)
        # Wrap first: the rotation is added to an angle of order one rather
        # than to an unwrapped sidereal time, which keeps its precision.
        gmst = jnp.mod(params["gmst"], 2 * jnp.pi)
        return gmst + EARTH_ROTATION_RATE * (params["t_c"] - tau)

    @staticmethod
    def _finite_size_factor(x: Float, y: Float) -> Complex:
        """Single-arm transfer function, Eq. (2.13) of arXiv:2304.09889.

        Args:
            x (Float): Arm length in units of the gravitational wavelength, $fL/c$.
            y (Float): Projection of the propagation direction onto the arm.

        Returns:
            Complex: Transfer function, equal to one when $x = 0$.
        """
        return 0.5 * (
            jnp.exp(-1j * jnp.pi * x * (1 + y)) * jnp.sinc(x * (1 - y))
            + jnp.exp(1j * jnp.pi * x * (1 - y)) * jnp.sinc(x * (1 + y))
        )

    def _finite_size_antenna_pattern(
        self,
        frequency: Float[Array, " n_sample"],
        ra: FloatScalar,
        dec: FloatScalar,
        psi: FloatScalar,
        gmst: Float,
    ) -> dict[str, Complex[Array, " n_sample"]]:
        """Antenna patterns with each arm weighted by its finite-size transfer function."""
        propagation = -self._source_direction(ra, dec, gmst)
        x = frequency * self.arm_length / C_SI
        arms = self.arms
        arm_tensors = [0.5 * jnp.einsum("i,j->ij", arm, arm) for arm in arms]
        transfer = [
            self._finite_size_factor(x, jnp.einsum("i...,i->...", propagation, arm))
            for arm in arms
        ]

        antenna_patterns = {}
        for polarization in self.polarization_mode:
            wave_tensor = polarization.tensor_from_sky(ra, dec, psi, gmst)
            arm_patterns = [
                jnp.einsum("ij,ij...->...", arm_tensor, wave_tensor)
                for arm_tensor in arm_tensors
            ]
            antenna_patterns[polarization.name] = (
                transfer[0] * arm_patterns[0] - transfer[1] * arm_patterns[1]
            )
        return antenna_patterns

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict[str, Float],
    ) -> Complex[Array, " n_sample"]:
        """Project the sky-frame waveform with a frequency-dependent response.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.
            h_sky (dict[str, Float[Array, "n_sample"]]): Dictionary mapping polarization names
                to frequency-domain waveforms.
            params (dict[str, Float]): Source parameters containing ``ra``, ``dec``,
                ``psi``, ``trigger_time``, ``t_c`` and ``gmst`` (at the trigger time),
                and, when the Earth's rotation is switched on, ``M_c``, ``eta``,
                ``s1_z`` and ``s2_z``.

        Returns:
            Complex[Array, "n_sample"]: Complex strain measured by the detector in frequency domain.
        """
        if not (self.earth_rotation or self.finite_size):
            return super().fd_response(frequency, h_sky, params)

        ra, dec, psi = params["ra"], params["dec"], params["psi"]
        if self.earth_rotation:
            gmst = self.gmst_at_frequency(frequency, params)
        else:
            gmst = params["gmst"]

        if self.finite_size:
            antenna_pattern = self._finite_size_antenna_pattern(
                frequency, ra, dec, psi, gmst
            )
        else:
            antenna_pattern = self.antenna_pattern(ra, dec, psi, gmst)
        time_shift = self.delay_from_geocenter(ra, dec, gmst)
        return self._project(frequency, h_sky, antenna_pattern, time_shift, params)


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
