import logging
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES
from evosax.core.restart import RestartParams, RestartState, cma_cond
from jax.scipy.special import logsumexp
from jaxtyping import Array, Complex, Float
from ripplegw.interfaces import DistanceScaledWaveform, Waveform
from scipy.interpolate import interp1d

from jimgw.core.base import LikelihoodBase
from jimgw.core.constants import EARTH_RADIUS_LIGHT_S, MTSUN
from jimgw.core.prior import Prior, find_specific_prior
from jimgw.core.single_event.detector import Detector
from jimgw.core.single_event.marginalization_config import (
    DistanceMargConfig,
    PhaseMargConfig,
    TimeMargConfig,
)
from jimgw.core.single_event.utils import (
    FixedParameters,
    apply_fixed_parameters,
    complex_inner_product,
    inner_product,
)
from jimgw.core.transforms import NtoMTransform
from jimgw.core.utils import log_i0, round_up_to_power_of_two
from jimgw.typing import ComplexScalar, FloatLike, FloatScalar

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Shared single-event likelihood
# ----------------------------------------------------------------------


class SingleEventLikelihood(LikelihoodBase):
    detectors: Sequence[Detector]
    waveform: Waveform
    fixed_parameters: FixedParameters
    ref_dist: FloatLike
    _detector_data_selections: tuple[slice | Array, ...]
    _detector_psd_selections: tuple[slice | Array, ...]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        *,
        fixed_parameters: Optional[FixedParameters] = None,
        f_min: Optional[float | dict[str, float]] = None,
        f_max: Optional[float | dict[str, float]] = None,
    ) -> None:
        """
        Args:
            detectors (Sequence[Detector]): Detectors with initialized data and PSD.
                The detectors will be frozen to prevent accidental changes after the
                likelihood is constructed.
            waveform (Waveform): Waveform model to evaluate.
            fixed_parameters (Optional[dict]): Parameters held constant during
                sampling. Values may be scalars or callables
                ``f(params) -> Float | dict``; callables are applied in insertion
                order. Defaults to None (no fixed parameters).
            f_min (Optional[float | dict[str, float]]): Minimum frequency for
                likelihood evaluation.
            f_max (Optional[float | dict[str, float]]): Maximum frequency for
                likelihood evaluation.

        Raises:
            ValueError: If the waveform is a time-domain model, if any detector has
                uninitialized data or PSD, if the data of the detectors do not carry
                one trigger time inside them, or if the frequency range of the
                likelihood is invalid.
        """
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        # Check that the waveform is frequency-domain
        if getattr(waveform, "waveform_metadata", {}).get("domain") == "TD":
            raise ValueError(
                "The likelihood requires a frequency-domain waveform model, "
                f"got the time-domain {waveform!r}."
            )
        # Check that at least one detector is provided
        if len(detectors) == 0:
            raise ValueError("A likelihood needs at least one detector.")
        # Check that all detectors have initialized data and PSD
        for detector in detectors:
            detector._validate_geometry()
            if detector.data.is_empty:
                raise ValueError(
                    f"Detector '{detector.name}' does not have initialized data. "
                    f"Please set data using detector.set_data(), detector.set_zero_noise() "
                    f"or detector.simulate_and_set_noise() before initializing the likelihood."
                )
            if detector.psd.is_empty:
                raise ValueError(
                    f"Detector '{detector.name}' does not have initialized PSD. "
                    f"Please set PSD using detector.set_psd() or detector.load_and_set_psd() "
                    f"before initializing the likelihood."
                )
        # Check that all detectors have different names
        names = [detector.name for detector in detectors]
        if len(set(names)) != len(names):
            raise ValueError(
                f"The detectors of a likelihood must have different names, got {names}."
            )

        # Keep the supplied detector instances and their order.
        self.detectors = tuple(detectors)
        self.waveform = waveform
        self.fixed_parameters = fixed_parameters if fixed_parameters is not None else {}
        self._shared_trigger_time(self.detectors)
        self.time_marginalization = False
        self.phase_marginalization = False
        self.distance_marginalization = False

        self._prepare_detector_frequency_selections(f_min, f_max)

        # Common validation has succeeded. These shared instances remain frozen even
        # if a subclass subsequently fails to configure marginalization or caches.
        for detector in self.detectors:
            detector._freeze()

    def _warn_if_phase_marginalization_is_inexact(self) -> None:
        """Warn that phase marginalization needs a signal with the (2, 2) mode alone."""
        metadata = getattr(self.waveform, "waveform_metadata", {})
        reasons = []
        if metadata.get("is_precessing"):
            reasons.append("precession")
        if (metadata.get("highest_mode") or 2) > 2:
            reasons.append(f"modes up to m = {metadata['highest_mode']}")
        if reasons:
            logger.warning(
                "phase_marginalization assumes a signal with only the (2, 2) mode, "
                "but %r has %s, so the marginalized likelihood may be inaccurate.",
                self.waveform,
                " and ".join(reasons),
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def duration(self) -> FloatLike:
        """Duration of the data segment in seconds (taken from the first detector)."""
        return self.detectors[0].data.duration

    @property
    def df(self) -> FloatLike:
        """Shared frequency spacing of the frozen detector data."""
        return 1.0 / self.duration

    @property
    def trigger_time(self) -> float:
        """Shared trigger time of the frozen detector data."""
        return self.detectors[0].data.trigger_time

    @property
    def detector_names(self) -> list[str]:
        """Names of the detectors used in this likelihood."""
        return [detector.name for detector in self.detectors]

    @property
    def detector_frequencies(self) -> list[Float[Array, " n_freq"]]:
        """Frequencies in the frequency range of the likelihood, one array per detector."""
        return [
            detector.data.frequencies[selection]
            for detector, selection in zip(
                self.detectors, self._detector_data_selections
            )
        ]

    @property
    def waveform_caches_distance(self) -> bool:
        """Whether a waveform cache can be reused across ``d_L`` changes.

        Derived from the waveform's type by default, so subclasses may
        override it to control the behavior directly. Consumed by the
        waveform-cache block-dependency inference (see
        ``jimgw.core.single_event.blocked_likelihood``): when True, ``d_L``
        is excluded from the cache's dependency set, so a ``d_L``-only
        proposal block reuses the cache instead of rebuilding it.
        """
        return isinstance(self.waveform, DistanceScaledWaveform)

    # ------------------------------------------------------------------
    # Direct evaluation
    # ------------------------------------------------------------------

    def evaluate(self, params: Mapping[str, FloatLike]) -> FloatScalar:
        """Prepare parameters and evaluate the likelihood.

        Constants are injected directly; callables receive the current params
        dict and may return a scalar or a dict (the matching key is extracted).
        Callables are applied in insertion order.
        """
        return self._evaluate(self._prepare_parameters(params))

    def _prepare_parameters(
        self, params: Mapping[str, FloatLike]
    ) -> dict[str, FloatLike]:
        """Add marginalization defaults and fixed parameters."""
        prepared_params = dict(params)
        if self.time_marginalization:
            prepared_params["t_c"] = 0.0
        if self.phase_marginalization:
            prepared_params["phase_c"] = 0.0
        if self.distance_marginalization:
            prepared_params["d_L"] = self.ref_dist
        apply_fixed_parameters(prepared_params, self.fixed_parameters)
        return prepared_params

    @abstractmethod
    def _evaluate(self, params: Mapping[str, FloatLike]) -> FloatScalar:
        """Core likelihood evaluation method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    # ------------------------------------------------------------------
    # Waveform cache evaluation
    # ------------------------------------------------------------------

    def generate_waveform(self, params: Mapping[str, FloatLike]) -> Any:
        """Generate a reusable waveform cache.

        Evaluated at unit distance when ``waveform_caches_distance`` is True,
        so ``d_L`` is not a cache dependency; otherwise ``d_L`` is a
        dependency like any other waveform parameter. Every other effective
        waveform input is always a dependency, regardless of distance support.

        Args:
            params (dict[str, Float]): Source parameters to build the cache from.

        Returns:
            Any: An opaque cache understood by this likelihood's
                `evaluate_from_waveform`.
        """
        return self._generate_waveform(self._prepare_parameters(params))

    @abstractmethod
    def _generate_waveform(self, params: Mapping[str, FloatLike]) -> Any:
        """Build the waveform cache from already-prepared parameters."""
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate_from_waveform(
        self,
        params: Mapping[str, FloatLike],
        waveform_cache: Any,
    ) -> FloatScalar:
        """Evaluate this likelihood from a waveform cache."""
        return self._evaluate_from_waveform(
            self._prepare_parameters(params), waveform_cache
        )

    @abstractmethod
    def _evaluate_from_waveform(
        self,
        params: Mapping[str, FloatLike],
        waveform_cache: Any,
    ) -> FloatScalar:
        """Evaluate the likelihood from a generated waveform cache."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _waveform_sky_for_cache(
        self,
        frequencies: Float[Array, " n_freq"],
        params: Mapping[str, FloatLike],
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate sky-frame polarizations for a waveform cache.

        Evaluated at ``d_L = 1`` when ``waveform_caches_distance`` is True, so
        ``d_L`` is not a cache dependency; otherwise ``d_L`` is a dependency
        like any other waveform parameter.
        """
        # Not using `waveform_caches_distance` for passing type checks
        if isinstance(self.waveform, DistanceScaledWaveform):
            return self.waveform.at_unit_distance(frequencies, params)
        return self.waveform(frequencies, params)

    def _waveform_sky_from_cache(
        self,
        cached_polarizations: dict[str, Complex[Array, " n_freq"]],
        params: Mapping[str, FloatLike],
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Recover physical-distance polarizations from a cache entry.

        Rescales by ``1 / d_L`` when ``waveform_caches_distance`` is True;
        otherwise returns the cache unchanged, since ``d_L`` was already
        baked in by ``_waveform_sky_for_cache``.
        """
        if not self.waveform_caches_distance:
            return cached_polarizations
        distance_scale = 1.0 / params["d_L"]
        return {
            polarization: strain * distance_scale
            for polarization, strain in cached_polarizations.items()
        }

    # ------------------------------------------------------------------
    # Detector validation
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_trigger_time(detectors: Sequence[Detector]) -> float:
        """Return the trigger time shared by the data of every detector.

        The detectors are independent, so the likelihood checks that they all analyse
        the same segment: the data of each must have the same trigger time, which by
        default is 2 s before the end of the data, and it must lie inside the data of
        each.
        """

        data_trigger_times = {d.name: d.data.trigger_time for d in detectors}
        shared = set(data_trigger_times.values())
        if len(shared) != 1:
            raise ValueError(
                "The data of the detectors have different trigger times: "
                f"{data_trigger_times}. Use data from the same segment, or set the "
                "same data.trigger_time on all of them."
            )
        trigger_time = shared.pop()
        for detector in detectors:
            start_time = float(detector.data.start_time)
            end_time = start_time + float(detector.data.duration)
            if not start_time <= trigger_time <= end_time:
                raise ValueError(
                    f"The trigger time {trigger_time} is outside the data of detector "
                    f"'{detector.name}', which span [{start_time}, {end_time}]. Check "
                    "data.trigger_time and data.start_time."
                )
        return trigger_time

    @staticmethod
    def _validated_frequency_range(
        detector: Detector, f_min: Optional[float], f_max: Optional[float]
    ) -> tuple[float, float]:
        """Validate the frequency range of the likelihood for a detector.

        A missing ``f_min`` or ``f_max`` is first filled in from the frequency range
        of the data (or the Nyquist frequency).
        """

        data_f_min, data_f_max = detector.data.frequency_bounds
        nyquist = float(detector.data.sampling_frequency) / 2
        low = float(data_f_min if f_min is None else f_min)
        if f_max is None or f_max == jnp.inf:
            high = min(float(data_f_max), float(detector.data.frequencies[-1]))
        else:
            high = float(f_max)
            # The Nyquist frequency is 1 / delta_t / 2, which is off by a rounding error
            # for some sampling rates (744 Hz, for example): do not reject f_max for that.
            if high > nyquist and not jnp.isclose(high, nyquist, rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"f_max = {high} Hz is above the Nyquist frequency of detector "
                    f"{detector.name}'s data, {nyquist} Hz."
                )
        if not low > 0:
            raise ValueError(
                f"The frequency range of the likelihood for detector "
                f"{detector.name} starts at {low} Hz. Pass a positive f_min: "
                "waveforms are not defined at 0 Hz."
            )
        if not low < high:
            raise ValueError(
                f"The frequency range of the likelihood for detector "
                f"{detector.name}, [{low}, {high}] Hz, is empty. f_min must be "
                "below f_max."
            )
        return low, high

    # ------------------------------------------------------------------
    # Frequency selections
    # ------------------------------------------------------------------

    def _prepare_detector_frequency_selections(
        self,
        f_min: Optional[float | dict[str, float]],
        f_max: Optional[float | dict[str, float]],
    ) -> None:
        """Check the frequency range of every detector before fixing the FFT of any."""

        detector_names = set(self.detector_names)
        for name, bounds in (("f_min", f_min), ("f_max", f_max)):
            if isinstance(bounds, dict) and set(bounds) != detector_names:
                missing = sorted(detector_names - set(bounds))
                unknown = sorted(set(bounds) - detector_names, key=str)
                raise ValueError(
                    f"{name} must contain exactly the detector names; "
                    f"missing names: {missing}, unknown names: {unknown}."
                )
        spacings = jnp.array([1.0 / d.data.duration for d in self.detectors])
        if not jnp.allclose(spacings, spacings[0], rtol=1e-12, atol=0.0):
            raise ValueError(
                f"All detectors must have the same frequency spacing, got {spacings} "
                "Hz. Use data of the same duration."
            )
        frequency_ranges = []
        for detector in self.detectors:
            detector_f_min = f_min[detector.name] if isinstance(f_min, dict) else f_min
            detector_f_max = f_max[detector.name] if isinstance(f_max, dict) else f_max
            frequency_ranges.append(
                self._validated_frequency_range(
                    detector, detector_f_min, detector_f_max
                )
            )
        data_selections = []
        psd_selections = []
        for detector, frequency_range in zip(self.detectors, frequency_ranges):
            _, fd_data, psd = detector.slice_to_frequency_range(*frequency_range)
            if not bool(jnp.all(jnp.isfinite(fd_data / psd))):
                raise ValueError(
                    f"PSD-weighted data for detector '{detector.name}' must be "
                    f"finite within the frequency range of the likelihood {frequency_range}."
                )
            data_selections.append(
                self._frequency_selection(detector.data.frequencies, *frequency_range)
            )
            psd_selections.append(
                self._frequency_selection(detector.psd.frequencies, *frequency_range)
            )
        self._detector_frequency_ranges = tuple(frequency_ranges)
        self._detector_data_selections = tuple(data_selections)
        self._detector_psd_selections = tuple(psd_selections)

    @classmethod
    def _frequency_selection(
        cls, frequencies: Float[Array, " n_freq"], f_min: float, f_max: float
    ) -> slice | Array:
        """Resolve the frequencies from f_min to f_max before tracing any evaluation."""
        indices = jnp.flatnonzero((frequencies >= f_min) & (frequencies <= f_max))
        return cls._static_selection(indices)

    @staticmethod
    def _static_selection(indices: Array) -> slice | Array:
        """Use a static slice for consecutive indices, retaining other mappings."""
        if bool(jnp.all(jnp.diff(indices) == 1)):
            first = int(indices[0])
            return slice(first, first + len(indices))
        return indices

    def _detector_data_in_frequency_range(
        self, index: int
    ) -> tuple[
        Float[Array, " n_freq"], Complex[Array, " n_freq"], Float[Array, " n_freq"]
    ]:
        """Read the selected spectra from their frozen owners using static indices."""
        detector = self.detectors[index]
        return (
            detector.data.frequencies[self._detector_data_selections[index]],
            detector.data.fd[self._detector_data_selections[index]],
            detector.psd.values[self._detector_psd_selections[index]],
        )


# ----------------------------------------------------------------------
# Zero likelihood
# ----------------------------------------------------------------------


class ZeroLikelihood(LikelihoodBase):
    """Trivial likelihood that always returns zero.

    Useful for prior-only sampling or debugging.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Direct evaluation
    # ------------------------------------------------------------------

    def evaluate(self, params: Mapping[str, FloatLike]) -> FloatScalar:
        """Return zero regardless of the parameters.

        Args:
            params (dict[str, Float]): Ignored.

        Returns:
            FloatScalar: Always 0.0.
        """
        return jnp.zeros(())


# ----------------------------------------------------------------------
# Full frequency-domain likelihood
# ----------------------------------------------------------------------


class TransientLikelihoodFD(SingleEventLikelihood):
    """Frequency-domain transient gravitational wave likelihood.

    Supports optional marginalization over coalescence time, phase,
    and/or luminosity distance.

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        fixed_parameters: Parameters held constant during sampling.  Values
            may be constants or callables ``f(params) -> Float | dict``;
            callables are applied in insertion order.  See the likelihood
            tutorial for details and examples.
        f_min: Minimum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.  Must be positive
            and not below the minimum frequency of the data.  Defaults to that
            minimum frequency.
        f_max: Maximum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.  Must not exceed
            the maximum frequency of the data or the Nyquist frequency.  Defaults
            to the smaller of the two.
        time_marginalization: If provided, marginalize over coalescence time
            ``t_c``.  Pass a [`TimeMargConfig`][jimgw.core.single_event.likelihood.TimeMargConfig]
            object, a plain dict (e.g. ``{"tc_range": (-0.1, 0.1)}``), or ``True``
            (shorthand for ``TimeMargConfig()``).  ``False`` or the default ``None``
            disables time marginalization.
        phase_marginalization: If provided, marginalize over coalescence phase
            ``phase_c``.  Pass a [`PhaseMargConfig`][jimgw.core.single_event.likelihood.PhaseMargConfig]
            object, a plain dict ``{}``, or ``True`` (shorthand for ``PhaseMargConfig()``).
            ``False`` or the default ``None`` disables phase marginalization.
        distance_marginalization: If provided, marginalize over luminosity
            distance ``d_L``.  Pass a [`DistanceMargConfig`][jimgw.core.single_event.likelihood.DistanceMargConfig]
            object or a plain dict (e.g. ``{"distance_prior": prior, "n_dist_points": 10000}``).
            ``False`` or the default ``None`` disables distance marginalization.
            ``True`` is not supported — ``distance_prior`` has no default; pass a
            dict or `DistanceMargConfig` instead.

    Example:
        >>> likelihood = TransientLikelihoodFD(
        ...     detectors, waveform,
        ...     f_min=20, f_max=1024,
        ...     phase_marginalization=True,
        ...     time_marginalization={"tc_range": (-0.1, 0.1)},
        ... )
        >>> logL = likelihood.evaluate(params)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        *,
        fixed_parameters: Optional[FixedParameters] = None,
        f_min: Optional[float | dict[str, float]] = None,
        f_max: Optional[float | dict[str, float]] = None,
        time_marginalization: Optional[Union[TimeMargConfig, dict, bool]] = None,
        phase_marginalization: Optional[Union[PhaseMargConfig, dict, bool]] = None,
        distance_marginalization: Optional[
            Union[DistanceMargConfig, dict, bool]
        ] = None,
    ) -> None:
        super().__init__(
            detectors,
            waveform,
            fixed_parameters=fixed_parameters,
            f_min=f_min,
            f_max=f_max,
        )

        # --------------------------------------------------------------
        # Frequency setup
        # --------------------------------------------------------------

        _frequencies = self.detector_frequencies

        self.frequencies = jnp.unique(jnp.concatenate(_frequencies))
        self.frequency_indices = [
            jnp.searchsorted(self.frequencies, freqs) for freqs in _frequencies
        ]

        # --------------------------------------------------------------
        # Marginalization input validation
        # --------------------------------------------------------------

        if isinstance(time_marginalization, dict):
            time_marginalization = TimeMargConfig(**time_marginalization)
        elif time_marginalization is True:
            time_marginalization = TimeMargConfig()
        elif not time_marginalization:
            time_marginalization = None

        if isinstance(phase_marginalization, dict):
            phase_marginalization = PhaseMargConfig(**phase_marginalization)
        elif phase_marginalization is True:
            phase_marginalization = PhaseMargConfig()
        elif phase_marginalization is None or phase_marginalization is False:
            phase_marginalization = None
        elif not isinstance(phase_marginalization, PhaseMargConfig):
            # A wrong value would not fail later, so "false" would switch phase
            # marginalization on.
            raise ValueError(
                "phase_marginalization must be None, a bool, a dict or a PhaseMargConfig."
            )

        if isinstance(distance_marginalization, dict):
            distance_marginalization = DistanceMargConfig(**distance_marginalization)
        elif not distance_marginalization:
            distance_marginalization = None
        elif distance_marginalization is True:
            raise ValueError(
                "distance_marginalization=True is not supported because "
                "`distance_prior` has no default.  Pass a dict with `distance_prior` "
                "or a DistanceMargConfig instance instead."
            )

        # --------------------------------------------------------------
        # Marginalization flags
        # --------------------------------------------------------------

        self.time_marginalization = time_marginalization is not None
        self.phase_marginalization = phase_marginalization is not None
        self.distance_marginalization = distance_marginalization is not None

        if self.time_marginalization and self.distance_marginalization:
            raise NotImplementedError(
                "Joint time + distance marginalization is not yet supported."
            )

        if time_marginalization is not None:
            self._init_time_marginalization(time_marginalization)
        if self.phase_marginalization:
            self._init_phase_marginalization()
        if distance_marginalization is not None:
            self._init_distance_marginalization(distance_marginalization)

    # ------------------------------------------------------------------
    # Direct evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, params: Mapping[str, FloatLike]) -> FloatScalar:
        waveform_sky = self.waveform(self.frequencies, params)
        return self._likelihood(params, waveform_sky)

    # ------------------------------------------------------------------
    # Waveform cache evaluation
    # ------------------------------------------------------------------

    def _generate_waveform(
        self, params: Mapping[str, FloatLike]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Generate reusable sky-frame waveform polarizations.

        Evaluated at unit distance when ``waveform_caches_distance`` is True,
        so ``d_L`` is not a cache dependency; otherwise ``d_L`` is a
        dependency like any other waveform parameter.
        """
        return self._waveform_sky_for_cache(self.frequencies, params)

    def _evaluate_from_waveform(
        self,
        params: Mapping[str, FloatLike],
        waveform_cache: dict[str, Complex[Array, " n_freq"]],
    ) -> FloatScalar:
        """Core likelihood evaluation from a pre-generated waveform cache."""
        waveform_sky = self._waveform_sky_from_cache(waveform_cache, params)
        return self._likelihood(params, waveform_sky)

    # ------------------------------------------------------------------
    # Likelihood computation
    # ------------------------------------------------------------------

    def _likelihood(
        self,
        params: Mapping[str, FloatLike],
        waveform_sky: dict[str, Complex[Array, " n_freq"]],
    ) -> FloatScalar:
        """Core likelihood computation from a physical-distance sky-frame waveform."""

        # --------------------------------------------------------------
        # Choose accumulation type based on flags
        # --------------------------------------------------------------

        if self.time_marginalization:
            # Per-frequency complex array for FFT-based time marginalization
            complex_d_inner_h = jnp.zeros(len(self.frequencies), dtype=jnp.complex128)
            log_likelihood: FloatScalar = jnp.zeros(())

            for i, ifo in enumerate(self.detectors):
                frequencies, fd_data, psd = self._detector_data_in_frequency_range(i)
                waveform_sky_ifo = {
                    key: waveform_sky[key][self.frequency_indices[i]]
                    for key in waveform_sky
                }
                h_dec = ifo.fd_response(
                    frequencies,
                    waveform_sky_ifo,
                    params,
                )
                complex_d_inner_h = complex_d_inner_h.at[self.frequency_indices[i]].add(
                    4 * h_dec * jnp.conj(fd_data) / psd * self.df
                )
                optimal_SNR = inner_product(h_dec, h_dec, psd, self.df)
                log_likelihood += -optimal_SNR / 2

            if self.phase_marginalization:
                # joint time + phase marginalization
                log_likelihood += self._reduce_phase_time(complex_d_inner_h)
            else:
                # time only marginalization
                log_likelihood += self._reduce_time(complex_d_inner_h)
            return log_likelihood

        elif self.phase_marginalization or self.distance_marginalization:
            # Need complex or real accumulation across detectors
            complex_d_inner_h: ComplexScalar = jnp.zeros((), dtype=jnp.complex128)
            match_filter_snr: FloatScalar = jnp.zeros(())
            optimal_snr: FloatScalar = jnp.zeros(())

            for i, ifo in enumerate(self.detectors):
                frequencies, fd_data, psd = self._detector_data_in_frequency_range(i)
                waveform_sky_ifo = {
                    key: waveform_sky[key][self.frequency_indices[i]]
                    for key in waveform_sky
                }
                h_dec = ifo.fd_response(
                    frequencies,
                    waveform_sky_ifo,
                    params,
                )
                if self.phase_marginalization:
                    complex_d_inner_h += complex_inner_product(
                        h_dec, fd_data, psd, self.df
                    )
                else:
                    match_filter_snr += inner_product(h_dec, fd_data, psd, self.df)
                optimal_snr += inner_product(h_dec, h_dec, psd, self.df)

            if self.phase_marginalization and self.distance_marginalization:
                # joint phase + distance marginalization
                return self._reduce_phase_distance(complex_d_inner_h, optimal_snr)
            elif self.phase_marginalization:
                # phase only marginalization
                return self._reduce_phase(complex_d_inner_h, optimal_snr)
            else:
                # distance only marginalization
                return self._reduce_distance(match_filter_snr, optimal_snr)

        else:
            # No marginalization
            log_likelihood: FloatScalar = jnp.zeros(())
            for i, ifo in enumerate(self.detectors):
                frequencies, fd_data, psd = self._detector_data_in_frequency_range(i)
                waveform_sky_ifo = {
                    key: waveform_sky[key][self.frequency_indices[i]]
                    for key in waveform_sky
                }
                h_dec = ifo.fd_response(
                    frequencies,
                    waveform_sky_ifo,
                    params,
                )
                match_filter_SNR = inner_product(h_dec, fd_data, psd, self.df)
                optimal_SNR = inner_product(h_dec, h_dec, psd, self.df)
                log_likelihood += match_filter_SNR - optimal_SNR / 2
            return log_likelihood

    # ------------------------------------------------------------------
    # Time marginalization
    # ------------------------------------------------------------------

    def _init_time_marginalization(self, config: TimeMargConfig) -> None:
        # --------------------------------------------------------------
        # Time marginalization validation
        # --------------------------------------------------------------

        if "t_c" in self.fixed_parameters:
            raise ValueError("Cannot have t_c fixed while marginalizing over t_c")
        self.tc_range = config.tc_range
        low, high = self.tc_range
        if not bool(jnp.isfinite(low) & jnp.isfinite(high)) or not low < high:
            raise ValueError("tc_range must contain finite, increasing bounds.")
        time_frequency_indices = jnp.rint(self.frequencies * self.duration).astype(
            jnp.int64
        )
        # Preserve the usual half-sampling-rate grid, extending it when an odd-length
        # spectrum or a selected Nyquist bin needs one additional FFT coefficient.
        n_fft = max(
            max(d.data.n_time // 2 for d in self.detectors),
            int(time_frequency_indices[-1]) + 1,
        )
        self.time_frequency_indices = self._static_selection(time_frequency_indices)
        self.tc_array = jnp.fft.fftfreq(n_fft, float(self.df))
        # In FFT order an interval spanning zero wraps around the array. Shift it
        # first so the accepted times form a static slice instead of a gather.
        sorted_times = jnp.fft.fftshift(self.tc_array)
        time_indices = jnp.flatnonzero((sorted_times > low) & (sorted_times < high))
        if len(time_indices) == 0:
            raise ValueError(
                "tc_range does not contain any time-marginalization samples."
            )
        self._time_selection = self._static_selection(time_indices)
        grid_low = float(sorted_times[0])
        grid_high = grid_low + 1 / float(self.df)
        if float(low) < grid_low or float(high) > grid_high:
            logger.warning(
                "tc_range (%s, %s) extends beyond the time grid of the data, "
                "(%s, %s) s, so the marginalized likelihood may be inaccurate.",
                float(low),
                float(high),
                grid_low,
                grid_high,
            )
        # t_c has a uniform prior over tc_range: a sample of the time grid
        # weighs (time spacing) / (width of tc_range), and the time spacing is
        # 1 / (n_fft * df).
        self._log_time_weight = -jnp.log((high - low) * n_fft * float(self.df))

    def _time_fft(
        self, complex_d_inner_h: Complex[Array, " n_freq"]
    ) -> Complex[Array, " n_time"]:
        """Place every frequency of the likelihood on its FFT bin, zeroing unused bins."""
        dense = jnp.zeros(len(self.tc_array), dtype=complex_d_inner_h.dtype)
        dense = dense.at[self.time_frequency_indices].add(complex_d_inner_h)
        return jnp.fft.fft(dense, norm="backward")

    def _reduce_time(self, complex_d_inner_h: Complex[Array, " n_freq"]) -> FloatScalar:
        """FFT-based time marginalization (real part)."""
        fft_d_inner_h = jnp.fft.fftshift(self._time_fft(complex_d_inner_h))[
            self._time_selection
        ]
        return logsumexp(fft_d_inner_h.real) + self._log_time_weight

    # ------------------------------------------------------------------
    # Phase marginalization
    # ------------------------------------------------------------------

    def _init_phase_marginalization(self) -> None:
        # --------------------------------------------------------------
        # Phase marginalization validation
        # --------------------------------------------------------------

        if "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )
        self._warn_if_phase_marginalization_is_inexact()

    def _reduce_phase(
        self,
        complex_d_inner_h: complex | ComplexScalar,
        optimal_snr: FloatScalar,
    ) -> FloatScalar:
        """Phase marginalization via modified Bessel function (Thrane & Talbot 2019, Eq. 24)."""
        return -optimal_snr / 2 + log_i0(jnp.absolute(complex_d_inner_h))

    # ------------------------------------------------------------------
    # Distance marginalization
    # ------------------------------------------------------------------

    def _init_distance_marginalization(self, config: DistanceMargConfig) -> None:
        # --------------------------------------------------------------
        # Distance marginalization validation
        # --------------------------------------------------------------

        distance_prior = config.distance_prior
        n_dist_points = config.n_dist_points
        ref_dist = config.ref_dist

        if ref_dist is not None and not (jnp.isfinite(ref_dist) and ref_dist > 0):
            raise ValueError("ref_dist must be finite and positive.")

        if "d_L" in self.fixed_parameters:
            raise ValueError("Cannot have d_L fixed while marginalising over d_L")

        if list(distance_prior.parameter_names) != ["d_L"]:
            raise ValueError(
                f"distance_prior must be a 1D prior with parameter_names=['d_L'], "
                f"got parameter_names={list(distance_prior.parameter_names)}."
            )

        bounds = distance_prior.get_bounds()
        if bounds is None:
            raise ValueError(
                "The d_L sub-prior must have xmin and xmax attributes. "
                "Use a bounded prior such as PowerLawPrior or UniformPrior."
            )
        dist_min, dist_max = bounds

        if not bool(jnp.all(jnp.isfinite(jnp.asarray(bounds)))):
            raise ValueError("The d_L prior bounds must be finite.")

        if dist_min <= 0:
            raise ValueError(
                "The d_L prior's xmin must be > 0 (distance must be positive)"
            )
        if dist_max <= dist_min:
            raise ValueError("The d_L prior's xmax must be greater than xmin")

        if ref_dist is None:
            self.ref_dist = dist_min + (dist_max - dist_min) / 2.0
        else:
            self.ref_dist = ref_dist

        distance_grid = jnp.linspace(dist_min, dist_max, n_dist_points)
        delta_d = (dist_max - dist_min) / (n_dist_points - 1)
        self.scaling = self.ref_dist / distance_grid
        if not bool(jnp.all(jnp.isfinite(self.scaling**2))):
            raise ValueError(
                "ref_dist must yield finite distance scaling over the prior."
            )

        log_prob_fn = jax.vmap(lambda d: distance_prior.log_prob({"d_L": d}))
        log_w = log_prob_fn(distance_grid) + jnp.log(delta_d)
        if log_w.shape != distance_grid.shape or jnp.iscomplexobj(log_w):
            raise ValueError(
                "distance_prior must return a real scalar log probability."
            )
        max_log_weight = jnp.max(log_w)
        if not bool(jnp.isfinite(max_log_weight)):
            raise ValueError(
                "distance_prior must have finite, nonzero probability over the "
                "distance-marginalization grid."
            )
        # Remove the common offset before normalization so a large finite log
        # density cannot erase the relative weights through cancellation.
        log_w = log_w - max_log_weight
        log_normalization = logsumexp(log_w)
        self.log_weights = log_w - log_normalization

    def _reduce_distance(
        self, match_filter_snr: FloatScalar, optimal_snr: FloatScalar
    ) -> FloatScalar:
        """Distance marginalization using scaling + logsumexp."""
        log_integrand = (
            match_filter_snr * self.scaling
            - 0.5 * optimal_snr * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)

    # ------------------------------------------------------------------
    # Combined marginalization
    # ------------------------------------------------------------------

    def _reduce_phase_time(
        self, complex_d_inner_h: Complex[Array, " n_freq"]
    ) -> FloatScalar:
        """FFT-based time + phase marginalization (Bessel-weighted FFT)."""
        fft_d_inner_h = jnp.fft.fftshift(self._time_fft(complex_d_inner_h))[
            self._time_selection
        ]
        log_i0_abs_fft = log_i0(jnp.absolute(fft_d_inner_h))
        return logsumexp(log_i0_abs_fft) + self._log_time_weight

    def _reduce_phase_distance(
        self,
        complex_d_inner_h: complex | ComplexScalar,
        optimal_snr: FloatScalar,
    ) -> FloatScalar:
        """Phase + distance marginalization (Thrane & Talbot 2019, Eq. 79)."""
        abs_kappa = jnp.absolute(complex_d_inner_h)
        log_integrand = (
            log_i0(abs_kappa * self.scaling)
            - 0.5 * optimal_snr * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)


# ----------------------------------------------------------------------
# Heterodyned likelihood
# ----------------------------------------------------------------------


class HeterodynedTransientLikelihoodFD(SingleEventLikelihood):
    """Frequency-domain likelihood using the relative-binning (heterodyne) scheme.

    Optionally marginalizes over coalescence phase when ``phase_marginalization``
    is provided.

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        fixed_parameters: Dictionary of fixed parameter values.  Each value
            may be a constant ``Float``, a callable returning a scalar, **or**
            a callable returning a ``dict`` (e.g. ``transform.backward``).
            See [`TransientLikelihoodFD`][jimgw.core.single_event.likelihood.TransientLikelihoodFD]
            for a detailed description and example.
        f_min: Minimum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.  Must be positive
            and not below the minimum frequency of the data.  Defaults to that
            minimum frequency.
        f_max: Maximum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.  Must not exceed
            the maximum frequency of the data or the Nyquist frequency.  Defaults
            to the smaller of the two.
        n_bins: Number of frequency bins for relative binning.  Mutually
            exclusive with ``epsilon``; raises ``ValueError`` if both are set.
            When neither is set, ``epsilon=0.5`` is used as the default.
        epsilon: Maximum allowed phase change per bin (rad).  The bin count
            is set to ``max(1, int(total_phase / epsilon))``.  Mutually
            exclusive with ``n_bins``; raises ``ValueError`` if both are set.
            When neither is set, ``epsilon=0.5`` is used as the default.
        optimizer_popsize: Population size for the CMA-ES optimizer used
            when finding reference parameters automatically.  Defaults to 500.
        optimizer_n_steps: Maximum number of CMA-ES generations.  Defaults to 1000.
        optimizer_target: Optional log-likelihood value at which the CMA-ES
            search stops early, before ``optimizer_n_steps`` is reached.
            Defaults to None, which runs up to the full
            ``optimizer_n_steps`` generations; the search may still stop
            earlier if CMA-ES's convergence/stall criterion fires.
        reference_parameters: Pre-computed reference parameters (dict).  If
            supplied, the optimizer is skipped entirely.
        reference_waveform: Optional waveform instance used to compute the
            reference waveform.  Defaults to ``waveform`` when not provided.
        prior: Prior distribution from which the initial CMA-ES mean is
            drawn.  Required when ``reference_parameters`` is not provided.
        likelihood_transforms: Transforms mapping sampling parameters to
            likelihood parameters (e.g. mass-ratio → symmetric mass-ratio).
        phase_marginalization: If provided, marginalize over coalescence phase
            ``phase_c``.  Pass a [`PhaseMargConfig`][jimgw.core.single_event.likelihood.PhaseMargConfig]
            object, a plain dict ``{}``, or ``True`` (shorthand for ``PhaseMargConfig()``).
            ``None`` or ``False`` (default) disables phase marginalization.
    """

    n_bins: int
    epsilon: float
    reference_parameters: dict
    freq_grid: Float[Array, " n_valid+1"]
    _waveform_ref: dict[str, Complex[Array, " 2*n_valid"]]
    summary_data: dict[str, Complex[Array, "4 n_valid"]]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        *,
        fixed_parameters: Optional[FixedParameters] = None,
        f_min: Optional[float | dict[str, float]] = None,
        f_max: Optional[float | dict[str, float]] = None,
        n_bins: Optional[int] = None,
        epsilon: Optional[float] = None,
        optimizer_popsize: int = 500,
        optimizer_n_steps: int = 1000,
        optimizer_target: Optional[float] = None,
        reference_parameters: Optional[dict] = None,
        reference_waveform: Optional[Waveform] = None,
        prior: Optional[Prior] = None,
        likelihood_transforms: Optional[list[NtoMTransform]] = None,
        phase_marginalization: Optional[Union[PhaseMargConfig, dict, bool]] = None,
    ):
        super().__init__(
            detectors,
            waveform,
            fixed_parameters=fixed_parameters,
            f_min=f_min,
            f_max=f_max,
        )

        # --------------------------------------------------------------
        # Phase marginalization input validation
        # --------------------------------------------------------------

        if isinstance(phase_marginalization, dict):
            phase_marginalization = PhaseMargConfig(**phase_marginalization)
        elif phase_marginalization is True:
            phase_marginalization = PhaseMargConfig()
        elif phase_marginalization is None or phase_marginalization is False:
            phase_marginalization = None
        elif not isinstance(phase_marginalization, PhaseMargConfig):
            raise ValueError(
                "phase_marginalization must be None, a bool, a dict or a PhaseMargConfig."
            )
        self.phase_marginalization = phase_marginalization is not None

        # --------------------------------------------------------------
        # Binning validation
        # --------------------------------------------------------------

        # Reject bad binning inputs before computing or optimizing a reference.
        if n_bins is not None:
            if epsilon is not None:
                raise ValueError(
                    "'n_bins' and 'epsilon' are mutually exclusive; specify at most one."
                )
            if (
                isinstance(n_bins, bool)
                or not isinstance(n_bins, Integral)
                or n_bins <= 0
            ):
                raise ValueError(
                    f"'n_bins' must be a positive integer, got {n_bins!r}."
                )
            n_bins = int(n_bins)
        elif epsilon is None:
            epsilon = 0.5

        if epsilon is not None:
            try:
                epsilon_array = jnp.asarray(epsilon)
                valid_epsilon = (
                    epsilon_array.ndim == 0
                    and jnp.issubdtype(epsilon_array.dtype, jnp.number)
                    and not jnp.iscomplexobj(epsilon_array)
                    and bool(jnp.isfinite(epsilon_array) & (epsilon_array > 0))
                )
            except (TypeError, ValueError):
                valid_epsilon = False
            if not valid_epsilon:
                raise ValueError(
                    f"'epsilon' must be a finite positive number, got {epsilon!r}."
                )

        # --------------------------------------------------------------
        # Frequency setup (same as TransientLikelihoodFD)
        # --------------------------------------------------------------

        _frequencies = self.detector_frequencies

        self.frequencies = jnp.unique(jnp.concatenate(_frequencies))
        self.frequency_indices = [
            jnp.searchsorted(self.frequencies, freqs) for freqs in _frequencies
        ]

        # --------------------------------------------------------------
        # Phase marginalization validation
        # --------------------------------------------------------------

        if self.phase_marginalization and "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )
        if self.phase_marginalization:
            self._warn_if_phase_marginalization_is_inexact()

        # --------------------------------------------------------------
        # Heterodyne setup
        # --------------------------------------------------------------

        logger.info("Initializing heterodyned likelihood..")

        if likelihood_transforms is None:
            likelihood_transforms = []

        if reference_waveform is None:
            reference_waveform = waveform

        if reference_parameters:
            self.reference_parameters = reference_parameters.copy()
            apply_fixed_parameters(self.reference_parameters, self.fixed_parameters)
            logger.info(
                f"Found reference parameters, they are {self.reference_parameters}"
            )
        elif prior:
            logger.info("No reference parameters are provided, finding it...")
            reference_parameters = self.maximize_likelihood(
                prior=prior,
                likelihood_transforms=likelihood_transforms,
                optimizer_popsize=optimizer_popsize,
                optimizer_n_steps=optimizer_n_steps,
                optimizer_target=optimizer_target,
            )
            self.reference_parameters = {
                key: float(value) for key, value in reference_parameters.items()
            }
            logger.info(f"The reference parameters are {self.reference_parameters}")
        else:
            raise ValueError(
                "Either reference parameters or parameter names must be provided"
            )
        logger.info("Constructing reference waveforms..")

        self._waveform_ref = {}
        self.summary_data = {}

        if epsilon is not None:
            freqs_arr = self.frequencies
            phase = HeterodynedTransientLikelihoodFD._max_phase_diff(
                freqs_arr, freqs_arr[0], freqs_arr[-1]
            )
            n_bins = max(1, int(float(phase[-1]) / epsilon))
        # The input checks and epsilon calculation above resolve an integer count.
        assert isinstance(n_bins, int)
        freq_grid = self._make_binning_scheme(self.frequencies, n_bins=n_bins)

        ref_hpc = reference_waveform(self.frequencies, self.reference_parameters)
        masked_freq_grid = self._mask_and_set_frequency_arrays(ref_hpc, freq_grid)

        # --------------------------------------------------------------
        # Reference edge validation
        # --------------------------------------------------------------

        # Each unsuccessful pass removes at least one bin. The initial bin count
        # therefore bounds the work needed to find supported edges.
        for _ in range(self.n_bins):
            hpc_low = reference_waveform(self.freq_grid_low, self.reference_parameters)
            hpc_high = reference_waveform(
                self.freq_grid_high, self.reference_parameters
            )
            self._validate_reference_waveform(hpc_low, self.freq_grid_low)
            self._validate_reference_waveform(hpc_high, self.freq_grid_high)
            if hpc_low.keys() != hpc_high.keys() or hpc_low.keys() != ref_hpc.keys():
                raise ValueError(
                    "Reference waveform polarizations must match across grids."
                )
            # Grid-dependent cutoffs can remove further edge bins. Trim only the
            # ends and regenerate: removing lower bins changes the grid spacing.
            low_support = jnp.any(
                jnp.array([value != 0 for value in hpc_low.values()]), axis=0
            )
            high_support = jnp.any(
                jnp.array([value != 0 for value in hpc_high.values()]), axis=0
            )
            supported_bins = low_support & high_support
            if bool(jnp.all(supported_bins)):
                break
            supported_indices = jnp.flatnonzero(supported_bins)
            if not len(supported_indices):
                raise ValueError(
                    "Reference waveform must have finite, nonzero bin edges in "
                    "the frequency range of the likelihood; increase n_bins or choose a "
                    "supported range."
                )
            first, last = int(supported_indices[0]), int(supported_indices[-1])
            if not bool(jnp.all(supported_bins[first : last + 1])):
                raise ValueError(
                    "Reference waveform must be nonzero at bin edges inside its "
                    "supported range."
                )
            self.freq_grid = self.freq_grid[first : last + 2]
            self.n_bins = len(self.freq_grid) - 1
            masked_freq_grid = self.freq_grid
        else:
            raise RuntimeError("Reference bin refinement did not converge.")
        if reference_waveform is not waveform:
            for frequencies in (self.freq_grid_low, self.freq_grid_high):
                self._validate_reference_waveform(
                    waveform(frequencies, self.reference_parameters),
                    frequencies,
                    label="Waveform",
                )
        hpc_edges = {
            key: jnp.concatenate((hpc_low[key], hpc_high[key])) for key in hpc_low
        }
        response_frequencies = jnp.concatenate(
            (self.freq_grid_low, self.freq_grid_high)
        )

        # --------------------------------------------------------------
        # Reference response and coefficient validation
        # --------------------------------------------------------------

        for i, detector in enumerate(self.detectors):
            hpc_ifo = {key: ref_hpc[key][self.frequency_indices[i]] for key in ref_hpc}
            waveform_ref = detector.fd_response(
                _frequencies[i],
                hpc_ifo,
                self.reference_parameters,
            )
            self._waveform_ref[detector.name] = detector.fd_response(
                response_frequencies,
                hpc_edges,
                self.reference_parameters,
            )
            edge_reference = self._waveform_ref[detector.name]
            if not bool(jnp.all(jnp.isfinite(edge_reference) & (edge_reference != 0))):
                raise ValueError(
                    f"Reference response for {detector.name} must be finite and "
                    "nonzero at every bin edge; choose different reference "
                    "parameters or a supported frequency range."
                )
            self.summary_data[detector.name] = self._compute_coefficients(
                detector,
                waveform_ref,
                masked_freq_grid,
            )
            if not bool(jnp.all(jnp.isfinite(self.summary_data[detector.name]))):
                raise ValueError(
                    f"Heterodyne coefficients for {detector.name} must be finite; "
                    "check the reference waveform, strain and PSD."
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def freq_grid_low(self) -> Float[Array, " n_valid"]:
        """Lower frequency edge of each relative-binning interval."""
        return self.freq_grid[:-1]

    @property
    def freq_grid_high(self) -> Float[Array, " n_valid"]:
        """Upper frequency edge of each relative-binning interval."""
        return self.freq_grid[1:]

    @property
    def bin_widths(self) -> Float[Array, " n_valid"]:
        """Widths derived from the shared bin-edge grid."""
        return self.freq_grid_high - self.freq_grid_low

    # ------------------------------------------------------------------
    # Direct evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, params: Mapping[str, FloatLike]) -> FloatScalar:
        # Preserve both waveform grids: some models depend on grid spacing.
        waveform_low = self.waveform(self.freq_grid_low, params)
        waveform_high = self.waveform(self.freq_grid_high, params)
        return self._likelihood(params, waveform_low, waveform_high)

    # ------------------------------------------------------------------
    # Waveform cache evaluation
    # ------------------------------------------------------------------

    def _generate_waveform(
        self, params: Mapping[str, FloatLike]
    ) -> dict[str, dict[str, Complex[Array, " n_bins"]]]:
        """Generate bin-edge polarizations for cache reuse.

        Evaluated at unit distance when ``waveform_caches_distance`` is True,
        so ``d_L`` is not a cache dependency; otherwise ``d_L`` is a
        dependency like any other waveform parameter.
        """
        return {
            "low": self._waveform_sky_for_cache(self.freq_grid_low, params),
            "high": self._waveform_sky_for_cache(self.freq_grid_high, params),
        }

    def _evaluate_from_waveform(
        self,
        params: Mapping[str, FloatLike],
        waveform_cache: dict[str, dict[str, Complex[Array, " n_bins"]]],
    ) -> FloatScalar:
        """Core likelihood evaluation from a pre-generated waveform cache."""
        waveform_low = self._waveform_sky_from_cache(waveform_cache["low"], params)
        waveform_high = self._waveform_sky_from_cache(waveform_cache["high"], params)
        return self._likelihood(params, waveform_low, waveform_high)

    # ------------------------------------------------------------------
    # Likelihood computation
    # ------------------------------------------------------------------

    def _likelihood(
        self,
        params: Mapping[str, FloatLike],
        waveform_low: dict[str, Complex[Array, " n_bins"]],
        waveform_high: dict[str, Complex[Array, " n_bins"]],
    ) -> FloatScalar:
        """Core likelihood computation from physical-distance bin-edge polarizations."""
        log_likelihood: FloatScalar = jnp.zeros(())

        complex_d_inner_h: ComplexScalar = jnp.zeros((), dtype=jnp.complex128)

        # Project both edge grids together without changing waveform evaluation.
        frequencies = jnp.concatenate((self.freq_grid_low, self.freq_grid_high))
        waveform_sky = {
            key: jnp.concatenate((waveform_low[key], waveform_high[key]))
            for key in waveform_low
        }

        for detector in self.detectors:
            waveform_edges = detector.fd_response(
                frequencies,
                waveform_sky,
                params,
            )

            ratios = waveform_edges / self._waveform_ref[detector.name]
            r_low, r_high = jnp.split(ratios, 2)
            r0 = (r_low + r_high) / 2
            r1 = (r_high - r_low) / self.bin_widths

            _data = self.summary_data[detector.name]
            A0, A1, B0, B1 = _data[0], _data[1], _data[2], _data[3]

            if self.phase_marginalization:
                complex_d_inner_h += jnp.sum(A0 * r0.conj() + A1 * r1.conj())
                optimal_SNR = jnp.sum(
                    B0 * (r0.real**2 + r0.imag**2)
                    + 2 * B1 * (r0.real * r1.real + r0.imag * r1.imag)
                )
                log_likelihood += -optimal_SNR.real / 2
            else:
                match_filter_SNR = jnp.sum(A0 * r0.conj() + A1 * r1.conj())
                optimal_SNR = jnp.sum(
                    B0 * (r0.real**2 + r0.imag**2)
                    + 2 * B1 * (r0.real * r1.real + r0.imag * r1.imag)
                )
                log_likelihood += (match_filter_SNR - optimal_SNR / 2).real

        if self.phase_marginalization:
            log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))

        return log_likelihood

    # ------------------------------------------------------------------
    # Reference parameter optimization
    # ------------------------------------------------------------------

    def maximize_likelihood(
        self,
        prior: Prior,
        likelihood_transforms: list[NtoMTransform],
        optimizer_popsize: int = 500,
        optimizer_n_steps: int = 1000,
        optimizer_target: Optional[float] = None,
    ):
        """Find the maximum-likelihood parameters using CMA-ES.

        Uses ``evosax.CMA_ES`` (Covariance Matrix Adaptation Evolution
        Strategy) to search the full parameter space.  The initial mean is
        drawn from the prior and the entire ask/tell loop is compiled with
        ``jax.lax.while_loop`` for speed, stopping once ``optimizer_n_steps``
        generations have run, ``optimizer_target`` has been reached, or
        evosax's own CMA-ES convergence/stall criterion (``cma_cond``, using
        its built-in default tolerances) detects that the search has
        converged or numerically degenerated — whichever comes first.

        Args:
            prior: Prior used to seed the initial CMA-ES mean.
            likelihood_transforms: Transforms mapping sampling parameters to
                likelihood parameters.
            optimizer_popsize: Population size for CMA-ES.
                Defaults to 500.
            optimizer_n_steps: Maximum number of CMA-ES generations.
                Defaults to 1000.
            optimizer_target: Optional log-likelihood value (same scale as
                `evaluate`) at which to stop early, before
                ``optimizer_n_steps`` is reached. Defaults to None, which
                runs up to the full ``optimizer_n_steps`` generations (the
                search may still stop earlier if CMA-ES's own
                convergence/stall criterion fires).
        """
        # --------------------------------------------------------------
        # Optimizer validation
        # --------------------------------------------------------------

        for name, count in (
            ("optimizer_popsize", optimizer_popsize),
            ("optimizer_n_steps", optimizer_n_steps),
        ):
            if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                raise ValueError(f"{name} must be a positive integer, got {count!r}.")
        if optimizer_target is not None:
            try:
                target = jnp.asarray(optimizer_target)
                valid_target = (
                    target.ndim == 0
                    and jnp.issubdtype(target.dtype, jnp.number)
                    and not jnp.iscomplexobj(target)
                    and bool(jnp.isfinite(target))
                )
            except (TypeError, ValueError):
                valid_target = False
            if not valid_target:
                raise ValueError(
                    f"optimizer_target must be a finite real scalar or None, "
                    f"got {optimizer_target!r}."
                )
        parameter_names = list(prior.parameter_names)
        n_dim = len(parameter_names)

        # --------------------------------------------------------------
        # Reuse the frequencies of the likelihood for every detector
        # --------------------------------------------------------------

        frequency_grids = dict(zip(self.detector_names, self.detector_frequencies))

        # --------------------------------------------------------------
        # Build the full (un-marginalized) TransientLikelihoodFD objective
        # --------------------------------------------------------------

        full_likelihood = TransientLikelihoodFD(
            detectors=self.detectors,
            waveform=self.waveform,
            f_min={
                name: float(frequencies[0])
                for name, frequencies in frequency_grids.items()
            },
            f_max={
                name: float(frequencies[-1])
                for name, frequencies in frequency_grids.items()
            },
        )

        # --------------------------------------------------------------
        # Search space normalization
        # --------------------------------------------------------------

        # Prior sample statistics give each dimension unit variance so CMA-ES
        # can use std_init=1e-3 on a comparable scale for every parameter.
        n_init = max(optimizer_popsize, 1000)
        init_samples = prior.sample(jax.random.key(0), n_init)
        sample_matrix = jnp.column_stack(
            [init_samples[key] for key in parameter_names]
        )  # (n_init, n_dim)
        prior_mean = jnp.mean(sample_matrix, axis=0)
        prior_std = jnp.std(sample_matrix, axis=0)

        def _log_likelihood(z: Float[Array, " n_dim"]) -> FloatScalar:
            """Evaluate -logL for a single normalized parameter vector."""
            x = prior_mean + prior_std * z
            named_params: dict[str, FloatLike] = dict(
                zip(parameter_names, x, strict=True)
            )
            prior_log_prob = prior.log_prob(named_params)
            for transform in likelihood_transforms:
                named_params = transform.forward(named_params)
            named_params = apply_fixed_parameters(named_params, self.fixed_parameters)
            return jnp.where(
                jnp.isfinite(prior_log_prob),
                -full_likelihood.evaluate(named_params),
                jnp.inf,
            )

        _log_likelihood_vmap = jax.vmap(_log_likelihood)

        # --------------------------------------------------------------
        # Set up CMA-ES in normalized space: init_mean=0, std_init=1e-3
        # --------------------------------------------------------------

        es = CMA_ES(population_size=optimizer_popsize, solution=jnp.zeros(n_dim))
        es_params = replace(es.default_params, std_init=1e-3)
        key = jax.random.key(42)
        state = es.init(key, jnp.zeros(n_dim), es_params)

        logger.info(
            f"Running evosax CMA-ES: "
            f"{n_dim}D, popsize={optimizer_popsize}, n_steps={optimizer_n_steps}"
            + (
                f", optimizer_target={optimizer_target}"
                if optimizer_target is not None
                else ""
            )
        )

        target_fitness = -jnp.inf if optimizer_target is None else -optimizer_target

        stall_params = RestartParams()
        stall_restart_state = RestartState(restart_counter=0)
        _unused_fitness = jnp.zeros(1)

        def _cma_stalled(state) -> jax.Array:
            return cma_cond(
                None,
                _unused_fitness,
                state,
                es_params,
                stall_restart_state,
                stall_params,
            )

        def _cond(carry):
            state, _ = carry
            return (
                (state.generation_counter < optimizer_n_steps)
                & (state.best_fitness > target_fitness)
                & ~_cma_stalled(state)
            )

        def _step(carry):
            state, key = carry
            key, key_ask, key_tell = jax.random.split(key, 3)
            population, state = es.ask(key_ask, state, es_params)
            fitness = _log_likelihood_vmap(population)
            # Replace NaN/inf with a large penalty so CMA-ES state is never
            # corrupted by unphysical parameter samples (e.g. q < 0 → eta < 0
            # → waveform returns NaN).  Without this, jnp.argmin treats NaN as
            # the smallest value, best_solution never leaves its NaN initial
            # value, and the entire optimizer output is NaN.
            fitness = jnp.where(
                jnp.isfinite(fitness), fitness, jnp.finfo(jnp.float64).max
            )
            state, _ = es.tell(key_tell, population, fitness, state, es_params)
            return (state, key)

        state, _ = jax.lax.while_loop(_cond, _step, (state, key))

        best_fitness = float(state.best_fitness)

        target_reached = optimizer_target is not None and bool(
            state.best_fitness <= target_fitness
        )
        stalled = bool(_cma_stalled(state))
        if target_reached:
            stop_reason = "target_reached"
        elif stalled:
            stop_reason = "stalled"
        else:
            stop_reason = "max_steps"

        logger.debug(
            f"CMA-ES finished after {int(state.generation_counter)} generations "
            f"(limit {optimizer_n_steps}), best_fitness={best_fitness:.4f}, "
            f"stop_reason={stop_reason}"
        )
        best_z = state.best_solution

        # --------------------------------------------------------------
        # Convert best solution back to named parameters
        # --------------------------------------------------------------

        best_x = prior_mean + prior_std * best_z
        named_params = dict(zip(parameter_names, best_x, strict=True))
        for transform in likelihood_transforms:
            named_params = transform.forward(named_params)
        named_params = apply_fixed_parameters(named_params, self.fixed_parameters)
        return named_params

    # ------------------------------------------------------------------
    # Relative binning setup
    # ------------------------------------------------------------------

    def _make_binning_scheme(
        self,
        freqs: Float[Array, " n_freq"],
        n_bins: int,
        chi: float = 1.0,
    ) -> Float[Array, " n_bins+1"]:
        """Make ``n_bins`` frequency bins of equal phase change.

        ``n_bins`` must be a positive integer resolved by the caller
        (see :meth:`__init__`).
        """
        phase_diff_array = self._max_phase_diff(freqs, freqs[0], freqs[-1], chi=chi)
        total_phase = phase_diff_array[-1]
        phase_diff = jnp.linspace(0.0, total_phase, n_bins + 1)
        f_bins = interp1d(phase_diff_array, freqs)(phase_diff)
        return jnp.array(f_bins)

    def _mask_and_set_frequency_arrays(
        self,
        waveform: dict[str, Complex[Array, " n_freq"]],
        frequencies: Float[Array, " n_freq"],
    ) -> Float[Array, " n_valid+1"]:
        """
        Mask out trivial waveform pieces which are usually beyond merger frequency.

        This is to avoid creating NaNs from 0/0 when computing the r0 and r1,
        where the ratios between waveforms are taken.

        This setup step trims unsupported frequencies before JIT compilation,
        keeping the evaluation grids fixed and avoiding undefined waveform ratios.
        """
        # --------------------------------------------------------------
        # Reference support validation
        # --------------------------------------------------------------

        self._validate_reference_waveform(waveform, self.frequencies)
        h_amp = jnp.array([jnp.abs(p) for p in waveform.values()]).sum(axis=0)
        support = h_amp > 0
        support_indices = jnp.flatnonzero(support)
        if not len(support_indices):
            raise ValueError(
                "Reference waveform has no support in the frequency range of the likelihood."
            )
        first, last = int(support_indices[0]), int(support_indices[-1])
        if not bool(jnp.all(support[first : last + 1])):
            raise ValueError(
                "Reference waveform must be nonzero inside its supported range."
            )
        _valid_frequencies = self.frequencies[support_indices]
        valid_mask = (frequencies >= _valid_frequencies[0]) & (
            frequencies <= _valid_frequencies[-1]
        )

        masked_frequencies = frequencies[valid_mask]
        if len(masked_frequencies) < 2:
            raise ValueError(
                "Reference waveform support contains fewer than two bin edges; "
                "increase n_bins or choose a supported frequency range."
            )
        if not bool(jnp.all(jnp.diff(masked_frequencies) > 0)):
            raise ValueError("Heterodyne bin edges must be strictly increasing.")
        self.freq_grid = masked_frequencies
        self.n_bins = len(masked_frequencies) - 1

        return masked_frequencies

    @staticmethod
    def _validate_reference_waveform(
        waveform: dict[str, Complex[Array, " n_freq"]],
        frequencies: Float[Array, " n_freq"],
        *,
        label: str = "Reference waveform",
    ) -> None:
        """Reject unsupported reference grids before building likelihood ratios."""
        # --------------------------------------------------------------
        # Reference waveform validation
        # --------------------------------------------------------------

        if not waveform:
            raise ValueError(f"{label} must contain polarizations.")
        for polarization, strain in waveform.items():
            if strain.shape != frequencies.shape or not bool(
                jnp.all(jnp.isfinite(strain))
            ):
                raise ValueError(
                    f"{label} polarization {polarization!r} must contain "
                    "one finite value per frequency. Check the reference parameters "
                    "and whether the waveform supports the bin-edge grid; increasing "
                    "n_bins may be necessary."
                )

    @staticmethod
    def _max_phase_diff(
        freqs: Float[Array, " n_freq"],
        f_low: FloatLike,
        f_high: FloatLike,
        chi: float = 1.0,
    ) -> Float[Array, " n_freq"]:
        """Cumulative phase difference of the reference signal, starting at 0 at ``f_low``."""
        # Five power laws in frequency (arXiv:1806.08792) for the Newtonian chirp, the
        # spin-orbit term, the coalescence time and the merger-ringdown; see also
        # Eq. (7) of arXiv:2302.05333.
        gamma = jnp.array([-5.0, -2.0, 3.0, 5.0, 7.0]) / 3
        freq_2D = jax.lax.broadcast_in_dim(freqs, (freqs.size, gamma.size), [0])
        f_star = jnp.where(gamma >= 0, f_high, f_low)
        summand = (freq_2D / f_star) ** gamma * jnp.sign(gamma)
        dphi = 2 * jnp.pi * chi * jnp.sum(summand, axis=1)
        return dphi - dphi[0]

    def _compute_coefficients(
        self,
        detector: Detector,
        h_ref: Complex[Array, " n_freq"],
        f_bins: Float[Array, " n_valid+1"],
    ) -> Complex[Array, "4 n_valid"]:
        i = self.detectors.index(detector)
        freqs, data, psd = self._detector_data_in_frequency_range(i)

        data_prod = jnp.array(data * h_ref.conj()) / psd
        self_prod = jnp.array(h_ref * h_ref.conj()) / psd

        n_bins = f_bins.shape[0] - 1
        freq_bins_center = (f_bins[:-1] + f_bins[1:]) / 2

        # Assign each frequency sample to its bin: bin k is [f_bins[k], f_bins[k+1]).
        # A per-sample bin index with segment_sum keeps memory at O(n_freq), whereas
        # a dense (n_bins, n_freq) mask matrix is prohibitive for long durations.
        idx = jnp.searchsorted(f_bins, freqs, side="right") - 1
        # The half-open interval excludes any frequency that lands exactly on the
        # upper edge of the last bin (f_bins[-1]).  This happens whenever the
        # interpolated bin edge coincides with the last discrete frequency sample
        # (common when the waveform reaches f_max).  Close the last bin on the right.
        idx = jnp.where(freqs == f_bins[-1], n_bins - 1, idx)
        freq_shift = freqs - freq_bins_center[jnp.clip(idx, 0, n_bins - 1)]

        def _bin_sum(x: Complex[Array, " n_freq"]) -> Complex[Array, " n_valid"]:
            # Drop samples outside the binned range without accumulating them.
            return jax.ops.segment_sum(x, idx, num_segments=n_bins, mode="drop")

        # The resultant arrays have shape (n_valid), the dimension with "n_freq" is summed over.
        summary_data = jnp.array(
            [
                _bin_sum(data_prod),  # A0
                _bin_sum(data_prod * freq_shift),  # A1
                _bin_sum(self_prod),  # B0
                _bin_sum(self_prod * freq_shift),  # B1
            ]
        )

        return 4 / detector.data.duration * summary_data


# ----------------------------------------------------------------------
# Multibanded likelihood
# ----------------------------------------------------------------------


class MultibandedTransientLikelihoodFD(SingleEventLikelihood):
    """Multi-banded likelihood for gravitational wave transient events.

    This implements the multi-banding method described in S. Morisaki, 2021, arXiv:2104.07813.
    The method divides the frequency range into bands with different resolutions,
    using coarser grids at higher frequencies to speed up likelihood evaluation.

    Attributes:
        reference_chirp_mass (Float): Reference chirp mass for determining frequency bands.
        reference_chirp_mass_in_second (Float): Geometrised reference chirp mass in time unit [second].
        highest_mode (int): Maximum magnetic number of GW moments (fixed to 2 for 22-mode).
        accuracy_factor (Float): Parameter L controlling approximation accuracy.
        time_offset (Float): Time offset for band construction.
        delta_f_end (Float): Frequency scale for high-frequency tapering.
        durations (Array): Durations of each band.
        fb_dfb (Array): Starting frequencies and taper widths for each band.
        linear_coeffs (dict): Pre-computed coefficients for (d|h) inner product.
        quadratic_coeffs (dict): Pre-computed coefficients for (h|h) inner product.

    Args:
        detectors (Sequence[Detector]): List of detector objects.
        waveform (Waveform): Waveform model to evaluate.
        fixed_parameters (Optional[dict]): Fixed parameters for the likelihood.
        f_min (Optional[Float | dict[str, Float]]): Minimum frequency for
            likelihood evaluation, or a dict mapping detector name to a per-detector
            Float.  Must be positive and not below the minimum frequency of the
            data.  Defaults to that minimum frequency.
        f_max (Optional[Float | dict[str, Float]]): Maximum frequency for
            likelihood evaluation, or a dict mapping detector name to a per-detector
            Float.  Must not exceed the maximum frequency of the data or the
            Nyquist frequency.  Defaults to the smaller of the two.
        highest_mode (int): Maximum magnetic number (default 2, for 22-mode only).
        accuracy_factor (Float): Accuracy parameter L (default 5.0).
        prior (Optional[Prior]): Combined prior object.  Needed when *reference_chirp_mass*,
            *time_offset*, or *delta_f_end* are ``None`` so they can be inferred
            automatically from the prior bounds.
        reference_chirp_mass (Optional[Float]): Reference chirp mass in solar masses.
            Use the minimum of your chirp-mass prior for maximum speedup.  When
            ``None``, the value is inferred from the ``M_c`` component of *prior*.
        time_offset (Optional[Float]): Time offset in seconds.  When ``None``,
            inferred from the ``t_c`` (or ``t_{ifo}``) prior range; falls back
            to 2.12 s with a warning when the prior is unavailable and the data end
            2 s after the trigger time (otherwise it is an error).
        delta_f_end (Optional[Float]): End frequency taper scale in Hz.  When
            ``None``, inferred from the ``t_c`` prior range; falls back to 53.0 Hz.
        max_banding_frequency (Optional[Float]): Upper limit on band starting frequency.
        min_banding_duration (Float): Minimum duration for bands.
    """

    highest_mode: int
    accuracy_factor: float
    reference_chirp_mass: float
    reference_chirp_mass_in_second: float
    time_offset: float
    delta_f_end: float
    max_banding_frequency: float
    min_banding_duration: float

    durations: Float[Array, " n_bands"]
    fb_dfb: Float[Array, "n_bands+1 2"]

    unique_frequencies: Float[Array, " n_unique"]
    unique_to_original: Array

    linear_coeffs: dict[str, Complex[Array, " n_total_points"]]
    quadratic_coeffs: dict[str, Float[Array, " n_total_points"]]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        *,
        fixed_parameters: Optional[FixedParameters] = None,
        f_min: Optional[float | dict[str, float]] = None,
        f_max: Optional[float | dict[str, float]] = None,
        highest_mode: int = 2,
        accuracy_factor: float = 5.0,
        prior: Optional[Prior] = None,
        reference_chirp_mass: Optional[float] = None,
        time_offset: Optional[float] = None,
        delta_f_end: Optional[float] = None,
        max_banding_frequency: Optional[float] = None,
        min_banding_duration: float = 0.0,
    ):

        super().__init__(
            detectors,
            waveform,
            fixed_parameters=fixed_parameters,
            f_min=f_min,
            f_max=f_max,
        )

        logger.info("Initializing multi-banded likelihood...")

        reference_chirp_mass = self._resolve_reference_chirp_mass(
            reference_chirp_mass, prior
        )
        time_offset, delta_f_end = self._resolve_time_params(
            time_offset, delta_f_end, prior
        )
        # --------------------------------------------------------------
        # Banding validation
        # --------------------------------------------------------------

        self._validate_banding_params(
            reference_chirp_mass,
            highest_mode,
            accuracy_factor,
            time_offset,
            delta_f_end,
            min_banding_duration,
            max_banding_frequency,
        )

        self.reference_chirp_mass = reference_chirp_mass
        self.reference_chirp_mass_in_second = reference_chirp_mass * MTSUN
        self.highest_mode = highest_mode
        self.accuracy_factor = accuracy_factor
        self.time_offset = time_offset
        self.delta_f_end = delta_f_end
        self.min_banding_duration = min_banding_duration

        fmax_spa = (
            (15 / 968) ** (3 / 5)
            * (self.highest_mode / (2 * jnp.pi)) ** (8 / 5)
            / self.reference_chirp_mass_in_second
        )
        self.max_banding_frequency = (
            min(max_banding_frequency, fmax_spa)
            if max_banding_frequency is not None
            else fmax_spa
        )

        self._setup_frequency_bands()
        self._setup_integers()
        self._setup_waveform_frequency_points()
        self._setup_linear_coefficients()
        self._setup_quadratic_coefficients()

        logger.info("Multi-banding setup complete with %d bands", self.n_bands)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def minimum_frequency(self) -> float:
        """Lowest f_min of the likelihood across the detectors."""
        return min(f_min for f_min, _ in self._detector_frequency_ranges)

    @property
    def maximum_frequency(self) -> float:
        """Highest f_max of the likelihood across the detectors."""
        return max(f_max for _, f_max in self._detector_frequency_ranges)

    @property
    def n_bands(self) -> int:
        """Number of frequency bands."""
        return len(self.durations)

    # ------------------------------------------------------------------
    # Direct evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, params: Mapping[str, FloatLike]) -> FloatScalar:
        waveform_sky = self.waveform(self.unique_frequencies, params)
        return self._likelihood(params, waveform_sky)

    # ------------------------------------------------------------------
    # Waveform cache evaluation
    # ------------------------------------------------------------------

    def _generate_waveform(
        self, params: Mapping[str, FloatLike]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Generate polarizations at multiband frequencies for cache reuse.

        Evaluated at unit distance when ``waveform_caches_distance`` is True,
        so ``d_L`` is not a cache dependency; otherwise ``d_L`` is a
        dependency like any other waveform parameter.
        """
        return self._waveform_sky_for_cache(self.unique_frequencies, params)

    def _evaluate_from_waveform(
        self,
        params: Mapping[str, FloatLike],
        waveform_cache: dict[str, Complex[Array, " n_freq"]],
    ) -> FloatScalar:
        """Core likelihood evaluation from a pre-generated waveform cache."""
        waveform_sky = self._waveform_sky_from_cache(waveform_cache, params)
        return self._likelihood(params, waveform_sky)

    # ------------------------------------------------------------------
    # Likelihood computation
    # ------------------------------------------------------------------

    def _likelihood(
        self,
        params: Mapping[str, FloatLike],
        waveform_sky: dict[str, Complex[Array, " n_freq"]],
    ) -> FloatScalar:
        """Core likelihood computation from physical-distance multiband polarizations."""
        log_likelihood: FloatScalar = jnp.zeros(())

        for detector in self.detectors:
            # Get detector response at banded frequencies.
            h_det_unique = detector.fd_response(
                self.unique_frequencies,
                waveform_sky,
                params,
            )
            strain = h_det_unique[self.unique_to_original]
            d_inner_h = jnp.sum(strain * self.linear_coeffs[detector.name])
            h_inner_h = jnp.sum(
                jnp.real(strain * jnp.conj(strain))
                * self.quadratic_coeffs[detector.name]
            )
            log_likelihood += jnp.real(d_inner_h) - h_inner_h / 2

        return log_likelihood

    # ------------------------------------------------------------------
    # Configuration and prior bounds
    # ------------------------------------------------------------------

    def _resolve_reference_chirp_mass(
        self,
        reference_chirp_mass: Optional[float],
        prior: Optional[Prior],
    ) -> float:
        """Return ``reference_chirp_mass``, inferring from the M_c prior minimum when not provided."""
        if reference_chirp_mass is not None:
            return reference_chirp_mass
        if prior is None:
            raise ValueError(
                "Either reference_chirp_mass or a prior with an M_c component must be provided."
            )
        mc_prior = find_specific_prior(prior, "M_c")
        mc_bounds = mc_prior.get_bounds() if mc_prior is not None else None
        if mc_bounds is None:
            raise ValueError(
                "reference_chirp_mass=None but no M_c prior found. "
                "Pass either reference_chirp_mass or a prior with an M_c component."
            )
        mc_min, _ = mc_bounds
        logger.info(
            "reference_chirp_mass inferred from M_c prior minimum: %.4f M_sun", mc_min
        )
        return mc_min

    def _resolve_time_params(
        self,
        time_offset: Optional[float],
        delta_f_end: Optional[float],
        prior: Optional[Prior],
    ) -> tuple[float, float]:
        """Return ``(time_offset, delta_f_end)``, inferring from t_c prior bounds when not provided.

        Inference uses the geocentric coalescence time ``t_c`` only.
        Detector-frame time ``t_det`` is not supported because
        ``t_det = t_c + sky_delay(ra, dec)`` and the delay is sky-position-dependent,
        so ``t_c`` bounds cannot be derived from a ``t_det`` prior at setup time.
        Falls back to defaults (2.12 s, 53.0 Hz) when inference is not possible and
        the data end 2 s after the trigger time, which the defaults assume.
        """
        inferred_to: Optional[float] = None
        inferred_dfe: Optional[float] = None
        t_end = min(d.data.end_time - self.trigger_time for d in self.detectors)

        if prior is not None and (time_offset is None or delta_f_end is None):
            tc_prior = find_specific_prior(prior, "t_c")
            tc_bounds = tc_prior.get_bounds() if tc_prior is not None else None
            if tc_bounds is not None:
                tc_min, tc_max = tc_bounds
                RE_S = EARTH_RADIUS_LIGHT_S
                denom = t_end - tc_max - RE_S

                if denom <= 0:
                    raise ValueError(
                        f"Cannot infer delta_f_end from t_c prior: "
                        f"t_end - xmax - s = {t_end:.4f} - {tc_max:.4f} - {RE_S:.6f} = {denom:.6f} <= 0. "
                        "Check that the t_c prior upper bound is well within the data segment."
                    )
                inferred_to = t_end - tc_min + RE_S
                inferred_dfe = 100.0 / denom

        uses_default = (time_offset is None and inferred_to is None) or (
            delta_f_end is None and inferred_dfe is None
        )
        if uses_default and not jnp.isclose(t_end, 2.0, atol=1e-3):
            raise ValueError(
                "time_offset and delta_f_end cannot be inferred without a t_c prior, "
                "and their defaults (2.12 s, 53.0 Hz) assume that the data end 2 s "
                f"after the trigger time, but these end {t_end:.3f} s after it. Pass "
                "time_offset and delta_f_end, or a prior that contains t_c."
            )

        if time_offset is None:
            if inferred_to is not None:
                time_offset = inferred_to
                logger.info("time_offset inferred from t_c prior: %.4f s", time_offset)
            else:
                time_offset = 2.12
                logger.warning(
                    "time_offset cannot be inferred from prior; using default 2.12 s"
                )

        if delta_f_end is None:
            if inferred_dfe is not None:
                delta_f_end = inferred_dfe
                logger.info("delta_f_end inferred from t_c prior: %.4f Hz", delta_f_end)
            else:
                delta_f_end = 53.0
                logger.warning(
                    "delta_f_end cannot be inferred from prior; using default 53.0 Hz"
                )

        return time_offset, delta_f_end

    def _validate_banding_params(
        self,
        reference_chirp_mass: float,
        highest_mode: int,
        accuracy_factor: float,
        time_offset: float,
        delta_f_end: float,
        min_banding_duration: float,
        max_banding_frequency: Optional[float],
    ) -> None:
        """Validate the related multi-banding configuration values."""
        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if (
            isinstance(highest_mode, bool)
            or not isinstance(highest_mode, Integral)
            or highest_mode <= 0
        ):
            raise ValueError(
                f"highest_mode must be a positive integer, got {highest_mode!r}."
            )
        values = [
            ("reference_chirp_mass", reference_chirp_mass, False),
            ("accuracy_factor", accuracy_factor, False),
            ("time_offset", time_offset, True),
            ("delta_f_end", delta_f_end, False),
            ("min_banding_duration", min_banding_duration, True),
        ]
        if max_banding_frequency is not None:
            values.append(("max_banding_frequency", max_banding_frequency, False))
        for name, value, allow_zero in values:
            try:
                scalar = jnp.asarray(value)
                valid = (
                    scalar.ndim == 0
                    and jnp.issubdtype(scalar.dtype, jnp.number)
                    and not jnp.iscomplexobj(scalar)
                    and bool(jnp.isfinite(scalar))
                    and bool(scalar >= 0 if allow_zero else scalar > 0)
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                comparison = ">= 0" if allow_zero else "> 0"
                raise ValueError(
                    f"{name} must be {comparison} and a finite real scalar, got {value!r}."
                )

    # ------------------------------------------------------------------
    # Frequency bands
    # ------------------------------------------------------------------

    def _compute_tau_dtaudf(self, f: FloatLike) -> tuple[FloatLike, FloatLike]:
        """Compute time-to-merger and its derivative using 0PN formula.

        Parameters
        ----------
        f : Float
            Input frequency in Hz.

        Returns
        -------
        tuple[Float, Float]
            (tau, dtaudf) where tau is time-to-merger in seconds and dtaudf is its derivative (negative, in seconds/Hz).
        """
        f_22 = 2 * f / self.highest_mode
        piMf = self.reference_chirp_mass_in_second * (
            jnp.pi * self.reference_chirp_mass_in_second * f_22
        ) ** (-8 / 3)
        tau = 5 / 256 * piMf
        dtaudf = -5 / 96 * piMf / f
        return tau, dtaudf

    def _find_starting_frequency(
        self, duration: float, f_now: FloatLike
    ) -> tuple[Optional[FloatLike], Optional[FloatLike]]:
        """Find the starting frequency of the next band by bisection.

        Parameters
        ----------
        duration : Float
            Duration of the next band.
        f_now : Float
            Starting frequency of current band.

        Returns
        -------
        tuple[Optional[Float], Optional[Float]]
            (fnext, dfnext) or (None, None) if no valid frequency exists.
        """

        def _is_above_fnext(f):
            # Conditions (10) and (51) of arXiv:2104.07813: the signal fits in the
            # band, and the transition from the current band is smooth.
            tau, dtaudf = self._compute_tau_dtaudf(f)
            cond1 = (
                duration
                - self.time_offset
                - tau
                - self.accuracy_factor * jnp.sqrt(-dtaudf)
            ) > 0
            cond2 = f - 1.0 / jnp.sqrt(-dtaudf) - f_now > 0
            return cond1 and cond2

        fmin, fmax = f_now, self.max_banding_frequency

        if not _is_above_fnext(fmax):
            return None, None

        # Bisection search
        f = (fmin + fmax) / 2.0
        while fmax - fmin > 1e-2 / duration:
            f = (fmin + fmax) / 2.0
            if _is_above_fnext(f):
                fmax = f
            else:
                fmin = f

        _, dtaudf = self._compute_tau_dtaudf(f)
        return f, 1.0 / jnp.sqrt(-dtaudf)

    def _setup_frequency_bands(self) -> None:
        """Set up frequency bands with geometrically decreasing durations.

        Bands have durations T, T/2, T/4, ... where T is the original data duration.

        Sets:
            self.durations: Array of band durations
            self.fb_dfb: Array of [starting_freq, taper_width] for each band
        """
        original_duration = float(self.duration)
        minimum_frequency = self.minimum_frequency
        maximum_frequency = self.maximum_frequency

        durations_list = [original_duration]
        fb_dfb_list = [[minimum_frequency, 0.0]]

        dnext: float = original_duration / 2

        while dnext > max(self.time_offset, self.min_banding_duration):
            f_now, _ = fb_dfb_list[-1]
            fnext, dfnext = self._find_starting_frequency(dnext, f_now)

            if (
                fnext is not None
                and dfnext is not None
                and fnext < min(maximum_frequency, self.max_banding_frequency)
            ):
                durations_list.append(dnext)
                fb_dfb_list.append([fnext, dfnext])
                dnext /= 2
            else:
                break

        # Add final boundary
        fb_dfb_list.append([maximum_frequency + self.delta_f_end, self.delta_f_end])

        self.durations = jnp.array(durations_list)
        self.fb_dfb = jnp.array(fb_dfb_list)

        logger.info(
            f"Frequency range divided into {self.n_bands} bands with "
            f"intervals: {', '.join(['1/' + str(d) + ' Hz' for d in durations_list])}"
        )

    def _setup_integers(self) -> None:
        """Set up integer indices for each band.

        Sets:
            self.Nbs: Number of samples in downsampled data per band
            self.Mbs: Number of samples in shortened data per band
            self.Ks_Ke: Start/end frequency indices per band
        """
        original_duration = float(self.duration)
        durations = self.durations.tolist()
        fb_dfb = self.fb_dfb.tolist()

        Nbs_list = []
        Mbs_list = []
        Ks_Ke_list = []

        for b in range(self.n_bands):
            dnow = durations[b]
            f_now, dfnow = fb_dfb[b]
            fnext = fb_dfb[b + 1][0]

            Nb = max(
                round_up_to_power_of_two(int(2.0 * fnext * original_duration + 1)),
                2**b,
            )
            Nbs_list.append(Nb)
            Mbs_list.append(Nb // (2**b))
            Ks_Ke_list.append(
                [jnp.ceil((f_now - dfnow) * dnow), jnp.floor(fnext * dnow)]
            )

        self.Nbs = jnp.array(Nbs_list, dtype=jnp.int32)
        self.Mbs = jnp.array(Mbs_list, dtype=jnp.int32)
        self.Ks_Ke = jnp.array(Ks_Ke_list, dtype=jnp.int32)

    def _setup_waveform_frequency_points(self) -> None:
        """Set up frequency points where waveforms are evaluated.

        Creates banded frequency points and finds unique frequencies to avoid
        redundant waveform evaluations.

        Sets:
            self.banded_frequency_points: All frequency points across bands
            self.start_end_idxs: Start/end indices for each band
            self.unique_frequencies: Unique frequencies for waveform evaluation
            self.unique_to_original: Mapping from unique back to banded
        """
        durations = self.durations.tolist()
        Ks_Ke = self.Ks_Ke.tolist()

        band_freqs_list = []
        start_end_list = []
        start_idx = 0

        for b in range(self.n_bands):
            Ks, Ke = Ks_Ke[b]
            band_freqs = jnp.arange(Ks, Ke + 1) / durations[b]
            band_freqs_list.append(band_freqs)
            end_idx = start_idx + Ke - Ks
            start_end_list.append([start_idx, end_idx])
            start_idx = end_idx + 1

        banded_freq_array = jnp.concatenate(band_freqs_list)
        unique_freqs, idxs = jnp.unique(banded_freq_array, return_inverse=True)

        self.banded_frequency_points = banded_freq_array
        self.start_end_idxs = jnp.array(start_end_list, dtype=jnp.int32)
        self.unique_frequencies = unique_freqs
        self.unique_to_original = idxs.astype(jnp.int32)

    # ------------------------------------------------------------------
    # Band windows
    # ------------------------------------------------------------------

    def _get_window_sequence(
        self, delta_f: float, start_idx: int, length: int, band: int
    ) -> Array:
        """Compute cosine-tapered window function for a frequency band.

        Window is 1 in band interior, with smooth cosine tapers at edges.

        Parameters
        ----------
        delta_f : Float
            Frequency interval.
        start_idx : int
            Starting frequency index (frequency = start_idx * delta_f).
        length : int
            Number of frequency points.
        band : int
            Band index.

        Returns
        -------
        Array
            Window sequence of given length.
        """

        f_now, dfnow = self.fb_dfb[band].tolist()
        fnext, dfnext = self.fb_dfb[band + 1].tolist()

        window = jnp.zeros(length)

        increase_start = max(
            0, min(length, int(jnp.floor((f_now - dfnow) / delta_f)) - start_idx + 1)
        )
        unity_start = max(0, min(length, int(jnp.ceil(f_now / delta_f)) - start_idx))
        decrease_start = max(
            0, min(length, int(jnp.floor((fnext - dfnext) / delta_f)) - start_idx + 1)
        )
        decrease_stop = max(0, min(length, int(jnp.ceil(fnext / delta_f)) - start_idx))

        window = window.at[unity_start:decrease_start].set(1.0)

        if increase_start < unity_start and dfnow > 0:
            frequencies = (
                jnp.arange(increase_start, unity_start) + start_idx
            ) * delta_f
            window = window.at[increase_start:unity_start].set(
                (1.0 + jnp.cos(jnp.pi * (frequencies - f_now) / dfnow)) / 2.0
            )

        if decrease_start < decrease_stop:
            frequencies = (
                jnp.arange(decrease_start, decrease_stop) + start_idx
            ) * delta_f
            window = window.at[decrease_start:decrease_stop].set(
                (1.0 - jnp.cos(jnp.pi * (frequencies - fnext) / dfnext)) / 2.0
            )

        return window

    # ------------------------------------------------------------------
    # Likelihood coefficients
    # ------------------------------------------------------------------

    def _setup_linear_coefficients(self) -> None:
        """Pre-compute coefficients for (d|h) inner product.

        For each band:
        1. Apply frequency mask and divide by PSD
        2. IFFT to time domain, take last M^(b) samples
        3. FFT back to get shortened data
        4. Multiply by window and normalization factor

        Sets:
            self.linear_coeffs: Per-detector coefficients across the banded frequency grid
        """
        Ks_Ke = self.Ks_Ke.tolist()
        Nbs = self.Nbs.tolist()
        Mbs = self.Mbs.tolist()
        durations = self.durations.tolist()
        N = Nbs[-1]

        self.linear_coeffs = {}

        for i, detector in enumerate(self.detectors):
            logger.info(f"Pre-computing linear coefficients for {detector.name}")
            _, data_fd, psd = self._detector_data_in_frequency_range(i)
            weighted_data = jnp.zeros(detector.data.n_freq, dtype=data_fd.dtype)
            weighted_data = weighted_data.at[self._detector_data_selections[i]].set(
                data_fd / psd
            )
            valid_len = min(len(weighted_data), N // 2 + 1)
            fddata = (
                jnp.zeros(N // 2 + 1, dtype=complex)
                .at[:valid_len]
                .set(weighted_data[:valid_len])
            )

            band_coeffs = []

            for b in range(self.n_bands):
                Ks, Ke = Ks_Ke[b]
                Nb = Nbs[b]
                Mb = Mbs[b]
                db = durations[b]

                window = self._get_window_sequence(1.0 / db, Ks, Ke - Ks + 1, b)

                fddata_band = fddata[: Nb // 2 + 1].at[-1].set(0.0)

                tddata = jnp.fft.irfft(fddata_band)[-Mb:]
                fddata_shortened = jnp.fft.rfft(tddata)[Ks : Ke + 1]

                band_coeffs.append((4.0 / db) * window * jnp.conj(fddata_shortened))

            self.linear_coeffs[detector.name] = jnp.concatenate(band_coeffs)

    def _setup_quadratic_coefficients(self) -> None:
        """Pre-compute coefficients for (h|h) using linear interpolation.

        For each band and coarse frequency point, compute the weighted sum
        of 1/PSD values using linear interpolation weights.

        Sets:
            self.quadratic_coeffs: Per-detector coefficients on the unique frequency grid
        """

        original_duration = float(self.duration)
        start_end_idxs = self.start_end_idxs.tolist()
        durations = self.durations.tolist()
        fb_dfb = self.fb_dfb.tolist()
        banded_frequency_points = self.banded_frequency_points.tolist()

        self.quadratic_coeffs = {}

        for i, detector in enumerate(self.detectors):
            _, _, psd = self._detector_data_in_frequency_range(i)
            inverse_psd = jnp.zeros(detector.data.n_freq, dtype=psd.dtype)
            inverse_psd = inverse_psd.at[self._detector_data_selections[i]].set(1 / psd)

            band_coeffs = []

            for b in range(self.n_bands):
                logger.debug(f"Pre-computing quadratic coefficients for band {b}")

                start_idx, end_idx = start_end_idxs[b]
                banded_freqs = banded_frequency_points[start_idx : end_idx + 1]
                prefactor = 4 * durations[b] / original_duration

                f_now, dfnow = fb_dfb[b]
                fnext = fb_dfb[b + 1][0]
                start_idx_orig = int(jnp.ceil((f_now - dfnow) * original_duration))
                window_length = (
                    int(jnp.floor(fnext * original_duration)) - start_idx_orig + 1
                )

                window = self._get_window_sequence(
                    1.0 / original_duration, start_idx_orig, window_length, b
                )

                # Compute window / PSD
                end_idx_orig = min(
                    start_idx_orig + len(window) - 1, len(inverse_psd) - 1
                )
                valid_len = end_idx_orig - start_idx_orig + 1

                window_over_psd = (
                    inverse_psd[start_idx_orig : end_idx_orig + 1] * window[:valid_len]
                )

                # Compute coefficients using linear interpolation
                n_coeff = len(banded_freqs)
                coeffs = jnp.zeros(n_coeff)

                for k in range(n_coeff - 1):
                    sum_start = (
                        start_idx_orig
                        if k == 0
                        else max(
                            start_idx_orig,
                            int(jnp.ceil(original_duration * banded_freqs[k])),
                        )
                    )
                    sum_end = (
                        end_idx_orig
                        if k == n_coeff - 2
                        else min(
                            end_idx_orig,
                            int(jnp.ceil(original_duration * banded_freqs[k + 1])) - 1,
                        )
                    )

                    freqs_in_sum = (
                        jnp.arange(sum_start, sum_end + 1) / original_duration
                    )
                    local_start = sum_start - start_idx_orig
                    local_end = sum_end - start_idx_orig + 1
                    wop = window_over_psd[local_start:local_end]

                    coeffs = coeffs.at[k].add(
                        prefactor * jnp.sum((banded_freqs[k + 1] - freqs_in_sum) * wop)
                    )
                    coeffs = coeffs.at[k + 1].add(
                        prefactor * jnp.sum((freqs_in_sum - banded_freqs[k]) * wop)
                    )

                band_coeffs.append(coeffs)

            self.quadratic_coeffs[detector.name] = jnp.concatenate(band_coeffs)


# ----------------------------------------------------------------------
# Likelihood presets
# ----------------------------------------------------------------------

likelihood_presets = {
    "TransientLikelihoodFD": TransientLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
    "MultibandedTransientLikelihoodFD": MultibandedTransientLikelihoodFD,
}
