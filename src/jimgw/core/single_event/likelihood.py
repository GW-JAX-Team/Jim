import logging
from typing import Callable, Optional, Sequence, Union
from abc import abstractmethod
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from scipy.interpolate import interp1d
from evosax.algorithms import CMA_ES
from ripplegw.interfaces import Waveform

from jimgw.core.utils import log_i0
from jimgw.core.prior import Prior
from jimgw.core.base import LikelihoodBase
from jimgw.core.transforms import NtoMTransform
from jimgw.core.single_event.detector import Detector
from jimgw.core.single_event.utils import (
    inner_product,
    complex_inner_product,
    apply_fixed_parameters,
)
from jimgw.core.single_event.marginalization_config import (
    PhaseMargConfig,
    TimeMargConfig,
    DistanceMargConfig,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.constants import MTSUN, EARTH_RADIUS_LIGHT_S

logger = logging.getLogger(__name__)


class SingleEventLikelihood(LikelihoodBase):
    detectors: Sequence[Detector]
    waveform: Waveform
    fixed_parameters: dict[
        str, Float | Callable[[dict[str, Float]], Float | dict[str, Float]]
    ]

    @property
    def duration(self) -> Float:
        """Duration of the data segment in seconds (taken from the first detector)."""
        return self.detectors[0].data.duration

    @property
    def detector_names(self) -> list[str]:
        """Names of the detectors used in this likelihood."""
        return [detector.name for detector in self.detectors]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[
            dict[
                str,
                Float | Callable[[dict[str, Float]], Float | dict[str, Float]],
            ]
        ] = None,
    ) -> None:
        """
        Args:
            detectors (Sequence[Detector]): Detectors with initialized data and PSD.
            waveform (Waveform): Waveform model to evaluate.
            fixed_parameters (Optional[dict]): Parameters held constant during
                sampling. Values may be scalars or callables
                ``f(params) -> Float | dict``; callables are applied in insertion
                order. Defaults to None (no fixed parameters).

        Raises:
            ValueError: If any detector has uninitialized data or PSD.
        """
        # Check that all detectors have initialized data and PSD
        for detector in detectors:
            if detector.data.is_empty:
                raise ValueError(
                    f"Detector '{detector.name}' does not have initialized data. "
                    f"Please set data using detector.set_data() or detector.inject_signal() "
                    f"before initializing the likelihood."
                )
            if detector.psd.is_empty:
                raise ValueError(
                    f"Detector '{detector.name}' does not have initialized PSD. "
                    f"Please set PSD using detector.set_psd() or detector.load_and_set_psd() "
                    f"before initializing the likelihood."
                )

        self.detectors = detectors
        self.waveform = waveform
        self.fixed_parameters = fixed_parameters if fixed_parameters is not None else {}

    def evaluate(self, params: dict[str, Float]) -> Float:
        """Apply ``fixed_parameters`` overrides and evaluate the likelihood.

        Constants are injected directly; callables receive the current params
        dict and may return a scalar or a dict (the matching key is extracted).
        Callables are applied in insertion order.
        """
        params = params.copy()
        apply_fixed_parameters(params, self.fixed_parameters)
        return self._likelihood(params)

    @abstractmethod
    def _likelihood(self, params: dict[str, Float]) -> Float:
        """Core likelihood evaluation method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")


class ZeroLikelihood(LikelihoodBase):
    """Trivial likelihood that always returns zero.

    Useful for prior-only sampling or debugging.
    """

    def __init__(self) -> None:
        pass

    def evaluate(self, params: dict[str, Float]) -> Float:
        """Return zero regardless of the parameters.

        Args:
            params (dict[str, Float]): Ignored.

        Returns:
            Float: Always 0.0.
        """
        return 0.0


# ---------------------------------------------------------------------------
# Unified transient likelihood
# ---------------------------------------------------------------------------


class TransientLikelihoodFD(SingleEventLikelihood):
    """Frequency-domain transient gravitational wave likelihood.

    Supports optional analytic marginalization over coalescence time, phase,
    and/or luminosity distance via typed config objects.  Each marginalization
    mode is activated by passing the corresponding config object (or a plain
    dict shorthand) to the relevant parameter.

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        fixed_parameters: Parameters held constant during sampling.  Values
            may be constants or callables ``f(params) -> Float | dict``;
            callables are applied in insertion order.  See the likelihood
            tutorial for details and examples.
        f_min: Minimum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.
        f_max: Maximum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.
        trigger_time: GPS time of the event trigger.
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
        ...     f_min=20, f_max=1024, trigger_time=1234567890,
        ...     phase_marginalization=True,
        ...     time_marginalization={"tc_range": (-0.1, 0.1)},
        ... )
        >>> logL = likelihood.evaluate(params)
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[
            dict[
                str,
                Float | Callable[[dict[str, Float]], Float | dict[str, Float]],
            ]
        ] = None,
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = jnp.inf,
        trigger_time: Float = 0,
        time_marginalization: Optional[Union[TimeMargConfig, dict, bool]] = None,
        phase_marginalization: Optional[Union[PhaseMargConfig, dict, bool]] = None,
        distance_marginalization: Optional[
            Union[DistanceMargConfig, dict, bool]
        ] = None,
    ) -> None:
        super().__init__(detectors, waveform, fixed_parameters)

        # --- coerce marginalization inputs ---
        if isinstance(time_marginalization, (bool, dict)):
            if time_marginalization is False:
                time_marginalization = None
            else:
                time_marginalization = TimeMargConfig(
                    **(
                        time_marginalization
                        if isinstance(time_marginalization, dict)
                        else {}
                    )
                )
        if isinstance(phase_marginalization, (bool, dict)):
            if phase_marginalization is False:
                phase_marginalization = None
            else:
                phase_marginalization = PhaseMargConfig(
                    **(
                        phase_marginalization
                        if isinstance(phase_marginalization, dict)
                        else {}
                    )
                )
        if isinstance(distance_marginalization, (bool, dict)):
            if distance_marginalization is True:
                raise ValueError(
                    "distance_marginalization=True is not supported because "
                    "`distance_prior` has no default.  Pass a dict with `distance_prior` "
                    "or a DistanceMargConfig instance instead."
                )
            elif isinstance(distance_marginalization, dict):
                distance_marginalization = DistanceMargConfig(
                    **distance_marginalization
                )
            else:  # False
                distance_marginalization = None

        # --- frequency setup (from former BaseTransientLikelihoodFD) ---
        _frequencies = []
        for detector in detectors:
            f_min_ifo = f_min[detector.name] if isinstance(f_min, dict) else f_min
            f_max_ifo = f_max[detector.name] if isinstance(f_max, dict) else f_max
            detector.set_frequency_bounds(f_min_ifo, f_max_ifo)
            _frequencies.append(detector.sliced_frequencies)

        assert all(
            jnp.isclose(
                _frequencies[0][1] - _frequencies[0][0],
                freq[1] - freq[0],
            )
            for freq in _frequencies
        ), "All detectors must have the same frequency spacing."

        self.df = _frequencies[0][1] - _frequencies[0][0]
        self.frequencies = jnp.unique(jnp.concatenate(_frequencies))
        self.frequency_masks = [
            jnp.isin(self.frequencies, detector.sliced_frequencies)
            for detector in detectors
        ]

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(self.trigger_time)

        # --- marginalization flags ---
        self.time_marginalization = time_marginalization is not None
        self.phase_marginalization = phase_marginalization is not None
        self.distance_marginalization = distance_marginalization is not None

        if self.time_marginalization and self.distance_marginalization:
            raise NotImplementedError(
                "Joint time + distance marginalization is not yet supported."
            )

        if time_marginalization is not None:
            self._init_time_marginalization(time_marginalization.tc_range)
        if self.phase_marginalization:
            self._init_phase_marginalization()
        if distance_marginalization is not None:
            self._init_distance_marginalization(
                distance_marginalization.distance_prior,
                distance_marginalization.n_dist_points,
                distance_marginalization.ref_dist,
            )

    def evaluate(self, params: dict[str, Float]) -> Float:
        params = params.copy()
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        if self.time_marginalization:
            params["t_c"] = 0.0
        if self.phase_marginalization:
            params["phase_c"] = 0.0
        if self.distance_marginalization:
            params["d_L"] = self.ref_dist
        apply_fixed_parameters(params, self.fixed_parameters)
        return self._likelihood(params)

    def _likelihood(self, params: dict[str, Float]) -> Float:
        waveform_sky = self.waveform(self.frequencies, params)

        # --- choose accumulation type based on flags ---
        if self.time_marginalization:
            # Per-frequency complex array for FFT-based time marginalization
            complex_d_inner_h = jnp.zeros(len(self.frequencies), dtype=jnp.complex128)
            log_likelihood = 0.0

            for i, ifo in enumerate(self.detectors):
                psd = ifo.sliced_psd
                waveform_sky_ifo = {
                    key: waveform_sky[key][self.frequency_masks[i]]
                    for key in waveform_sky
                }
                h_dec = ifo.fd_response(
                    ifo.sliced_frequencies, waveform_sky_ifo, params
                )
                complex_d_inner_h = complex_d_inner_h.at[self.frequency_masks[i]].add(
                    4 * h_dec * jnp.conj(ifo.sliced_fd_data) / psd * self.df
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
            complex_d_inner_h = 0.0 + 0.0j
            match_filter_snr = 0.0
            optimal_snr = 0.0

            for i, ifo in enumerate(self.detectors):
                psd = ifo.sliced_psd
                waveform_sky_ifo = {
                    key: waveform_sky[key][self.frequency_masks[i]]
                    for key in waveform_sky
                }
                h_dec = ifo.fd_response(
                    ifo.sliced_frequencies, waveform_sky_ifo, params
                )
                if self.phase_marginalization:
                    complex_d_inner_h += complex_inner_product(
                        h_dec, ifo.sliced_fd_data, psd, self.df
                    )
                else:
                    match_filter_snr += inner_product(
                        h_dec, ifo.sliced_fd_data, psd, self.df
                    )
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
            log_likelihood = 0.0
            for i, ifo in enumerate(self.detectors):
                psd = ifo.sliced_psd
                waveform_sky_ifo = {
                    key: waveform_sky[key][self.frequency_masks[i]]
                    for key in waveform_sky
                }
                h_dec = ifo.fd_response(
                    ifo.sliced_frequencies, waveform_sky_ifo, params
                )
                match_filter_SNR = inner_product(
                    h_dec, ifo.sliced_fd_data, psd, self.df
                )
                optimal_SNR = inner_product(h_dec, h_dec, psd, self.df)
                log_likelihood += match_filter_SNR - optimal_SNR / 2
            return log_likelihood

    # --- time marginalization helpers ---

    def _init_time_marginalization(self, tc_range: tuple[Float, Float]) -> None:
        if "t_c" in self.fixed_parameters:
            raise ValueError("Cannot have t_c fixed while marginalizing over t_c")
        self.tc_range = tc_range
        fs = self.detectors[0].data.sampling_frequency
        duration = self.detectors[0].data.duration
        self.tc_array = jnp.fft.fftfreq(int(duration * fs / 2), 1.0 / duration)
        self.pad_low = jnp.zeros(int(self.frequencies[0] * duration))
        n_pad_high = int(
            (fs / 2.0 - 1.0 / duration - float(self.frequencies[-1])) * duration
        )
        self.pad_high = jnp.zeros(max(0, n_pad_high))

    def _reduce_time(self, complex_d_inner_h: Float[Array, " n_freq"]) -> Float:
        """FFT-based time marginalization (real part)."""
        complex_d_inner_h_positive_f = jnp.concatenate(
            (self.pad_low, complex_d_inner_h, self.pad_high)
        )
        fft_d_inner_h = jnp.fft.fft(complex_d_inner_h_positive_f, norm="backward")
        fft_d_inner_h = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            fft_d_inner_h.real,
            jnp.zeros_like(fft_d_inner_h.real) - jnp.inf,
        )
        return logsumexp(fft_d_inner_h) - jnp.log(len(self.tc_array))

    # --- phase marginalization helpers ---

    def _init_phase_marginalization(self) -> None:
        if "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

    def _reduce_phase(self, complex_d_inner_h: complex, optimal_snr: Float) -> Float:
        """Phase marginalization via modified Bessel function (Thrane & Talbot 2019, Eq. 24)."""
        return -optimal_snr / 2 + log_i0(jnp.absolute(complex_d_inner_h))

    # --- distance marginalization helpers ---

    def _init_distance_marginalization(
        self,
        distance_prior: Prior,
        n_dist_points: int,
        ref_dist: Optional[float],
    ) -> None:
        if "d_L" in self.fixed_parameters:
            raise ValueError("Cannot have d_L fixed while marginalising over d_L")

        if list(distance_prior.parameter_names) != ["d_L"]:
            raise ValueError(
                f"distance_prior must be a 1D prior with parameter_names=['d_L'], "
                f"got parameter_names={list(distance_prior.parameter_names)}."
            )

        if not hasattr(distance_prior, "xmin") or not hasattr(distance_prior, "xmax"):
            raise ValueError(
                "The d_L sub-prior must have xmin and xmax attributes. "
                "Use a bounded prior such as PowerLawPrior or UniformPrior."
            )

        dist_min = float(getattr(distance_prior, "xmin"))
        dist_max = float(getattr(distance_prior, "xmax"))

        if dist_min <= 0:
            raise ValueError(
                "The d_L prior's xmin must be > 0 (distance must be positive)"
            )
        if dist_max <= dist_min:
            raise ValueError("The d_L prior's xmax must be greater than xmin")

        if n_dist_points < 2:
            raise ValueError("n_dist_points must be at least 2")

        if ref_dist is None:
            self.ref_dist = (dist_min + dist_max) / 2.0
        else:
            if ref_dist <= 0:
                raise ValueError("ref_dist must be > 0")
            self.ref_dist = ref_dist

        distance_grid = jnp.linspace(dist_min, dist_max, n_dist_points)
        delta_d = (dist_max - dist_min) / (n_dist_points - 1)
        self.scaling = self.ref_dist / distance_grid

        log_prob_fn = jax.vmap(lambda d: distance_prior.log_prob({"d_L": d}))
        log_w = log_prob_fn(distance_grid) + jnp.log(delta_d)
        self.log_weights = log_w - logsumexp(log_w)

    def _reduce_distance(self, match_filter_snr: Float, optimal_snr: Float) -> Float:
        """Distance marginalization using scaling + logsumexp."""
        log_integrand = (
            match_filter_snr * self.scaling
            - 0.5 * optimal_snr * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)

    # --- combined marginalization helpers ---

    def _reduce_phase_time(self, complex_d_inner_h: Float[Array, " n_freq"]) -> Float:
        """FFT-based time + phase marginalization (Bessel-weighted FFT)."""
        complex_d_inner_h_positive_f = jnp.concatenate(
            (self.pad_low, complex_d_inner_h, self.pad_high)
        )
        fft_d_inner_h = jnp.fft.fft(complex_d_inner_h_positive_f, norm="backward")
        log_i0_abs_fft = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            log_i0(jnp.absolute(fft_d_inner_h)),
            jnp.zeros_like(fft_d_inner_h.real) - jnp.inf,
        )
        return logsumexp(log_i0_abs_fft) - jnp.log(len(self.tc_array))

    def _reduce_phase_distance(
        self, complex_d_inner_h: complex, optimal_snr: Float
    ) -> Float:
        """Phase + distance marginalization (Thrane & Talbot 2019, Eq. 79)."""
        abs_kappa = jnp.absolute(complex_d_inner_h)
        log_integrand = (
            log_i0(abs_kappa * self.scaling)
            - 0.5 * optimal_snr * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)


# ---------------------------------------------------------------------------
# Heterodyned (relative-binning) likelihood
# ---------------------------------------------------------------------------


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
        f_max: Maximum frequency for likelihood evaluation.
        trigger_time: GPS time of the event trigger.
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
    reference_parameters: dict
    freq_grid_low: Array
    freq_grid_high: Array
    freq_grid_center: Array
    bin_widths: Array
    waveform_low_ref: dict[str, Float[Array, " n_bin"]]
    waveform_high_ref: dict[str, Float[Array, " n_bin"]]
    A0_array: dict[str, Float[Array, " n_bin"]]
    A1_array: dict[str, Float[Array, " n_bin"]]
    B0_array: dict[str, Float[Array, " n_bin"]]
    B1_array: dict[str, Float[Array, " n_bin"]]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[
            dict[
                str,
                Float | Callable[[dict[str, Float]], Float | dict[str, Float]],
            ]
        ] = None,
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = jnp.inf,
        trigger_time: float = 0,
        n_bins: Optional[int] = None,
        epsilon: Optional[float] = None,
        optimizer_popsize: int = 500,
        optimizer_n_steps: int = 1000,
        reference_parameters: Optional[dict] = None,
        reference_waveform: Optional[Waveform] = None,
        prior: Optional[Prior] = None,
        likelihood_transforms: Optional[list[NtoMTransform]] = None,
        phase_marginalization: Optional[Union[PhaseMargConfig, dict, bool]] = None,
    ):
        super().__init__(detectors, waveform, fixed_parameters)

        # --- coerce phase marginalization input ---
        if isinstance(phase_marginalization, bool):
            phase_marginalization = PhaseMargConfig() if phase_marginalization else None
        elif isinstance(phase_marginalization, dict):
            phase_marginalization = PhaseMargConfig(**phase_marginalization)
        self.phase_marginalization = phase_marginalization is not None

        # --- frequency setup (same as TransientLikelihoodFD) ---
        _frequencies = []
        for detector in detectors:
            f_min_ifo = f_min[detector.name] if isinstance(f_min, dict) else f_min
            f_max_ifo = f_max[detector.name] if isinstance(f_max, dict) else f_max
            detector.set_frequency_bounds(f_min_ifo, f_max_ifo)
            _frequencies.append(detector.sliced_frequencies)

        assert all(
            jnp.isclose(
                _frequencies[0][1] - _frequencies[0][0],
                freq[1] - freq[0],
            )
            for freq in _frequencies
        ), "All detectors must have the same frequency spacing."

        self.df = _frequencies[0][1] - _frequencies[0][0]
        self.frequencies = jnp.unique(jnp.concatenate(_frequencies))
        self.frequency_masks = [
            jnp.isin(self.frequencies, detector.sliced_frequencies)
            for detector in detectors
        ]

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(self.trigger_time)

        # --- phase marginalization flag ---
        if self.phase_marginalization and "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

        # --- heterodyne setup ---
        logger.info("Initializing heterodyned likelihood..")

        if reference_parameters is None:
            reference_parameters = {}
        if likelihood_transforms is None:
            likelihood_transforms = []

        if reference_waveform is None:
            reference_waveform = waveform

        if reference_parameters:
            self.reference_parameters = reference_parameters.copy()
            apply_fixed_parameters(self.reference_parameters, self.fixed_parameters)
            logger.info(
                f"Reference parameters provided, which are {self.reference_parameters}"
            )
        elif prior:
            logger.info("No reference parameters are provided, finding it...")
            reference_parameters = self.maximize_likelihood(
                prior=prior,
                likelihood_transforms=likelihood_transforms,
                optimizer_popsize=optimizer_popsize,
                optimizer_n_steps=optimizer_n_steps,
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

        self.reference_parameters["trigger_time"] = self.trigger_time
        self.reference_parameters["gmst"] = self.gmst

        self.waveform_low_ref = {}
        self.waveform_high_ref = {}
        self.A0_array = {}
        self.A1_array = {}
        self.B0_array = {}
        self.B1_array = {}

        frequency_original = self.frequencies
        if n_bins is not None and epsilon is not None:
            raise ValueError(
                "'n_bins' and 'epsilon' are mutually exclusive; specify at most one."
            )
        if epsilon is not None and epsilon <= 0:
            raise ValueError(f"'epsilon' must be a positive number, got {epsilon!r}.")
        if n_bins is not None and n_bins <= 0:
            raise ValueError(f"'n_bins' must be a positive integer, got {n_bins!r}.")
        if epsilon is None and n_bins is None:
            epsilon = 0.5
        if epsilon is not None:
            freqs_arr = jnp.array(frequency_original)
            phase = HeterodynedTransientLikelihoodFD._max_phase_diff(
                freqs_arr, freqs_arr[0], freqs_arr[-1]
            )
            n_bins = max(1, int(float(phase[-1]) / epsilon))
        assert isinstance(n_bins, int)
        freq_grid, self.freq_grid_center = self._make_binning_scheme(
            jnp.array(frequency_original), n_bins=n_bins
        )
        self.freq_grid_low = freq_grid[:-1]
        self.freq_grid_high = freq_grid[1:]

        h_sky = reference_waveform(frequency_original, self.reference_parameters)

        h_amp = jnp.sum(
            jnp.array([jnp.abs(h_sky[pol]) for pol in h_sky.keys()]), axis=0
        )
        f_valid = frequency_original[jnp.where(h_amp > 0)[0]]
        f_waveform_max = jnp.max(f_valid)
        f_waveform_min = jnp.min(f_valid)

        mask_heterodyne_center = jnp.where(
            (self.freq_grid_high <= f_waveform_max)
            & (self.freq_grid_center >= f_waveform_min)
        )[0]
        self.freq_grid_center = self.freq_grid_center[mask_heterodyne_center]
        self.freq_grid_low = self.freq_grid_low[mask_heterodyne_center]
        self.freq_grid_high = self.freq_grid_high[mask_heterodyne_center]
        self.bin_widths = self.freq_grid_high - self.freq_grid_low
        self.n_bins = len(self.freq_grid_center)

        start_idx = mask_heterodyne_center[0]
        end_idx = mask_heterodyne_center[-1] + 2
        freq_grid = freq_grid[start_idx:end_idx]

        h_sky_low = reference_waveform(self.freq_grid_low, self.reference_parameters)
        h_sky_high = reference_waveform(self.freq_grid_high, self.reference_parameters)

        for i, detector in enumerate(self.detectors):
            h_sky_ifo = {key: h_sky[key][self.frequency_masks[i]] for key in h_sky}
            waveform_ref = detector.fd_response(
                detector.sliced_frequencies, h_sky_ifo, self.reference_parameters
            )
            self.waveform_low_ref[detector.name] = detector.fd_response(
                self.freq_grid_low, h_sky_low, self.reference_parameters
            )
            self.waveform_high_ref[detector.name] = detector.fd_response(
                self.freq_grid_high, h_sky_high, self.reference_parameters
            )
            A0, A1, B0, B1 = self._compute_coefficients(
                detector.sliced_fd_data,
                waveform_ref,
                detector.sliced_psd,
                detector.sliced_frequencies,
                freq_grid,
                self.freq_grid_center,
            )
            self.A0_array[detector.name] = A0[mask_heterodyne_center]
            self.A1_array[detector.name] = A1[mask_heterodyne_center]
            self.B0_array[detector.name] = B0[mask_heterodyne_center]
            self.B1_array[detector.name] = B1[mask_heterodyne_center]

    def evaluate(self, params: dict[str, Float]) -> Float:
        params = params.copy()
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        if self.phase_marginalization:
            params["phase_c"] = 0.0
        apply_fixed_parameters(params, self.fixed_parameters)
        return self._likelihood(params)

    def _likelihood(self, params: dict[str, Float]) -> Float:
        frequencies_low = self.freq_grid_low
        frequencies_high = self.freq_grid_high
        log_likelihood = 0.0
        waveform_sky_low = self.waveform(frequencies_low, params)
        waveform_sky_high = self.waveform(frequencies_high, params)

        complex_d_inner_h = 0.0 + 0.0j

        for detector in self.detectors:
            waveform_low = detector.fd_response(
                frequencies_low, waveform_sky_low, params
            )
            waveform_high = detector.fd_response(
                frequencies_high, waveform_sky_high, params
            )

            r_low = waveform_low / self.waveform_low_ref[detector.name]
            r_high = waveform_high / self.waveform_high_ref[detector.name]
            r0 = (r_low + r_high) / 2
            r1 = (r_high - r_low) / self.bin_widths

            if self.phase_marginalization:
                complex_d_inner_h += jnp.sum(
                    self.A0_array[detector.name] * r0.conj()
                    + self.A1_array[detector.name] * r1.conj()
                )
                optimal_SNR = jnp.sum(
                    self.B0_array[detector.name] * jnp.abs(r0) ** 2
                    + 2 * self.B1_array[detector.name] * (r0 * r1.conj()).real
                )
                log_likelihood += -optimal_SNR.real / 2
            else:
                match_filter_SNR = jnp.sum(
                    self.A0_array[detector.name] * r0.conj()
                    + self.A1_array[detector.name] * r1.conj()
                )
                optimal_SNR = jnp.sum(
                    self.B0_array[detector.name] * jnp.abs(r0) ** 2
                    + 2 * self.B1_array[detector.name] * (r0 * r1.conj()).real
                )
                log_likelihood += (match_filter_SNR - optimal_SNR / 2).real

        if self.phase_marginalization:
            log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))

        return log_likelihood

    @staticmethod
    def _max_phase_diff(
        freqs: Float[Array, " n_freq"],
        f_low: float | Float[Array, ""],
        f_high: float | Float[Array, ""],
        chi: float = 1.0,
    ) -> Float[Array, " n_freq"]:
        """
        Compute the cumulative phase difference used for bin construction.

        Uses 5 physically-motivated PN/IMR terms from arXiv:1806.08792:
        gamma ∈ {-5/3, -2/3, 1, 5/3, 7/3}, covering the dominant Newtonian
        chirp (0PN), spin-orbit (1.5PN), coalescence time, and phenomenological
        IMR contributions.  Each term is normalised by ``d_alpha`` so that its
        individual contribution spans exactly ``chi * 2π`` rad across
        [f_low, f_high].  The returned array starts at 0 (cumulative from
        f_low).
        """
        gamma = jnp.array([-5 / 3, -2 / 3, 1.0, 5 / 3, 7 / 3])
        # H(-γ)=1 for γ<0, H(γ)=1 for γ>0 (no γ=0 in our set)
        h_neg = (gamma < 0).astype(gamma.dtype)
        h_pos = (gamma > 0).astype(gamma.dtype)
        d_alpha = (
            chi * 2 * jnp.pi / jnp.abs(f_low**gamma * h_neg - f_high**gamma * h_pos)
        )
        freq_2D = jax.lax.broadcast_in_dim(freqs, (freqs.size, gamma.size), [0])
        d_phi = jnp.sum(jnp.sign(gamma) * d_alpha * freq_2D**gamma, axis=1)
        return d_phi - d_phi[0]

    def _make_binning_scheme(
        self,
        freqs: Float[Array, " n_freq"],
        n_bins: int,
        chi: float = 1.0,
    ) -> tuple[Float[Array, " n_bins + 1"], Float[Array, " n_bins"]]:
        """Make ``n_bins`` frequency bins of equal phase change.

        ``n_bins`` must be a positive integer resolved by the caller
        (see :meth:`__init__`).
        """
        phase_diff_array = self._max_phase_diff(freqs, freqs[0], freqs[-1], chi=chi)
        total_phase = phase_diff_array[-1]
        phase_diff = jnp.linspace(0.0, total_phase, n_bins + 1)
        f_bins = interp1d(phase_diff_array, freqs)(phase_diff)
        f_bins_center = (f_bins[:-1] + f_bins[1:]) / 2
        return jnp.array(f_bins), jnp.array(f_bins_center)

    @staticmethod
    def _compute_coefficients(data, h_ref, psd, freqs, f_bins, f_bins_center):
        df = freqs[1] - freqs[0]
        data_prod = jnp.array(data * h_ref.conj()) / psd
        self_prod = jnp.array(h_ref * h_ref.conj()) / psd

        freq_bins_left = f_bins[:-1]
        freq_bins_right = f_bins[1:]

        freqs_broadcast = freqs[None, :]
        left_bounds = freq_bins_left[:, None]
        right_bounds = freq_bins_right[:, None]

        mask = (freqs_broadcast >= left_bounds) & (freqs_broadcast < right_bounds)
        # The half-open interval [left, right) excludes any frequency that lands
        # exactly on the upper edge of the last bin (f_bins[-1]).  This happens
        # whenever the interpolated bin edge coincides with the last discrete
        # frequency sample (common when the waveform reaches f_max).  Extend the
        # last row to a closed interval by OR-ing in the equality condition.
        mask = mask.at[-1].set(mask[-1] | (freqs == freq_bins_right[-1]))

        f_bins_center_broadcast = f_bins_center[:, None]
        freq_shift_matrix = (freqs_broadcast - f_bins_center_broadcast) * mask

        A0_array = 4 * jnp.sum(data_prod[None, :] * mask, axis=1) * df
        A1_array = 4 * jnp.sum(data_prod[None, :] * freq_shift_matrix, axis=1) * df
        B0_array = 4 * jnp.sum(self_prod[None, :] * mask, axis=1) * df
        B1_array = 4 * jnp.sum(self_prod[None, :] * freq_shift_matrix, axis=1) * df

        return A0_array, A1_array, B0_array, B1_array

    def maximize_likelihood(
        self,
        prior: Prior,
        likelihood_transforms: list[NtoMTransform],
        optimizer_popsize: int = 500,
        optimizer_n_steps: int = 1000,
    ):
        """Find the maximum-likelihood parameters using CMA-ES.

        Uses ``evosax.CMA_ES`` (Covariance Matrix Adaptation Evolution
        Strategy) to search the full parameter space.  The initial mean is
        drawn from the prior and the entire ask/tell loop is compiled with
        ``jax.lax.scan`` for speed.

        Args:
            prior: Prior used to seed the initial CMA-ES mean.
            likelihood_transforms: Transforms mapping sampling parameters to
                likelihood parameters.
            optimizer_popsize: Population size for CMA-ES.
                Defaults to 500.
            optimizer_n_steps: Number of CMA-ES generations.
                Defaults to 1000.
        """
        parameter_names = list(prior.parameter_names)
        n_dim = len(parameter_names)

        # ------------------------------------------------------------------
        # Reconstruct f_min / f_max per detector from already-set bounds
        # ------------------------------------------------------------------
        f_min_dict = {d.name: d.frequency_bounds[0] for d in self.detectors}
        f_max_dict = {d.name: d.frequency_bounds[1] for d in self.detectors}

        # ------------------------------------------------------------------
        # Build the full (un-marginalized) TransientLikelihoodFD objective
        # ------------------------------------------------------------------
        full_likelihood = TransientLikelihoodFD(
            detectors=self.detectors,
            waveform=self.waveform,
            f_min=f_min_dict,
            f_max=f_max_dict,
            trigger_time=self.trigger_time,
        )

        # ------------------------------------------------------------------
        # Normalize the search space using the prior sample statistics so
        # that every dimension has unit variance before CMA-ES sees it.
        # CMA-ES then operates with std_init=1 (default) in a space where
        # each parameter already lives on a comparable scale.
        # ------------------------------------------------------------------
        n_init = max(optimizer_popsize, 1000)
        init_samples = prior.sample(jax.random.key(0), n_init)
        sample_matrix = jnp.column_stack(
            [init_samples[key] for key in parameter_names]
        )  # (n_init, n_dim)
        prior_mean = jnp.mean(sample_matrix, axis=0)
        prior_std = jnp.std(sample_matrix, axis=0)

        def _log_likelihood(z: Float[Array, " n_dim"]) -> Float:
            """Evaluate -logL for a single normalized parameter vector."""
            x = prior_mean + prior_std * z
            named_params = dict(zip(parameter_names, x, strict=True))
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

        # ------------------------------------------------------------------
        # Set up CMA-ES in normalized space: init_mean=0, std_init=1
        # ------------------------------------------------------------------
        es = CMA_ES(population_size=optimizer_popsize, solution=jnp.zeros(n_dim))
        es_params = es.default_params.replace(std_init=1e-3)  # type: ignore[attr-defined]  # evosax stubs
        key = jax.random.key(42)
        state = es.init(key, jnp.zeros(n_dim), es_params)

        logger.info(
            f"Running evosax CMA-ES: "
            f"{n_dim}D, popsize={optimizer_popsize}, n_steps={optimizer_n_steps}"
        )

        def _step(carry, _):
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
            return (state, key), None

        (state, _), _ = jax.lax.scan(
            _step, (state, key), None, length=optimizer_n_steps
        )

        best_fitness = float(state.best_fitness)
        logger.debug(
            f"CMA-ES finished after {optimizer_n_steps} generations, "
            f"best_fitness={best_fitness:.4f}"
        )
        best_z = state.best_solution

        # ------------------------------------------------------------------
        # Convert best solution back to named parameters
        # ------------------------------------------------------------------
        best_x = prior_mean + prior_std * best_z
        named_params = dict(zip(parameter_names, best_x, strict=True))
        for transform in likelihood_transforms:
            named_params = transform.forward(named_params)
        named_params = apply_fixed_parameters(named_params, self.fixed_parameters)
        return named_params


class MultibandedTransientLikelihoodFD(SingleEventLikelihood):
    """Multi-banded likelihood for gravitational wave transient events.

    This implements the multi-banding method described in S. Morisaki, 2021, arXiv:2104.07813.
    The method divides the frequency range into bands with different resolutions,
    using coarser grids at higher frequencies to speed up likelihood evaluation.

    Attributes:
        reference_chirp_mass (Float): Reference chirp mass for determining frequency bands.
        reference_chirp_mass_in_second (Float): Same value converted to seconds.
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
        f_min (Float | dict[str, Float]): Minimum frequency for likelihood
            evaluation, or a dict mapping detector name to per-detector Float.
        f_max (Float | dict[str, Float]): Maximum frequency for likelihood
            evaluation, or a dict mapping detector name to per-detector Float.
        trigger_time (Float): GPS time of the event trigger.
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
            to 2.12 s with a warning when the prior is unavailable.
        delta_f_end (Optional[Float]): End frequency taper scale in Hz.  When
            ``None``, inferred from the ``t_c`` prior range; falls back to 53.0 Hz.
        max_banding_frequency (Optional[Float]): Upper limit on band starting frequency.
        min_banding_duration (Float): Minimum duration for bands.
    """

    highest_mode: int
    accuracy_factor: Float
    reference_chirp_mass: Float
    reference_chirp_mass_in_second: Float
    time_offset: Float
    delta_f_end: Float
    max_banding_frequency: Float
    min_banding_duration: Float

    durations: Float[Array, " n_bands"]
    fb_dfb: Float[Array, "n_bands+1 2"]

    unique_frequencies: Float[Array, " n_unique"]
    unique_to_original: Array

    linear_coeffs: dict[str, Float[Array, " n_total_points"]]
    quadratic_coeffs: dict[str, Float[Array, " n_total_points"]]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[
            dict[str, Float | Callable[[dict[str, Float]], Float | dict[str, Float]]]
        ] = None,
        f_min: Float | dict[str, Float] = 0,
        f_max: Float | dict[str, Float] = jnp.inf,
        trigger_time: Float = 0,
        highest_mode: int = 2,
        accuracy_factor: Float = 5.0,
        prior: Optional[Prior] = None,
        reference_chirp_mass: Optional[Float] = None,
        time_offset: Optional[Float] = None,
        delta_f_end: Optional[Float] = None,
        max_banding_frequency: Optional[Float] = None,
        min_banding_duration: Float = 0.0,
    ):

        super().__init__(detectors, waveform, fixed_parameters)

        logger.info("Initializing multi-banded likelihood...")

        reference_chirp_mass = self._resolve_reference_chirp_mass(
            reference_chirp_mass, prior
        )
        time_offset, delta_f_end = self._resolve_time_params(
            time_offset, delta_f_end, prior, float(trigger_time), detectors
        )
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

        _f_mins = []
        _f_maxs = []
        for detector in detectors:
            f_min_ifo = f_min if not isinstance(f_min, dict) else f_min[detector.name]
            f_max_ifo = f_max if not isinstance(f_max, dict) else f_max[detector.name]
            detector.set_frequency_bounds(f_min_ifo, f_max_ifo)
            sliced = detector.sliced_frequencies
            _f_mins.append(float(sliced[0]))
            _f_maxs.append(float(sliced[-1]))

        self.minimum_frequency = min(_f_mins)
        self.maximum_frequency = max(_f_maxs)

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

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(trigger_time)

        self._setup_frequency_bands()
        self._setup_integers()
        self._setup_waveform_frequency_points()
        self._setup_linear_coefficients()
        self._setup_quadratic_coefficients()

        logger.info("Multi-banding setup complete with %d bands", self.n_bands)

    # ── Prior-inference and validation helpers ────────────────────────────────

    def _find_leaf_prior(self, prior: Prior, param_name: str) -> Optional[Prior]:
        """Recursively search *prior* for the bounded leaf that owns *param_name*.

        Returns the first component that has ``xmin``/``xmax`` attributes and
        lists *param_name* in its ``parameter_names``, or ``None`` if not found.
        """
        if param_name not in prior.parameter_names:
            return None
        if hasattr(prior, "xmin") and hasattr(prior, "xmax"):
            return prior
        if hasattr(prior, "base_prior"):
            components = getattr(prior, "base_prior")
            if hasattr(components, "__iter__"):
                for p in components:
                    result = self._find_leaf_prior(p, param_name)
                    if result is not None:
                        return result
        return None

    def _resolve_reference_chirp_mass(
        self,
        reference_chirp_mass: Optional[Float],
        prior: Optional[Prior],
    ) -> float:
        """Return ``reference_chirp_mass``, inferring from the M_c prior minimum when not provided."""
        if reference_chirp_mass is not None:
            return reference_chirp_mass
        if prior is None:
            raise ValueError(
                "Either reference_chirp_mass or a prior with an M_c component must be provided."
            )
        mc_prior = self._find_leaf_prior(prior, "M_c")
        if mc_prior is None:
            raise ValueError(
                "reference_chirp_mass=None but no M_c prior found. "
                "Pass either reference_chirp_mass or a prior with an M_c component."
            )
        mc_min = float(getattr(mc_prior, "xmin"))
        logger.info(
            "reference_chirp_mass inferred from M_c prior minimum: %.4f M_sun", mc_min
        )
        return mc_min

    def _resolve_time_params(
        self,
        time_offset: Optional[Float],
        delta_f_end: Optional[Float],
        prior: Optional[Prior],
        trigger_time: float,
        detectors: Sequence[Detector],
    ) -> tuple[Float, Float]:
        """Return ``(time_offset, delta_f_end)``, inferring from t_c prior bounds when not provided.

        Inference uses the geocentric coalescence time ``t_c`` only.
        Detector-frame time ``t_det`` is not supported because
        ``t_det = t_c + sky_delay(ra, dec)`` and the delay is sky-position-dependent,
        so ``t_c`` bounds cannot be derived from a ``t_det`` prior at setup time.
        Falls back to defaults (2.12 s, 53.0 Hz) when inference is not possible.
        """
        inferred_to: Optional[Float] = None
        inferred_dfe: Optional[Float] = None

        if prior is not None and (time_offset is None or delta_f_end is None):
            tc_prior = self._find_leaf_prior(prior, "t_c")
            if tc_prior is not None:
                t_end = min(
                    float(d.data.start_time) + float(d.data.duration) - trigger_time
                    for d in detectors
                )
                s = EARTH_RADIUS_LIGHT_S
                tc_max = float(getattr(tc_prior, "xmax"))
                denom = t_end - tc_max - s
                if denom <= 0:
                    raise ValueError(
                        f"Cannot infer delta_f_end from t_c prior: "
                        f"t_end - xmax - s = {t_end:.4f} - {tc_max:.4f} - {s:.6f} = {denom:.6f} <= 0. "
                        "Check that the t_c prior upper bound is well within the data segment."
                    )
                inferred_to = t_end - float(getattr(tc_prior, "xmin")) + s
                inferred_dfe = 100.0 / denom

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
        reference_chirp_mass: Float,
        highest_mode: int,
        accuracy_factor: Float,
        time_offset: Float,
        delta_f_end: Float,
        min_banding_duration: Float,
        max_banding_frequency: Optional[Float],
    ) -> None:
        """Raise ValueError for any out-of-range banding parameter."""
        if reference_chirp_mass <= 0:
            raise ValueError(
                f"reference_chirp_mass must be > 0, got {reference_chirp_mass}"
            )
        if highest_mode <= 0:
            raise ValueError(f"highest_mode must be > 0, got {highest_mode}")
        if accuracy_factor <= 0:
            raise ValueError(f"accuracy_factor must be > 0, got {accuracy_factor}")
        if time_offset < 0:
            raise ValueError(f"time_offset must be >= 0, got {time_offset}")
        if delta_f_end <= 0:
            raise ValueError(f"delta_f_end must be > 0, got {delta_f_end}")
        if min_banding_duration < 0:
            raise ValueError(
                f"min_banding_duration must be >= 0, got {min_banding_duration}"
            )
        if max_banding_frequency is not None and max_banding_frequency <= 0:
            raise ValueError(
                f"max_banding_frequency must be > 0, got {max_banding_frequency}"
            )

    # ── Band structure ────────────────────────────────────────────────────────

    @property
    def n_bands(self) -> int:
        """Number of frequency bands."""
        return len(self.durations)

    def _tau(self, f: Float) -> Float:
        """Compute time-to-merger using 0PN formula.

        Parameters
        ----------
        f : Float
            Input frequency in Hz.

        Returns
        -------
        Float
            Time-to-merger in seconds.
        """
        f_22 = 2 * f / self.highest_mode
        return (
            5
            / 256
            * self.reference_chirp_mass_in_second
            * (jnp.pi * self.reference_chirp_mass_in_second * f_22) ** (-8 / 3)
        )

    def _dtaudf(self, f: Float) -> Float:
        """Compute derivative of time-to-merger using 0PN formula.

        Parameters
        ----------
        f : Float
            Input frequency in Hz.

        Returns
        -------
        Float
            Derivative of time-to-merger (negative, in seconds/Hz).
        """
        f_22 = 2 * f / self.highest_mode
        return (
            -5
            / 96
            * self.reference_chirp_mass_in_second
            * (jnp.pi * self.reference_chirp_mass_in_second * f_22) ** (-8 / 3)
            / f
        )

    def _find_starting_frequency(
        self, duration: Float, f_now: Float
    ) -> tuple[Optional[Float], Optional[Float]]:
        """Find starting frequency of next band via bisection search.

        Finds frequency satisfying conditions (10) and (51) of arXiv:2104.07813:
        - Time containment: tau(f) + L * sqrt(-dtau/df) < duration - time_offset
        - Smooth transition: f - 1/sqrt(-dtau/df) > f_now

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
            cond1 = (
                duration
                - self.time_offset
                - self._tau(f)
                - self.accuracy_factor * jnp.sqrt(-self._dtaudf(f))
            ) > 0
            cond2 = f - 1.0 / jnp.sqrt(-self._dtaudf(f)) - f_now > 0
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

        return f, 1.0 / jnp.sqrt(-self._dtaudf(f))

    def _setup_frequency_bands(self) -> None:
        """Set up frequency bands with geometrically decreasing durations.

        Bands have durations T, T/2, T/4, ... where T is the original data duration.

        Sets:
            self.durations: Array of band durations
            self.fb_dfb: Array of [starting_freq, taper_width] for each band
        """
        original_duration = self.detectors[0].data.duration

        durations_list = [original_duration]
        fb_dfb_list = [[self.minimum_frequency, 0.0]]

        dnext = original_duration / 2

        while dnext > max(self.time_offset, self.min_banding_duration):
            f_now, _ = fb_dfb_list[-1]
            fnext, dfnext = self._find_starting_frequency(dnext, f_now)

            if fnext is not None and fnext < min(
                self.maximum_frequency, self.max_banding_frequency
            ):
                durations_list.append(dnext)
                fb_dfb_list.append([fnext, dfnext])
                dnext /= 2
            else:
                break

        # Add final boundary
        fb_dfb_list.append(
            [self.maximum_frequency + self.delta_f_end, self.delta_f_end]
        )

        self.durations = jnp.array(durations_list)
        self.fb_dfb = jnp.array(fb_dfb_list)

        logger.info(
            f"Frequency range divided into {self.n_bands} bands with "
            f"intervals: {', '.join(['1/' + str(d) + ' Hz' for d in durations_list])}"
        )

    def _round_up_to_power_of_two(self, n: int) -> int:
        """Round up to the nearest power of two."""
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()

    def _setup_integers(self) -> None:
        """Set up integer indices for each band.

        Sets:
            self.Nbs: Number of samples in downsampled data per band
            self.Mbs: Number of samples in shortened data per band
            self.Ks_Ke: Start/end frequency indices per band
        """
        original_duration = self.detectors[0].data.duration
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
                self._round_up_to_power_of_two(
                    int(2.0 * fnext * original_duration + 1)
                ),
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

    def _get_window_sequence(
        self, delta_f: Float, start_idx: int, length: int, band: int
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

    def _setup_linear_coefficients(self) -> None:
        """Pre-compute coefficients for (d|h) inner product.

        For each band:
        1. Apply frequency mask and divide by PSD
        2. IFFT to time domain, take last M^(b) samples
        3. FFT back to get shortened data
        4. Multiply by window and normalization factor

        Sets:
            self.linear_coeffs: Dict mapping detector name to coefficient array
        """
        Ks_Ke = self.Ks_Ke.tolist()
        Nbs = self.Nbs.tolist()
        Mbs = self.Mbs.tolist()
        durations = self.durations.tolist()
        N = Nbs[-1]

        self.linear_coeffs = {}

        for detector in self.detectors:
            logger.info(f"Pre-computing linear coefficients for {detector.name}")
            data_fd = jnp.array(detector.data.fd)
            psd = jnp.array(detector.psd.values)
            freq_mask = jnp.array(detector.frequency_mask)

            valid_len = min(len(data_fd), N // 2 + 1)
            mask_valid = freq_mask[:valid_len]
            safe_psd = jnp.where(mask_valid, psd[:valid_len], 1.0)
            values = jnp.where(mask_valid, data_fd[:valid_len] / safe_psd, 0.0)
            fddata = jnp.zeros(N // 2 + 1, dtype=complex).at[:valid_len].set(values)

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
            self.quadratic_coeffs: Dict mapping detector name to coefficient array
        """

        original_duration = self.detectors[0].data.duration
        start_end_idxs = self.start_end_idxs.tolist()
        durations = self.durations.tolist()
        fb_dfb = self.fb_dfb.tolist()
        banded_frequency_points = self.banded_frequency_points.tolist()

        self.quadratic_coeffs = {}

        for detector in self.detectors:
            psd = jnp.array(detector.psd.values)
            freq_mask = jnp.array(detector.frequency_mask)

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
                end_idx_orig = min(start_idx_orig + len(window) - 1, len(psd) - 1)
                valid_len = end_idx_orig - start_idx_orig + 1

                local_mask = freq_mask[start_idx_orig : end_idx_orig + 1]
                psd_slice = psd[start_idx_orig : end_idx_orig + 1]
                safe_psd = jnp.where(local_mask, psd_slice, 1.0)
                window_over_psd = (
                    jnp.where(local_mask, 1.0 / safe_psd, 0.0) * window[:valid_len]
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

    def evaluate(self, params: dict[str, Float]) -> Float:
        """Evaluate the log-likelihood for given parameters.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary of model parameters.

        Returns
        -------
        Float
            Log-likelihood value.
        """
        params = params.copy()
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        apply_fixed_parameters(params, self.fixed_parameters)
        return self._likelihood(params)

    def _likelihood(self, params: dict[str, Float]) -> Float:
        """Core likelihood evaluation using multi-banding.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary of model parameters.

        Returns
        -------
        Float
            Log-likelihood value.
        """
        # Generate waveform at unique frequencies
        waveform_sky = self.waveform(self.unique_frequencies, params)

        log_likelihood = 0.0

        for detector in self.detectors:
            # Get detector response at banded frequencies
            # First evaluate at unique frequencies, then map to banded
            h_det_unique = detector.fd_response(
                self.unique_frequencies, waveform_sky, params
            )

            # Map from unique to banded frequency points
            strain = h_det_unique[self.unique_to_original]

            # Compute (d|h) using pre-computed linear coefficients
            d_inner_h = jnp.sum(strain * self.linear_coeffs[detector.name])

            # Compute (h|h) using pre-computed quadratic coefficients and linear interpolation
            h_inner_h = jnp.sum(
                jnp.real(strain * jnp.conj(strain))
                * self.quadratic_coeffs[detector.name]
            )

            # Accumulate log-likelihood: Re(d|h) - (h|h)/2
            log_likelihood += jnp.real(d_inner_h) - h_inner_h / 2

        return log_likelihood


likelihood_presets = {
    "TransientLikelihoodFD": TransientLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
    "MultibandedTransientLikelihoodFD": MultibandedTransientLikelihoodFD,
}
