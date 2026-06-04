"""Cross-validation tests comparing jimgw likelihood classes against bilby.

All likelihood classes are tested:

    Jim class                                              bilby equivalent
    ─────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
    TransientLikelihoodFD                                  GravitationalWaveTransient (no marginalization)
    TransientLikelihoodFD(phase_marginalization=True)      GravitationalWaveTransient (phase_marginalization=True)
    TransientLikelihoodFD(time_marginalization={})         GravitationalWaveTransient (time_marginalization=True)
    TransientLikelihoodFD(distance_marginalization={...})  GravitationalWaveTransient (distance_marginalization=True)
    TransientLikelihoodFD(phase+distance marg)             GravitationalWaveTransient (phase+distance marginalization)
    TransientLikelihoodFD(time+phase marg)                 GravitationalWaveTransient (time+phase marginalization)
    HeterodynedTransientLikelihoodFD                       RelativeBinningGravitationalWaveTransient
    HeterodynedTransientLikelihoodFD(phase_marg=True)      RelativeBinningGravitationalWaveTransient (phase_marg=True)
    MultibandedTransientLikelihoodFD                       MBGravitationalWaveTransient

Uses IMRPhenomPv2 with GW150914 data fixtures.

Requires bilby (with bilby_cython for sky transform cross-checks) and LALSuite to be installed.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

from tests.utils import check_bilby_available

try:
    check_bilby_available()
    import bilby

    BILBY_AVAILABLE = True
except ImportError:
    BILBY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BILBY_AVAILABLE,
    reason="bilby required for cross-validation tests",
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# ─── GW150914 parameters ──────────────────────────────────────────────────────
GPS = 1126259462.4
F_MIN = 20.0
F_MAX = 1024.0
F_REF = 20.0

# Single tolerance used by every cross-validation assertion.
CROSS_VAL_RTOL = 1e-3

# Reference parameters in bilby convention (GW150914-like)
BILBY_PARAMS = {
    "mass_1": 35.6,
    "mass_2": 30.6,
    "a_1": 0.32,
    "tilt_1": 0.56,
    "phi_jl": 1.02,
    "a_2": 0.18,
    "tilt_2": 0.82,
    "phi_12": 1.55,
    "theta_jn": 2.91,
    "phase": 1.2,
    "luminosity_distance": 410.0,
    "geocent_time": GPS,  # t_c = 0 relative to trigger
    "ra": 1.375,
    "dec": -1.2108,
    "psi": 2.659,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def bilby_to_jim_params(bilby_params: dict) -> dict:
    """Convert bilby-convention parameters to jim-convention parameters.

    Uses bilby's own spin-angle→Cartesian conversion so that only the
    waveform implementation (ripple vs LAL) can cause differences; the
    parameterization mapping is identical by construction.
    """
    from bilby.gw.conversion import bilby_to_lalsimulation_spins

    m1 = bilby_params["mass_1"]
    m2 = bilby_params["mass_2"]
    total_mass = m1 + m2
    eta = (m1 * m2) / total_mass**2
    # chirp mass
    M_c = (m1 * m2) ** (3 / 5) / total_mass ** (1 / 5)

    iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
        theta_jn=bilby_params["theta_jn"],
        phi_jl=bilby_params["phi_jl"],
        tilt_1=bilby_params["tilt_1"],
        tilt_2=bilby_params["tilt_2"],
        phi_12=bilby_params["phi_12"],
        a_1=bilby_params["a_1"],
        a_2=bilby_params["a_2"],
        mass_1=m1,
        mass_2=m2,
        reference_frequency=F_REF,
        phase=bilby_params["phase"],
    )

    return {
        "M_c": M_c,
        "eta": eta,
        "s1_x": s1x,
        "s1_y": s1y,
        "s1_z": s1z,
        "s2_x": s2x,
        "s2_y": s2y,
        "s2_z": s2z,
        "iota": iota,
        "d_L": bilby_params["luminosity_distance"],
        "phase_c": bilby_params["phase"],
        # t_c is the offset from trigger_time; geocent_time == GPS → t_c = 0
        "t_c": bilby_params["geocent_time"] - GPS,
        "ra": bilby_params["ra"],
        "dec": bilby_params["dec"],
        "psi": bilby_params["psi"],
    }


def load_jim_detectors():
    """Load H1 and L1 with GW150914 fixture data and PSD."""
    from jimgw.core.single_event.detector import get_H1, get_L1
    from jimgw.core.single_event.data import Data, PowerSpectrum

    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        data = Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
        ifo.set_data(data)
        psd = PowerSpectrum.from_file(
            str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
        )
        ifo.set_psd(psd)
    return ifos


def build_bilby_ifo_from_jim(jim_ifo, f_min: float, f_max: float):
    """Create a bilby interferometer backed by the same FD data as *jim_ifo*.

    We extract jim's windowed FD strain (the full one-sided array) and the
    interpolated PSD and hand them directly to bilby.  Because the FD data
    already has the Tukey window applied, bilby sees a window factor of 1,
    so the inner-product normalisations are identical.
    """
    # Trigger the FFT if not already done
    jim_ifo.data.fft()

    fd_strain_full = np.array(jim_ifo.data.fd, dtype=complex)
    frequencies_full = np.array(jim_ifo.data.frequencies, dtype=float)
    duration = float(jim_ifo.data.duration)
    sampling_frequency = float(jim_ifo.data.sampling_frequency)
    start_time = float(jim_ifo.data.start_time)

    # Zero out contributions outside [f_min, f_max].  bilby's time-marginalization
    # FFT runs over the *full* one-sided frequency array without a frequency mask,
    # so keeping out-of-band data would add spurious contributions absent in jim
    # (which pads those bins with zeros).  Zeroing here makes both FFTs identical.
    out_of_band = (frequencies_full < f_min) | (frequencies_full > f_max)
    fd_strain_full[out_of_band] = 0.0

    # Interpolate jim PSD onto the full frequency grid and extract the values array
    psd_full = np.array(
        jim_ifo.psd.interpolate(jim_ifo.data.frequencies).values, dtype=float
    )

    bilby_ifo = bilby.gw.detector.get_empty_interferometer(jim_ifo.name)
    bilby_ifo.set_strain_data_from_frequency_domain_strain(
        frequency_domain_strain=fd_strain_full,
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    bilby_ifo.power_spectral_density = (
        bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=frequencies_full,
            psd_array=psd_full,
        )
    )
    bilby_ifo.minimum_frequency = f_min
    bilby_ifo.maximum_frequency = f_max
    return bilby_ifo


def ripple_pv2_bilby_source(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    tilt_1,
    phi_12,
    a_2,
    tilt_2,
    phi_jl,
    theta_jn,
    phase,
    **kwargs,
):
    """bilby-compatible frequency-domain source model that calls RippleIMRPhenomPv2.

    Accepts bilby-convention (mass_1/2, spin angles) and converts to Cartesian
    spins via bilby's own ``bilby_to_lalsimulation_spins``, so the parameter
    mapping is identical by construction.  The waveform is then evaluated with
    ripple, giving the same result as jim's ``RippleIMRPhenomPv2`` waveform class.

    Special handling:
    - ``fiducial`` kwarg (bilby relative-binning): 1 → evaluate on the full
      frequency grid (``frequency_array``); 0 (default) → evaluate only at the
      bin edges supplied via ``frequency_bin_edges``.
    - ``frequency_bin_edges`` kwarg: used by bilby's relative-binning likelihood
      when ``fiducial=0``.  It stores the bin-edge frequencies in
      ``waveform_arguments['frequency_bin_edges']`` and leaves the key *as-is*
      (bilby's own LAL source functions rename it to ``frequencies`` internally,
      but our custom source must read it directly as ``frequency_bin_edges``).
    - Non-finite values (NaN/Inf) at DC and very low frequencies are replaced
      with 0.  This is required for time-marginalized likelihoods that FFT the
      full frequency array (including f=0) without applying a frequency mask.
    """
    import jax.numpy as jnp
    import numpy as np
    from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
    from bilby.gw.conversion import bilby_to_lalsimulation_spins as b2lal

    # bilby's relative-binning likelihood sets waveform_arguments['fiducial']:
    #   1 → computing the fiducial waveform  → return full frequency grid array
    #   0 → likelihood evaluation            → return bin-edge array
    # bilby's MB likelihood sets waveform_arguments['frequencies'] to the unique
    # MB frequencies so the waveform is only evaluated at those points.
    # Check 'frequencies' (MB key) before 'fiducial' (relative-binning key) so that
    # MB evaluations use the correct frequency array.
    fiducial = kwargs.get("fiducial", 1)
    mb_frequencies = kwargs.get("frequencies", None)
    if mb_frequencies is not None:
        eval_freqs = mb_frequencies
    elif fiducial == 1:
        # Full-grid evaluation (fiducial setup, or standard likelihoods)
        eval_freqs = frequency_array
    else:
        # Bin-edge evaluation for relative-binning likelihood
        eval_freqs = kwargs.get(
            "frequency_bin_edges", kwargs.get("frequencies", frequency_array)
        )

    total_mass = mass_1 + mass_2
    eta = mass_1 * mass_2 / total_mass**2
    M_c = (mass_1 * mass_2) ** 0.6 / total_mass**0.2

    iota, s1x, s1y, s1z, s2x, s2y, s2z = b2lal(
        theta_jn=theta_jn,
        phi_jl=phi_jl,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        phi_12=phi_12,
        a_1=a_1,
        a_2=a_2,
        mass_1=mass_1,
        mass_2=mass_2,
        reference_frequency=F_REF,
        phase=phase,
    )

    theta = jnp.array(
        [M_c, eta, s1x, s1y, s1z, s2x, s2y, s2z, luminosity_distance, 0.0, phase, iota]
    )
    f_jax = jnp.array(eval_freqs, dtype=jnp.float64)
    hp, hc = gen_IMRPhenomPv2_hphc(f_jax, theta, F_REF)
    hp_arr = np.array(hp, dtype=complex)
    hc_arr = np.array(hc, dtype=complex)
    # Replace NaN/Inf (e.g. at f=0 Hz) with 0; GW waveforms vanish at DC.
    hp_arr[~np.isfinite(hp_arr)] = 0.0
    hc_arr[~np.isfinite(hc_arr)] = 0.0
    return {"plus": hp_arr, "cross": hc_arr}


def build_bilby_waveform_generator(duration: float, sampling_frequency: float):
    """Build a bilby WaveformGenerator backed by RippleIMRPhenomPv2.

    Using the same ripple waveform in both jim and bilby ensures that any
    remaining difference in the evaluated log-likelihoods comes purely from
    the likelihood infrastructure (detector response, inner product, frequency
    masking), not from waveform model disagreement.
    """
    return bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=ripple_pv2_bilby_source,
        # Must be None: ripple_pv2_bilby_source handles the spin conversion
        # internally.  bilby's default convert_to_lal_binary_black_hole_parameters
        # mangles the parameter dict in a way that makes the source function
        # return NaN (e.g. when called without a reference_frequency key).
        parameter_conversion=None,
        waveform_arguments={},
    )


# ─── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def setup():
    """Module-level fixture that loads data, creates detectors and waveforms once."""
    from jimgw.core.single_event.waveform import RippleIMRPhenomPv2

    jim_ifos = load_jim_detectors()
    # Set frequency bounds required by TransientLikelihoodFD
    for ifo in jim_ifos:
        ifo.set_frequency_bounds(F_MIN, F_MAX)

    bilby_ifos = [build_bilby_ifo_from_jim(ifo, F_MIN, F_MAX) for ifo in jim_ifos]

    waveform = RippleIMRPhenomPv2(f_ref=F_REF)
    duration = float(jim_ifos[0].data.duration)
    sampling_frequency = float(jim_ifos[0].data.sampling_frequency)

    jim_params = bilby_to_jim_params(BILBY_PARAMS)
    bilby_params = BILBY_PARAMS.copy()

    return {
        "jim_ifos": jim_ifos,
        "bilby_ifos": bilby_ifos,
        "waveform": waveform,
        "jim_params": jim_params,
        "bilby_params": bilby_params,
        "duration": duration,
        "sampling_frequency": sampling_frequency,
    }


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestTransientLikelihoodFD:
    """TransientLikelihoodFD cross-validation against GravitationalWaveTransient."""

    def test_base(self, setup):
        """No marginalization."""
        from jimgw.core.single_event.likelihood import TransientLikelihoodFD

        jim_ll = TransientLikelihoodFD(
            detectors=setup["jim_ifos"],
            waveform=setup["waveform"],
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
        ).evaluate(setup["jim_params"].copy())

        bilby_ll = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=setup["bilby_ifos"],
            waveform_generator=build_bilby_waveform_generator(
                setup["duration"], setup["sampling_frequency"]
            ),
        ).log_likelihood_ratio(setup["bilby_params"].copy())

        print(f"\n[BaseTransient] jim={float(jim_ll):.4f}  bilby={float(bilby_ll):.4f}")
        assert jnp.isclose(jim_ll, bilby_ll, rtol=CROSS_VAL_RTOL), (
            f"TransientLikelihoodFD mismatch: jim={float(jim_ll):.6f}, "
            f"bilby={float(bilby_ll):.6f}"
        )

    def test_phase_marginalized(self, setup):
        """Phase marginalization.

        TransientLikelihoodFD.evaluate() always overrides phase_c=0 when
        phase_marginalization is active before projecting the waveform.  The Cartesian spin
        components in jim_params are derived from bilby_to_lalsimulation_spins(phase=...).
        If those spins were computed for a non-zero phase, the resulting (spins, phase_c=0)
        pair is inconsistent and gives a different waveform than bilby evaluates.

        Fix: compute jim_params with phase=0 so that the spins and the phase that
        evaluate() uses (phase_c=0) are consistent.  bilby ignores the input 'phase'
        when phase_marginalization=True (it evaluates at phase=0 internally), so
        both sides produce the same waveform.
        """
        from jimgw.core.single_event.likelihood import TransientLikelihoodFD

        # Use phase=0 for jim: spins computed at phase=0 are consistent with
        # the phase_c=0 that TransientLikelihoodFD.evaluate() enforces.
        bilby_params_ph0 = {**setup["bilby_params"], "phase": 0.0}
        jim_params_ph0 = bilby_to_jim_params(bilby_params_ph0)

        jim_ll = TransientLikelihoodFD(
            detectors=setup["jim_ifos"],
            waveform=setup["waveform"],
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
            phase_marginalization=True,
        ).evaluate(jim_params_ph0)

        priors = bilby.core.prior.PriorDict()
        priors["phase"] = bilby.core.prior.Uniform(
            minimum=0.0, maximum=2 * np.pi, boundary="periodic", name="phase"
        )

        bilby_ll = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=setup["bilby_ifos"],
            waveform_generator=build_bilby_waveform_generator(
                setup["duration"], setup["sampling_frequency"]
            ),
            phase_marginalization=True,
            priors=priors,
        ).log_likelihood_ratio(bilby_params_ph0.copy())

        print(f"\n[PhaseMarg] jim={float(jim_ll):.4f}  bilby={float(bilby_ll):.4f}")
        assert jnp.isclose(jim_ll, bilby_ll, rtol=CROSS_VAL_RTOL), (
            f"TransientLikelihoodFD(marginalize_phase) mismatch: jim={float(jim_ll):.6f}, "
            f"bilby={float(bilby_ll):.6f}"
        )

    def test_time_marginalized(self, setup):
        """Time marginalization.

        To get exact normalisation agreement we make both sides integrate over the
        **full segment** [start_time, start_time+duration]:

        * Jim uses ``tc_range = (-duration/2 - margin, duration/2 + margin)`` so
          that every FFT bin is in range and the denominator is ``-log(N_fft)``.
        * bilby gets a ``Uniform(start_time, start_time+duration)`` prior, which gives the
          same weight ``log(dt / duration) = -log(N_fft)`` per bin.

        Out-of-band FD strain is zeroed in ``build_bilby_ifo_from_jim`` so that
        bilby's unmasked FFT sees the same in-band-only data as jim.
        """
        from jimgw.core.single_event.likelihood import TransientLikelihoodFD

        duration = float(setup["jim_ifos"][0].data.duration)
        start_time = float(setup["jim_ifos"][0].data.start_time)
        # Cover the full FFT window so jim's denominator is -log(N_fft), matching
        # bilby's log(dt/duration) = -log(N_fft) normalisation.
        tc_range = (-duration / 2 - 0.1, duration / 2 + 0.1)

        jim_ll = TransientLikelihoodFD(
            detectors=setup["jim_ifos"],
            waveform=setup["waveform"],
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
            time_marginalization={"tc_range": tc_range},
        ).evaluate(setup["jim_params"].copy())

        priors = bilby.core.prior.PriorDict()
        priors["geocent_time"] = bilby.core.prior.Uniform(
            minimum=start_time,
            maximum=start_time + duration,
            name="geocent_time",
        )

        bilby_ll = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=setup["bilby_ifos"],
            waveform_generator=build_bilby_waveform_generator(
                setup["duration"], setup["sampling_frequency"]
            ),
            time_marginalization=True,
            jitter_time=False,
            priors=priors,
        ).log_likelihood_ratio(setup["bilby_params"].copy())

        print(f"\n[TimeMarg] jim={float(jim_ll):.4f}  bilby={float(bilby_ll):.4f}")
        assert jnp.isclose(jim_ll, bilby_ll, rtol=CROSS_VAL_RTOL), (
            f"TransientLikelihoodFD(marginalize_time) mismatch: jim={float(jim_ll):.6f}, "
            f"bilby={float(bilby_ll):.6f}"
        )

    @pytest.mark.slow
    def test_distance_marginalized(self, setup):
        """Distance marginalization.

        Jim uses direct logsumexp over a fine distance grid;
        bilby uses a 2-D spline look-up table.
        """
        from jimgw.core.single_event.likelihood import TransientLikelihoodFD
        from jimgw.core.prior import PowerLawPrior

        dist_min, dist_max = 100.0, 2000.0

        jim_distance_prior = PowerLawPrior(
            xmin=dist_min,
            xmax=dist_max,
            alpha=2.0,
            parameter_names=["d_L"],
        )

        jim_ll = TransientLikelihoodFD(
            detectors=setup["jim_ifos"],
            waveform=setup["waveform"],
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
            distance_marginalization={
                "distance_prior": jim_distance_prior,
                "n_dist_points": 10000,
            },
        ).evaluate(setup["jim_params"].copy())

        bilby_priors = bilby.core.prior.PriorDict()
        bilby_priors["luminosity_distance"] = bilby.core.prior.PowerLaw(
            alpha=2,
            minimum=dist_min,
            maximum=dist_max,
            name="luminosity_distance",
            unit="Mpc",
        )

        # Store the lookup table alongside the test rather than in the repo root.
        _lookup_table = str(
            Path(__file__).parent / ".distance_marginalization_lookup.npz"
        )
        bilby_ll = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=setup["bilby_ifos"],
            waveform_generator=build_bilby_waveform_generator(
                setup["duration"], setup["sampling_frequency"]
            ),
            distance_marginalization=True,
            distance_marginalization_lookup_table=_lookup_table,
            priors=bilby_priors,
        ).log_likelihood_ratio(setup["bilby_params"].copy())

        print(f"\n[DistMarg] jim={float(jim_ll):.4f}  bilby={float(bilby_ll):.4f}")
        assert jnp.isclose(jim_ll, bilby_ll, rtol=CROSS_VAL_RTOL), (
            f"TransientLikelihoodFD(marginalize_distance) mismatch: jim={float(jim_ll):.6f}, "
            f"bilby={float(bilby_ll):.6f}"
        )

    @pytest.mark.slow
    def test_phase_distance_marginalized(self, setup):
        """Phase+distance marginalization.

        Uses phase=0 for parameter conversion so that jim's forced ``phase_c=0`` remains
        consistent with the cartesian spin components.
        """
        from jimgw.core.single_event.likelihood import TransientLikelihoodFD
        from jimgw.core.prior import PowerLawPrior

        dist_min, dist_max = 100.0, 2000.0

        bilby_params_ph0 = {**setup["bilby_params"], "phase": 0.0}
        jim_params_ph0 = bilby_to_jim_params(bilby_params_ph0)

        jim_distance_prior = PowerLawPrior(
            xmin=dist_min,
            xmax=dist_max,
            alpha=2.0,
            parameter_names=["d_L"],
        )

        jim_ll = TransientLikelihoodFD(
            detectors=setup["jim_ifos"],
            waveform=setup["waveform"],
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
            phase_marginalization=True,
            distance_marginalization={
                "distance_prior": jim_distance_prior,
                "n_dist_points": 10000,
            },
        ).evaluate(jim_params_ph0.copy())

        bilby_priors = bilby.core.prior.PriorDict()
        bilby_priors["luminosity_distance"] = bilby.core.prior.PowerLaw(
            alpha=2,
            minimum=dist_min,
            maximum=dist_max,
            name="luminosity_distance",
            unit="Mpc",
        )
        bilby_priors["phase"] = bilby.core.prior.Uniform(
            minimum=0.0,
            maximum=2 * np.pi,
            boundary="periodic",
            name="phase",
        )

        _lookup_table = str(
            Path(__file__).parent / ".phase_distance_marginalization_lookup.npz"
        )
        bilby_ll = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=setup["bilby_ifos"],
            waveform_generator=build_bilby_waveform_generator(
                setup["duration"], setup["sampling_frequency"]
            ),
            phase_marginalization=True,
            distance_marginalization=True,
            distance_marginalization_lookup_table=_lookup_table,
            priors=bilby_priors,
        ).log_likelihood_ratio(bilby_params_ph0.copy())

        print(f"\n[PhaseDistMarg] jim={float(jim_ll):.4f}  bilby={float(bilby_ll):.4f}")
        assert jnp.isclose(jim_ll, bilby_ll, rtol=CROSS_VAL_RTOL), (
            f"TransientLikelihoodFD(marginalize_phase+distance) mismatch: jim={float(jim_ll):.6f}, "
            f"bilby={float(bilby_ll):.6f}"
        )

    def test_phase_time_marginalized(self, setup):
        """Phase+time marginalization.

        Same phase-consistency caveat as test_phase_marginalized: jim_params
        must be derived with phase=0 so that the cartesian spins and the phase_c=0
        that TransientLikelihoodFD.evaluate() enforces are consistent.

        Same time-normalisation fix as test_time_marginalized: use the full
        segment as the time prior so jim and bilby share the same -log(N_fft) norm.
        """
        from jimgw.core.single_event.likelihood import TransientLikelihoodFD

        bilby_params_ph0 = {**setup["bilby_params"], "phase": 0.0}
        jim_params_ph0 = bilby_to_jim_params(bilby_params_ph0)

        duration = float(setup["jim_ifos"][0].data.duration)
        start_time = float(setup["jim_ifos"][0].data.start_time)
        tc_range = (-duration / 2 - 0.1, duration / 2 + 0.1)

        jim_ll = TransientLikelihoodFD(
            detectors=setup["jim_ifos"],
            waveform=setup["waveform"],
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
            time_marginalization={"tc_range": tc_range},
            phase_marginalization=True,
        ).evaluate(jim_params_ph0)

        priors = bilby.core.prior.PriorDict()
        priors["geocent_time"] = bilby.core.prior.Uniform(
            minimum=start_time,
            maximum=start_time + duration,
            name="geocent_time",
        )
        priors["phase"] = bilby.core.prior.Uniform(
            minimum=0.0, maximum=2 * np.pi, boundary="periodic", name="phase"
        )

        bilby_ll = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=setup["bilby_ifos"],
            waveform_generator=build_bilby_waveform_generator(
                setup["duration"], setup["sampling_frequency"]
            ),
            time_marginalization=True,
            phase_marginalization=True,
            jitter_time=False,
            priors=priors,
        ).log_likelihood_ratio(bilby_params_ph0.copy())

        print(f"\n[PhaseTimeMarg] jim={float(jim_ll):.4f}  bilby={float(bilby_ll):.4f}")
        assert jnp.isclose(jim_ll, bilby_ll, rtol=CROSS_VAL_RTOL), (
            f"TransientLikelihoodFD(marginalize_time+phase) mismatch: jim={float(jim_ll):.6f}, "
            f"bilby={float(bilby_ll):.6f}"
        )


class TestHeterodynedTransientLikelihoodFD:
    """HeterodynedTransientLikelihoodFD cross-validation against RelativeBinningGravitationalWaveTransient.

    Both are evaluated using the *same* fiducial/reference parameters so that the
    summary data arrays are computed at the same point.  The log-likelihood is
    then compared at a slightly perturbed parameter point (to avoid the trivial
    zero at the reference).

    Tolerance: CROSS_VAL_RTOL; the binning schemes differ between jim
    (phase-based) and bilby (chi/epsilon-based).
    """

    def test_base(self, setup):
        """No marginalization."""
        from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD

        # Use the shared test point as fiducial/reference
        jim_ref_params = setup["jim_params"].copy()
        bilby_ref_params = setup["bilby_params"].copy()

        # Initialize bilby first with its default epsilon so we can read back
        # the number of bins it chose, then initialize Jim with the same count
        # for an apples-to-apples comparison.
        bilby_likelihood = (
            bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
                interferometers=setup["bilby_ifos"],
                waveform_generator=build_bilby_waveform_generator(
                    setup["duration"], setup["sampling_frequency"]
                ),
                fiducial_parameters=bilby_ref_params,
            )
        )
        n_bins = bilby_likelihood.number_of_bins

        jim_likelihood = HeterodynedTransientLikelihoodFD(
            detectors=setup["jim_ifos"],
            waveform=setup["waveform"],
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
            n_bins=n_bins,
            reference_parameters=jim_ref_params,
        )

        # Slightly perturb mass to move away from the fiducial point
        bilby_eval_params = setup["bilby_params"].copy()
        bilby_eval_params["mass_1"] = setup["bilby_params"]["mass_1"] * 1.01
        bilby_eval_params["mass_2"] = setup["bilby_params"]["mass_2"] * 1.01

        jim_eval_params = bilby_to_jim_params(bilby_eval_params)

        jim_ll = jim_likelihood.evaluate(jim_eval_params)
        bilby_ll = bilby_likelihood.log_likelihood_ratio(bilby_eval_params)

        print(f"\n[Heterodyned] jim={float(jim_ll):.4f}  bilby={float(bilby_ll):.4f}")
        assert jnp.isclose(jim_ll, bilby_ll, rtol=CROSS_VAL_RTOL), (
            f"HeterodynedTransientLikelihoodFD mismatch: jim={float(jim_ll):.6f}, "
            f"bilby={float(bilby_ll):.6f}"
        )

    def test_phase_marginalized(self, setup):
        """Phase marginalization.

        Use phase=0 for jim reference: HeterodynedTransientLikelihoodFD
        internally sets phase_c=0 when building the summary data, so the
        reference waveform must also be computed at phase_c=0 for consistency.
        """
        from jimgw.core.single_event.likelihood import (
            HeterodynedTransientLikelihoodFD,
        )

        bilby_ref_params = setup["bilby_params"].copy()
        bilby_ref_params_ph0 = {**bilby_ref_params, "phase": 0.0}
        jim_ref_params = bilby_to_jim_params(bilby_ref_params_ph0)

        priors = bilby.core.prior.PriorDict()
        priors["phase"] = bilby.core.prior.Uniform(
            minimum=0.0, maximum=2 * np.pi, boundary="periodic", name="phase"
        )

        # Initialize bilby first with its default epsilon so we can read back
        # the number of bins it chose, then initialize Jim with the same count
        # for an apples-to-apples comparison.
        bilby_likelihood = (
            bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
                interferometers=setup["bilby_ifos"],
                waveform_generator=build_bilby_waveform_generator(
                    setup["duration"], setup["sampling_frequency"]
                ),
                fiducial_parameters=bilby_ref_params,
                phase_marginalization=True,
                priors=priors,
            )
        )
        n_bins = bilby_likelihood.number_of_bins

        jim_likelihood = HeterodynedTransientLikelihoodFD(
            detectors=setup["jim_ifos"],
            waveform=setup["waveform"],
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
            n_bins=n_bins,
            reference_parameters=jim_ref_params,
            phase_marginalization=True,
        )

        # Slightly perturb to move away from reference point; keep phase=0 so
        # that jim's internal phase_c=0 override remains consistent with the spins.
        bilby_eval_params = {
            **bilby_ref_params_ph0,
            "mass_1": setup["bilby_params"]["mass_1"] * 1.01,
            "mass_2": setup["bilby_params"]["mass_2"] * 1.01,
        }

        jim_eval_params = bilby_to_jim_params(bilby_eval_params)

        jim_ll = jim_likelihood.evaluate(jim_eval_params)
        bilby_ll = bilby_likelihood.log_likelihood_ratio(bilby_eval_params)

        print(
            f"\n[HeterodynedPhaseMarg] jim={float(jim_ll):.4f}  bilby={float(bilby_ll):.4f}"
        )
        assert jnp.isclose(jim_ll, bilby_ll, rtol=CROSS_VAL_RTOL), (
            f"HeterodynedTransientLikelihoodFD(marginalize_phase) mismatch: jim={float(jim_ll):.6f}, "
            f"bilby={float(bilby_ll):.6f}"
        )


class TestMultibandedTransientLikelihoodFD:
    """MultibandedTransientLikelihoodFD cross-validation against MBGravitationalWaveTransient.

    Both use reference_chirp_mass=20.0 M_sun, accuracy_factor=5, time_offset=2.12,
    delta_f_end=53.0.  Band structure and all precomputed coefficients are identical
    by construction.

    Note: ripple_pv2_bilby_source must use kwargs['frequencies'] (the unique MB
    frequency points set by MBGravitationalWaveTransient._setup_waveform_frequency_points)
    rather than the full frequency_array, so that the waveform is evaluated at the
    correct MB frequencies before being remapped to banded points.
    """

    def test_base(self, setup):
        from jimgw.core.single_event.likelihood import MultibandedTransientLikelihoodFD

        REFERENCE_CHIRP_MASS = 20.0  # M_sun

        jim_ll = MultibandedTransientLikelihoodFD(
            detectors=setup["jim_ifos"],
            waveform=setup["waveform"],
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
            accuracy_factor=5.0,
            reference_chirp_mass=REFERENCE_CHIRP_MASS,
            time_offset=2.12,
            delta_f_end=53.0,
        ).evaluate(setup["jim_params"].copy())

        bilby_ll = bilby.gw.likelihood.MBGravitationalWaveTransient(
            interferometers=setup["bilby_ifos"],
            waveform_generator=build_bilby_waveform_generator(
                setup["duration"], setup["sampling_frequency"]
            ),
            reference_chirp_mass=REFERENCE_CHIRP_MASS,
            accuracy_factor=5.0,
            time_offset=2.12,
            delta_f_end=53.0,
        ).log_likelihood_ratio(setup["bilby_params"].copy())

        print(f"\n[Multibanded] jim={float(jim_ll):.4f}  bilby={float(bilby_ll):.4f}")
        assert jnp.isclose(jim_ll, bilby_ll, rtol=CROSS_VAL_RTOL), (
            f"MultibandedTransientLikelihoodFD mismatch: jim={float(jim_ll):.6f}, "
            f"bilby={float(bilby_ll):.6f}"
        )
