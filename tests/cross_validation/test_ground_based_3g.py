"""Cross-validation of the Earth-rotating detector response.

    Jim                                     Reference
    ──────────────────────────────────────  ────────────────────────────────────────────────
    time_to_merger                          LALSimulation TaylorF2ReducedSpin chirp time (2PN)
    GroundBased3G (rotation)                bilby antenna response and geocentre delay at the
                                            exact GMST of each emission time
    GroundBased3G (rotation + finite size)  bilby_xG frequency_dependent_antenna_response

``bilby_xG`` is installed with the ``cross-validation`` dependency group; it is
not a runtime dependency of jimgw, and the comparison skips without it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

bilby = pytest.importorskip("bilby")

from jimgw.core.constants import MSUN
from jimgw.core.single_event.detector import (
    GroundBased3G,
    get_ET,
    time_to_merger,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)

GPS = 1400000000.0
ARM_LENGTH = 1e4
M1, M2 = 1.45, 1.30
CHI1, CHI2 = 0.03, -0.02
M_C = (M1 * M2) ** (3 / 5) / (M1 + M2) ** (1 / 5)
ETA = M1 * M2 / (M1 + M2) ** 2
RA, DEC, PSI = 1.1, -0.4, 0.7
START_TIME = GPS - 100.0

# Frequencies from 2 Hz, where this BNS is about 21 hours from merger.
FREQUENCIES = np.geomspace(2.0, 2048.0, 200)


def _jim_params(t_c=0.0):
    return {
        "M_c": M_C,
        "eta": ETA,
        "s1_z": CHI1,
        "s2_z": CHI2,
        "ra": RA,
        "dec": DEC,
        "psi": PSI,
        "t_c": t_c,
        "trigger_time": GPS,
        "gmst": compute_gmst(GPS),
    }


def _jim_response(det, params):
    """Plus and cross responses, delay included, from unit polarizations."""
    from jimgw.core.single_event.data import Data

    det.set_data(Data(td=jnp.zeros(8), delta_t=1.0, start_time=START_TIME))
    frequency = jnp.asarray(FREQUENCIES)
    one, zero = jnp.ones_like(frequency) + 0j, jnp.zeros_like(frequency) + 0j
    plus = det.fd_response(frequency, {"p": one, "c": zero}, params)
    cross = det.fd_response(frequency, {"p": zero, "c": one}, params)
    return np.asarray(plus), np.asarray(cross)


def _bilby_interferometer(det, cls):
    return cls(
        name=det.name,
        power_spectral_density=bilby.gw.detector.PowerSpectralDensity.from_aligo(),
        minimum_frequency=2.0,
        maximum_frequency=2048.0,
        length=ARM_LENGTH / 1e3,
        latitude=np.degrees(det.latitude),
        longitude=np.degrees(det.longitude),
        elevation=det.elevation,
        xarm_azimuth=np.degrees(det.xarm_azimuth),
        yarm_azimuth=np.degrees(det.yarm_azimuth),
        xarm_tilt=det.xarm_tilt,
        yarm_tilt=det.yarm_tilt,
    )


@pytest.mark.parametrize("m1, m2", [(1.45, 1.30), (10.0, 1.4), (36.0, 29.0)])
def test_time_to_merger_matches_lal_at_2pn(m1, m2):
    """Non-spinning 2PN chirp time agrees with LALSimulation's."""
    lalsim = pytest.importorskip("lalsimulation")
    M_c = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    eta = m1 * m2 / (m1 + m2) ** 2
    for f in [2.0, 5.0, 20.0, 60.0]:
        expected = lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(
            f, m1 * MSUN, m2 * MSUN, 0.0, 4
        )
        actual = float(time_to_merger(jnp.array(f), M_c, eta))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("chi1, chi2", [(0.5, 0.4), (0.5, -0.4)])
def test_spinning_time_to_merger_matches_bilby_xg(chi1, chi2):
    """Aligned-spin 2PN chirp time agrees with bilby_xG's, spin-spin term included."""
    xg_utils = pytest.importorskip("bilby_xG.utils")
    expected = xg_utils.calculate_time_to_merger_for_any_mode(
        FREQUENCIES, M1, M2, chi1, chi2, mode=2
    )
    actual = np.asarray(time_to_merger(jnp.asarray(FREQUENCIES), M_C, ETA, chi1, chi2))
    # Both evaluate the same closed form and agree to ~1e-15; the spin-spin
    # term is at least ~4e-6 of tau over this band, so 1e-12 resolves it.
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_rotating_response_matches_bilby_at_emission_times():
    """Each frequency sees bilby's response at the exact GMST of its emission time."""
    det = GroundBased3G.from_detector(get_ET()[0])
    t_c = 0.3
    jim_plus, jim_cross = _jim_response(det, _jim_params(t_c))

    ifo = _bilby_interferometer(det, bilby.gw.detector.Interferometer)
    tau = np.asarray(time_to_merger(jnp.asarray(FREQUENCIES), M_C, ETA, CHI1, CHI2))
    emission = GPS + t_c - tau
    bilby_plus = np.empty(len(FREQUENCIES), dtype=complex)
    bilby_cross = np.empty(len(FREQUENCIES), dtype=complex)
    for i, (f, t) in enumerate(zip(FREQUENCIES, emission)):
        delay = ifo.time_delay_from_geocenter(RA, DEC, t)
        # The emission time sets the orientation; the merger time sets the
        # phase. GPS - START_TIME first: GPS + t_c alone rounds by ~5e-8 s.
        shift = np.exp(-2j * np.pi * f * ((GPS - START_TIME + t_c) + delay))
        bilby_plus[i] = ifo.antenna_response(RA, DEC, t, PSI, "plus") * shift
        bilby_cross[i] = ifo.antenna_response(RA, DEC, t, PSI, "cross") * shift

    # Linear against exact GMST differs by ~2e-9 rad over 21 hours.
    np.testing.assert_allclose(jim_plus, bilby_plus, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(jim_cross, bilby_cross, rtol=0.0, atol=1e-7)
    assert np.max(np.abs(bilby_plus)) > 0.1


@pytest.mark.parametrize("finite_size", [False, True])
def test_response_matches_bilby_xg(finite_size):
    """Full per-frequency response agrees with bilby_xG for a long BNS in one detector."""
    bilby_xg = pytest.importorskip("bilby_xG.interferometer")
    xg_utils = pytest.importorskip("bilby_xG.utils")

    det = GroundBased3G.from_detector(
        get_ET()[0], arm_length=ARM_LENGTH, finite_size=finite_size
    )
    jim_plus, jim_cross = _jim_response(det, _jim_params())

    ifo = _bilby_interferometer(det, bilby_xg.Interferometer)
    tau = xg_utils.calculate_time_to_merger_for_any_mode(
        FREQUENCIES, M1, M2, CHI1, CHI2, mode=2, safety=1
    )
    xg_plus, xg_cross = ifo.frequency_dependent_antenna_response(
        RA,
        DEC,
        GPS,
        PSI,
        frequencies=FREQUENCIES,
        start_time=START_TIME,
        times_to_coalescence=tau,
        finite_size=finite_size,
    )

    # bilby_xG takes the GMST rate from a one-day difference of the exact GMST
    # and computes its own tau; 1e-6 covers both at unit antenna amplitude.
    np.testing.assert_allclose(jim_plus, xg_plus, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(jim_cross, xg_cross, rtol=0.0, atol=1e-6)
