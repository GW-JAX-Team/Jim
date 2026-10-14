"""Check the rough duration guard against LAL's independent timing bounds."""

import jax.numpy as jnp
import pytest

from jimgw.core.constants import EARTH_RADIUS_LIGHT_S
from jimgw.core.single_event.detector import get_H1
from jimgw.core.single_event.waveform import RippleIMRPhenomD, RippleIMRPhenomXHM

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")


@pytest.mark.parametrize(
    ("chirp_mass", "eta", "f_min"),
    [
        (1.218, 0.25, 20.0),
        (7.0, 0.24, 20.0),
        (28.0, 0.24, 20.0),
        (28.0, 0.01, 20.0),
        (80.0, 0.25, 20.0),
        (28.0, 0.24, 10.0),
    ],
)
@pytest.mark.parametrize("higher_modes", [False, True])
@pytest.mark.parametrize("long_enough", [False, True])
@pytest.mark.parametrize("spins", [(1.0, 1.0), (0.0, 0.0), (0.3, -0.6)])
def test_conservative_duration_against_lal(
    chirp_mass, eta, f_min, higher_modes, long_enough, spins
):
    total_mass = chirp_mass / eta**0.6
    asymmetry = float(jnp.sqrt(1 - 4 * eta))
    m1 = total_mass * (1 + asymmetry) / 2 * lal.MSUN_SI
    m2 = total_mass * (1 - asymmetry) / 2 * lal.MSUN_SI
    f_start = f_min / 2 if higher_modes else f_min
    chirp = lalsim.SimInspiralChirpTimeBound(f_start, m1, m2, *spins)
    merger = lalsim.SimInspiralMergeTimeBound(m1, m2)
    ringdown = lalsim.SimInspiralRingdownTimeBound(m1 + m2, 0.998)
    padding = 3 / f_start + EARTH_RADIUS_LIGHT_S
    before = 1.1 * chirp + merger + padding if long_enough else 0.9 * chirp
    after = 1.2 * (merger + ringdown) + padding

    detector = get_H1()
    detector.set_zero_noise(
        duration=before + after,
        post_trigger_duration=after,
        sampling_frequency=128.0,
        trigger_time=0.0,
        f_min=f_min,
        f_max=64.0,
    )
    factory = RippleIMRPhenomXHM if higher_modes else RippleIMRPhenomD
    model = factory(f_ref=f_min)
    params = {
        "M_c": chirp_mass,
        "eta": eta,
        "s1_z": spins[0],
        "s2_z": spins[1],
        "t_c": 0.0,
    }
    if long_enough:
        detector._check_signal_duration(model, params, 0.0, False)
    else:
        with pytest.raises(ValueError, match="does not fit the segment"):
            detector._check_signal_duration(model, params, 0.0, False)
