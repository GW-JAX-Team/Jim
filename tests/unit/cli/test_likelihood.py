"""Tests for building a likelihood from the CLI configuration."""

from pathlib import Path

import pytest

from jimgw.cli._config import LikelihoodConfig
from jimgw.cli._likelihood import build_likelihood
from jimgw.core.constants import EARTH_RADIUS_LIGHT_S
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
    TransientLikelihoodFD,
)
from jimgw.core.single_event.waveform import RippleIMRPhenomD

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
GPS_TIME = 1126259462.4

REFERENCE_PARAMETERS = {
    "M_c": 30.0,
    "eta": 0.249,
    "s1_z": 0.0,
    "s2_z": 0.0,
    "d_L": 400.0,
    "phase_c": 0.0,
    "t_c": 0.0,
    "iota": 0.0,
    "ra": 1.375,
    "dec": -1.2108,
    "psi": 0.0,
}


@pytest.fixture
def detectors():
    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        data = Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
        data.trigger_time = GPS_TIME
        ifo.set_data(data)
        ifo.set_psd(
            PowerSpectrum.from_file(str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz"))
        )
    return ifos


def build(config, ifos, prior=None):
    return build_likelihood(
        LikelihoodConfig.model_validate({"f_min": 20.0, "f_max": 1024.0, **config}),
        ifos,
        RippleIMRPhenomD(f_ref=20.0),
        prior,
        [],
        None,
    )


def test_the_default_is_the_plain_transient_likelihood(detectors):
    likelihood = build({}, detectors)
    assert type(likelihood) is TransientLikelihoodFD
    assert not (
        likelihood.phase_marginalization
        or likelihood.time_marginalization
        or likelihood.distance_marginalization
    )


def test_marginalizations_are_passed_on(detectors):
    likelihood = build(
        {
            "phase_marginalization": True,
            "time_marginalization": {"tc_range": [-0.05, 0.1]},
        },
        detectors,
    )
    assert likelihood.phase_marginalization
    assert likelihood.time_marginalization
    assert likelihood.tc_range == (-0.05, 0.1)


def test_the_distance_prior_is_passed_on(detectors):
    likelihood = build(
        {
            "distance_marginalization": {
                "distance_prior": {
                    "d_L": {"type": "uniform", "min": 100.0, "max": 800.0}
                },
                "n_dist_points": 50,
            }
        },
        detectors,
    )
    assert likelihood.distance_marginalization
    assert len(likelihood.log_weights) == 50


@pytest.mark.slow
def test_the_heterodyne_likelihood_uses_the_provided_reference(detectors):
    likelihood = build(
        {
            "heterodyne": {
                "n_bins": 64,
                "reference_parameters": {
                    "type": "provided",
                    "values": REFERENCE_PARAMETERS,
                },
            }
        },
        detectors,
    )
    direct = HeterodynedTransientLikelihoodFD(
        detectors,
        RippleIMRPhenomD(f_ref=20.0),
        f_min=20.0,
        f_max=1024.0,
        n_bins=64,
        reference_parameters=REFERENCE_PARAMETERS,
    )
    assert type(likelihood) is HeterodynedTransientLikelihoodFD
    assert likelihood.n_bins == direct.n_bins
    assert float(likelihood.evaluate(REFERENCE_PARAMETERS)) == float(
        direct.evaluate(REFERENCE_PARAMETERS)
    )


class TestMultibandTimeWindowFromTheDetectorTimePrior:
    """With ``t_det`` in the prior, the time window comes from its bounds."""

    @staticmethod
    def prior(upper):
        return CombinePrior([UniformPrior(-0.1, upper, ["t_det"])])

    @pytest.mark.slow
    def test_time_offset_and_delta_f_end_follow_the_prior_bounds(self, detectors):
        likelihood = build(
            {"multiband": {"reference_chirp_mass": 20.0}},
            detectors,
            prior=self.prior(0.1),
        )
        t_end = detectors[0].data.end_time - detectors[0].data.trigger_time
        assert isinstance(likelihood, MultibandedTransientLikelihoodFD)
        assert likelihood.time_offset == pytest.approx(
            t_end + 0.1 + EARTH_RADIUS_LIGHT_S
        )
        assert likelihood.delta_f_end == pytest.approx(
            100.0 / (t_end - 0.1 - EARTH_RADIUS_LIGHT_S)
        )

    @pytest.mark.slow
    def test_explicit_values_are_not_overridden(self, detectors):
        likelihood = build(
            {
                "multiband": {
                    "reference_chirp_mass": 20.0,
                    "time_offset": 2.5,
                    "delta_f_end": 40.0,
                }
            },
            detectors,
            prior=self.prior(0.1),
        )
        assert likelihood.time_offset == 2.5
        assert likelihood.delta_f_end == 40.0

    def test_an_upper_bound_beyond_the_segment_is_refused(self, detectors):
        with pytest.raises(ValueError, match="Cannot infer delta_f_end"):
            build(
                {"multiband": {"reference_chirp_mass": 20.0}},
                detectors,
                prior=self.prior(2.5),
            )
