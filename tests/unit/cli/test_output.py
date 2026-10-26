"""Tests for the corner plot of the CLI: the truth markers of an injection and the parameters plotted."""

import logging

import numpy as np
import pytest

from jimgw.cli._config import PipelineConfig
from jimgw.cli._data import build_data
from jimgw.cli._jim import build_jim
from jimgw.cli._likelihood import build_likelihood
from jimgw.cli._output import _injection_truths_in_prior_space, _save_corner
from jimgw.cli._prior import build_prior
from jimgw.cli._waveform import build_waveform
from jimgw.core.single_event.helpers import (
    get_default_likelihood_transforms,
    get_default_periodic,
    get_default_sample_transforms,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)

GPS_TIME = 1126259462.4
INJECTED_T_C = 0.03
RA, DEC = 1.375, -1.21


def _raw_config(time_parameter, tmp_path):
    prior = {
        "M_c": {"type": "uniform", "min": 20.0, "max": 40.0},
        "q": {"type": "uniform", "min": 0.125, "max": 1.0},
        "s1_z": {"type": "uniform", "min": -0.99, "max": 0.99},
        "s2_z": {"type": "uniform", "min": -0.99, "max": 0.99},
        "iota": {"type": "sine"},
        "d_L": {"type": "power_law", "min": 1.0, "max": 2000.0, "alpha": 2.0},
        time_parameter: {"type": "uniform", "min": -0.1, "max": 0.1},
        "phase_c": {"type": "uniform", "min": 0.0, "max": 2 * np.pi},
        "psi": {"type": "uniform", "min": 0.0, "max": np.pi},
        "ra": {"type": "uniform", "min": 0.0, "max": 2 * np.pi},
        "dec": {"type": "cosine"},
    }
    return {
        "data": {
            "type": "injection",
            "detectors": ["H1", "L1"],
            "trigger_time": GPS_TIME,
            "duration": 4.0,
            "sampling_frequency": 2048.0,
            "noise_seed": 0,
            "injection_parameters": {
                "M_c": 28.3,
                "q": 0.85,
                "s1_z": 0.3,
                "s2_z": -0.2,
                "iota": 0.4,
                "d_L": 440.0,
                "t_c": INJECTED_T_C,
                "phase_c": 0.5,
                "psi": 0.1,
                "ra": RA,
                "dec": DEC,
            },
        },
        "waveform": {"approximant": "IMRPhenomD", "f_ref": 20.0},
        "prior": prior,
        "sampling": {"detector_frame_time": True},
        "likelihood": {"f_min": 20.0, "f_max": 1024.0},
        "sampler": {
            "type": "flowmc",
            "n_chains": 2,
            "n_local_steps": 2,
            "n_global_steps": 2,
            "n_training_loops": 1,
            "n_production_loops": 1,
            "n_epochs": 1,
            "global_thinning": 1,
        },
        "output": {"dir": str(tmp_path / "output")},
    }


def _truths(time_parameter, tmp_path):
    """The truth markers that the corner plot of a run without sampling would use."""
    cfg = PipelineConfig.model_validate(_raw_config(time_parameter, tmp_path))
    waveform = build_waveform(cfg.waveform)
    ifos = build_data(
        cfg.data,
        f_min=cfg.likelihood.f_min,
        f_max=cfg.likelihood.f_max,
        waveform=waveform,
    )
    prior = build_prior(cfg.prior)
    sample_transforms = get_default_sample_transforms(prior, ifos)
    likelihood_transforms = get_default_likelihood_transforms(
        prior,
        ifos,
        waveform,
        phase_marginalization=cfg.likelihood.phase_marginalization,
    )
    likelihood = build_likelihood(
        cfg.likelihood,
        ifos,
        waveform,
        prior=prior,
        likelihood_transforms=likelihood_transforms,
        data_cfg=cfg.data,
    )
    jim = build_jim(
        likelihood,
        prior,
        sample_transforms,
        likelihood_transforms,
        get_default_periodic(prior, sample_transforms),
        cfg,
        verbose=False,
    )
    truths = _injection_truths_in_prior_space(
        cfg.data.injection_parameters,
        jim.likelihood_transforms,
        waveform,
        ifos=list(jim.likelihood.detectors),
        jim=jim,
    )
    return truths, ifos


class TestInjectionTruths:
    """An injection is given by its geocentric ``t_c``; the prior may use either time."""

    @pytest.mark.slow
    def test_the_truth_follows_the_time_parameter_of_the_prior(self, tmp_path):
        truths_t_c, _ = _truths("t_c", tmp_path)
        truths_t_det, ifos = _truths("t_det", tmp_path)

        assert truths_t_c is not None and truths_t_det is not None
        assert "t_det" not in truths_t_c
        assert "t_c" not in truths_t_det
        np.testing.assert_allclose(truths_t_c["t_c"], INJECTED_T_C)
        # With the default time frame, the arrival time is that at the first detector.
        delay = ifos[0].delay_from_geocenter(RA, DEC, compute_gmst(GPS_TIME))
        np.testing.assert_allclose(truths_t_det["t_det"], INJECTED_T_C + float(delay))
        # Both are the same point, so the likelihood there is the same.
        np.testing.assert_allclose(
            truths_t_det["log_likelihood"],
            truths_t_c["log_likelihood"],
            rtol=0.0,
            atol=1e-6,
        )


class TestCornerParameters:
    @staticmethod
    def _samples():
        rng = np.random.default_rng(0)
        return {name: rng.normal(size=50) for name in ("M_c", "q", "t_det")}

    def test_a_missing_corner_parameter_is_reported(
        self, tmp_path, caplog, monkeypatch
    ):
        # The "jimgw" logger does not propagate to the root logger, where caplog listens.
        monkeypatch.setattr(logging.getLogger("jimgw"), "propagate", True)
        with caplog.at_level("WARNING"):
            _save_corner(tmp_path, self._samples(), ["M_c", "t_c"])
        assert (tmp_path / "corner.png").exists()
        message = " ".join(record.message for record in caplog.records)
        assert "['t_c']" in message
        assert "M_c, q, t_det" in message

    def test_nothing_is_reported_when_all_are_sampled(
        self, tmp_path, caplog, monkeypatch
    ):
        monkeypatch.setattr(logging.getLogger("jimgw"), "propagate", True)
        with caplog.at_level("WARNING"):
            _save_corner(tmp_path, self._samples(), ["M_c", "t_det"])
        assert "corner_parameters" not in " ".join(r.message for r in caplog.records)


@pytest.mark.parametrize("names", [None, ["M_c", "q", "t_det"]])
def test_the_corner_plot_is_written(tmp_path, names):
    _save_corner(tmp_path, TestCornerParameters._samples(), names)
    assert (tmp_path / "corner.png").exists()
