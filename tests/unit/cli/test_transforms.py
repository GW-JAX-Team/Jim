"""Config validation of the transform-relevant parameter groups, and NS AW time adaptation."""

import pytest
from pydantic import ValidationError

from jimgw.cli._config import PipelineConfig, PriorConfig, SamplingConfig, UniformSpec
from jimgw.cli._prior import adapt_prior_for_ns_time

_MINIMAL_DATA = {
    "type": "gwosc",
    "detectors": ["H1", "L1"],
    "trigger_time": 1126259462.4,
    "duration": 4.0,
    "psd_duration": 1024.0,
}

_MINIMAL_PRIOR = {
    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
}


def _make_pipeline_cfg(
    prior_raw=None, detector_frame_sky=True, detector_frame_time=True
):
    """Build a minimal PipelineConfig, merging prior_raw on top of _MINIMAL_PRIOR."""
    return PipelineConfig.model_validate(
        {
            "data": _MINIMAL_DATA,
            "waveform": {"approximant": "IMRPhenomXAS"},
            "prior": {**_MINIMAL_PRIOR, **(prior_raw or {})},
            "likelihood": {"f_min": 20.0, "f_max": 1024.0},
            "sampler": {"type": "flowmc"},
            "output": {"dir": "tests/tmp/test"},
            "sampling": {
                "detector_frame_sky": detector_frame_sky,
                "detector_frame_time": detector_frame_time,
            },
        }
    )


TRIGGER_TIME = 1126259462.4


def test_j_frame_iota_conflict():
    with pytest.raises(ValidationError, match="iota"):
        _make_pipeline_cfg(
            prior_raw={
                "theta_jn": {"type": "uniform", "min": 0.0, "max": 3.14159},
                "phi_jl": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "tilt_1": {"type": "sine"},
                "tilt_2": {"type": "sine"},
                "phi_12": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "a_1": {"type": "uniform", "min": 0.0, "max": 0.99},
                "a_2": {"type": "uniform", "min": 0.0, "max": 0.99},
                "iota": {"type": "sine"},  # conflict
                "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                "q": {"type": "uniform", "min": 0.125, "max": 1.0},
            }
        )


def test_j_frame_partial_params_rejected():
    with pytest.raises(ValidationError, match="missing"):
        _make_pipeline_cfg(
            prior_raw={
                "theta_jn": {"type": "uniform", "min": 0.0, "max": 3.14159},
                "phi_jl": {"type": "uniform", "min": 0.0, "max": 6.28318},
                # only 2 of 7 J-frame params
                "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                "q": {"type": "uniform", "min": 0.125, "max": 1.0},
            }
        )


# ----------------------------------------------------------------------
# Validation: mutually exclusive groups
# ----------------------------------------------------------------------


def test_spin_groups_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _make_pipeline_cfg(
            prior_raw={
                "s1_z": {"type": "uniform", "min": -0.99, "max": 0.99},
                "s2_z": {"type": "uniform", "min": -0.99, "max": 0.99},
                "s1_mag": {"type": "uniform", "min": 0.0, "max": 0.99},
                "s1_theta": {"type": "sine"},
                "s1_phi": {"type": "uniform", "min": 0.0, "max": 6.28318},
            }
        )


def test_sky_groups_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _make_pipeline_cfg(
            prior_raw={
                "ra": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "dec": {"type": "cosine"},
                "azimuth": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "zenith": {"type": "sine"},
            }
        )


def test_time_groups_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _make_pipeline_cfg(
            prior_raw={
                "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
                "t_det": {"type": "uniform", "min": -0.1, "max": 0.1},
            }
        )


def test_t_det_without_detector_frame_time_raises():
    with pytest.raises(ValidationError, match="t_det"):
        _make_pipeline_cfg(
            prior_raw={"t_det": {"type": "uniform", "min": -0.1, "max": 0.1}},
            detector_frame_time=False,
        )


def test_detector_sky_without_detector_frame_sky_raises():
    with pytest.raises(ValidationError, match="azimuth"):
        _make_pipeline_cfg(
            prior_raw={
                "azimuth": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "zenith": {"type": "sine"},
            },
            detector_frame_sky=False,
        )


# ----------------------------------------------------------------------
# adapt_prior_for_ns_time
# ----------------------------------------------------------------------


def _make_prior_cfg(params: dict):
    """Build a PriorConfig from a dict of {name: spec_dict}."""
    return PriorConfig.model_validate(params)


def test_adapt_ns_time_converts_t_c_to_t_det():
    """t_c in prior + detector-frame time → replaced by t_det with widened GPS bounds."""
    cfg = SamplingConfig(detector_frame_time=True)
    prior_cfg = _make_prior_cfg(
        {
            "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
            "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
            "d_L": {"type": "power_law", "min": 1.0, "max": 2000.0, "alpha": 2.0},
        }
    )

    result = adapt_prior_for_ns_time(prior_cfg, cfg)

    assert result is not None
    assert "t_c" not in result.root
    assert "t_det" in result.root
    spec = result.root["t_det"]
    assert isinstance(spec, UniformSpec)
    assert spec.min == prior_cfg.root["t_c"].min
    assert spec.max == prior_cfg.root["t_c"].max
    # Insertion order preserved: t_det sits where t_c was
    assert list(result.root.keys()) == ["M_c", "t_det", "d_L"]


def test_adapt_ns_time_geocentric_no_conversion():
    """detector_frame_time=False → no conversion; t_c sampled directly is already exact."""
    cfg = SamplingConfig(detector_frame_time=False)
    prior_cfg = _make_prior_cfg(
        {
            "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
        }
    )

    result = adapt_prior_for_ns_time(prior_cfg, cfg)

    assert result is None  # no change needed


def test_adapt_ns_time_t_det_in_prior_no_conversion():
    """User already put t_det in prior → no conversion needed."""
    cfg = SamplingConfig(detector_frame_time=True)
    lo = TRIGGER_TIME - 0.15
    hi = TRIGGER_TIME + 0.15
    prior_cfg = _make_prior_cfg(
        {
            "t_det": {"type": "uniform", "min": lo, "max": hi},
        }
    )

    result = adapt_prior_for_ns_time(prior_cfg, cfg)

    assert result is None  # t_det already in prior, no change


def test_adapt_ns_time_t_det_geocentric():
    """t_det in prior + geocentric sampling → adapted to widened t_c prior."""
    cfg = SamplingConfig(detector_frame_time=False)
    lo = TRIGGER_TIME - 0.15
    hi = TRIGGER_TIME + 0.15
    prior_cfg = _make_prior_cfg({"t_det": {"type": "uniform", "min": lo, "max": hi}})

    result = adapt_prior_for_ns_time(prior_cfg, cfg)

    assert result is not None
    assert "t_c" in result.root
    assert "t_det" not in result.root
    spec = result.root["t_c"]
    assert isinstance(spec, UniformSpec)
    assert spec.min == lo
    assert spec.max == hi
