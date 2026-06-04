"""Unit tests for transform inference logic."""

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


def _make_pipeline_cfg(prior_raw=None, sky_frame="detector", time_frame="detector"):
    """Build a minimal PipelineConfig, merging prior_raw on top of _MINIMAL_PRIOR."""
    return PipelineConfig.model_validate(
        {
            "data": _MINIMAL_DATA,
            "waveform": {"approximant": "IMRPhenomXAS"},
            "prior": {**_MINIMAL_PRIOR, **(prior_raw or {})},
            "likelihood": {"f_min": 20.0, "f_max": 1024.0},
            "sampler": {"type": "flowmc"},
            "output": {"dir": "tests/tmp/test"},
            "sampling": {"sky_frame": sky_frame, "time_frame": time_frame},
        }
    )


def _make_ifos():
    from jimgw.core.single_event.detector import get_detector_preset

    preset = get_detector_preset()
    return [preset["H1"], preset["L1"]]


TRIGGER_TIME = 1126259462.4


def _infer(prior_params, sky_frame="detector", time_frame="detector"):

    from jimgw.cli._transforms import (
        infer_likelihood_transforms,
        infer_sample_transforms,
    )

    ifos = _make_ifos()
    cfg = SamplingConfig(sky_frame=sky_frame, time_frame=time_frame)  # type: ignore[arg-type]
    prior_cfg = PriorConfig.model_validate({})
    sample_t = infer_sample_transforms(
        frozenset(prior_params),
        TRIGGER_TIME,
        ifos,
        cfg,
        prior_cfg=prior_cfg,
    )
    lh_t = infer_likelihood_transforms(
        frozenset(prior_params),
        TRIGGER_TIME,
        ifos,
        cfg,
        20.0,
    )
    return sample_t, lh_t


# ---------------------------------------------------------------------------
# Standard CBC case (the GW150914 default)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gw150914_default_transforms():
    params = {
        "M_c",
        "q",
        "s1_z",
        "s2_z",
        "iota",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "ra",
        "dec",
    }
    sample_t, lh_t = _infer(params)

    sample_names = [type(t).__name__ for t in sample_t]
    assert "GeocentricArrivalTimeToDetectorArrivalTimeTransform" in sample_names
    assert "SkyFrameToDetectorFrameSkyPositionTransform" in sample_names
    assert len(sample_t) == 2

    # Only q→eta in likelihood space; reverse sky/time handled by reversed sample transforms
    assert len(lh_t) == 1
    assert "q" in str(lh_t[0].name_mapping)


# ---------------------------------------------------------------------------
# Geocentric sky (no sky sample transform)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_geocentric_sky_no_sky_sample_transform():
    params = {
        "M_c",
        "q",
        "iota",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "ra",
        "dec",
        "s1_z",
        "s2_z",
    }
    sample_t, _ = _infer(params, sky_frame="geocentric")

    sample_names = [type(t).__name__ for t in sample_t]
    assert "SkyFrameToDetectorFrameSkyPositionTransform" not in sample_names
    assert "GeocentricArrivalTimeToDetectorArrivalTimeTransform" in sample_names


# ---------------------------------------------------------------------------
# Geocentric time frame (t_c sampled directly)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_geocentric_time_no_time_sample_transform():
    params = {
        "M_c",
        "q",
        "iota",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "ra",
        "dec",
        "s1_z",
        "s2_z",
    }
    sample_t, _ = _infer(params, time_frame="geocentric")

    sample_names = [type(t).__name__ for t in sample_t]
    assert "GeocentricArrivalTimeToDetectorArrivalTimeTransform" not in sample_names
    assert "SkyFrameToDetectorFrameSkyPositionTransform" in sample_names


# ---------------------------------------------------------------------------
# Detector-frame sky prior (azimuth/zenith in prior)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_detector_frame_sky_prior():
    params = {
        "M_c",
        "q",
        "iota",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "azimuth",
        "zenith",
        "s1_z",
        "s2_z",
    }
    sample_t, lh_t = _infer(params)

    sample_names = [type(t).__name__ for t in sample_t]
    # No sky sample transform (already in detector frame)
    assert "SkyFrameToDetectorFrameSkyPositionTransform" not in sample_names
    # Reverse sky transform must appear in likelihood transforms
    lh_repr = [repr(t) for t in lh_t]
    assert any("zenith" in r or "azimuth" in r for r in lh_repr)


# ---------------------------------------------------------------------------
# Detector-frame time prior (t_det in prior)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_detector_frame_time_prior():
    params = {
        "M_c",
        "q",
        "iota",
        "d_L",
        "t_det",
        "phase_c",
        "psi",
        "ra",
        "dec",
        "s1_z",
        "s2_z",
    }
    sample_t, lh_t = _infer(params)

    sample_names = [type(t).__name__ for t in sample_t]
    # No time sample transform
    assert "GeocentricArrivalTimeToDetectorArrivalTimeTransform" not in sample_names
    # Reverse time transform must appear in likelihood transforms
    lh_repr = [repr(t) for t in lh_t]
    assert any("t_det" in r or "t_c" in r for r in lh_repr)


# ---------------------------------------------------------------------------
# J-frame spin angles
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_j_frame_spin_angles():
    params = {
        "M_c",
        "q",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "ra",
        "dec",
        "theta_jn",
        "phi_jl",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "a_1",
        "a_2",
    }
    sample_t, lh_t = _infer(params)

    # Spin physics transforms are in likelihood_transforms (prior → likelihood space),
    # not in sample_transforms — so the sampler explores in J-frame angle space.
    sample_names = [type(t).__name__ for t in sample_t]
    assert "SpinAnglesToCartesianSpinTransform" not in sample_names

    lh_names = [type(t).__name__ for t in lh_t]
    assert "SpinAnglesToCartesianSpinTransform" in lh_names


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


# ---------------------------------------------------------------------------
# Spherical per-spin
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sphere_spin_transform():
    params = {
        "M_c",
        "q",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "iota",
        "ra",
        "dec",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
    }
    sample_t, lh_t = _infer(params)

    # Sphere spin physics transforms are in likelihood_transforms, not sample_transforms.
    sample_names = [type(t).__name__ for t in sample_t]
    assert "SphereSpinToCartesianSpinTransform" not in sample_names

    lh_names = [type(t).__name__ for t in lh_t]
    assert lh_names.count("SphereSpinToCartesianSpinTransform") == 2


# ---------------------------------------------------------------------------
# Cartesian spins — no transform needed
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cartesian_spins_no_transform():
    params = {
        "M_c",
        "q",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "iota",
        "ra",
        "dec",
        "s1_x",
        "s1_y",
        "s1_z",
        "s2_x",
        "s2_y",
        "s2_z",
    }
    sample_t, _ = _infer(params)

    sample_names = [type(t).__name__ for t in sample_t]
    assert "SphereSpinToCartesianSpinTransform" not in sample_names
    assert "SpinAnglesToCartesianSpinTransform" not in sample_names


# ---------------------------------------------------------------------------
# Validation: mutually exclusive groups
# ---------------------------------------------------------------------------


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


def test_t_det_geocentric_time_frame_raises():
    with pytest.raises(ValidationError, match="t_det"):
        _make_pipeline_cfg(
            prior_raw={"t_det": {"type": "uniform", "min": -0.1, "max": 0.1}},
            time_frame="geocentric",
        )


def test_detector_sky_geocentric_sky_frame_raises():
    with pytest.raises(ValidationError, match="azimuth"):
        _make_pipeline_cfg(
            prior_raw={
                "azimuth": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "zenith": {"type": "sine"},
            },
            sky_frame="geocentric",
        )


# ---------------------------------------------------------------------------
# adapt_prior_for_ns_time
# ---------------------------------------------------------------------------


def _make_prior_cfg(params: dict):
    """Build a PriorConfig from a dict of {name: spec_dict}."""
    return PriorConfig.model_validate(params)


def test_adapt_ns_time_converts_t_c_to_t_det():
    """t_c in prior + detector time_frame → replaced by t_det with widened GPS bounds."""
    ifos = _make_ifos()
    cfg = SamplingConfig(time_frame="detector")
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
    """time_frame='geocentric' → no conversion; t_c sampled directly is already exact."""
    ifos = _make_ifos()
    cfg = SamplingConfig(time_frame="geocentric")
    prior_cfg = _make_prior_cfg(
        {
            "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
        }
    )

    result = adapt_prior_for_ns_time(prior_cfg, cfg)

    assert result is None  # no change needed


def test_adapt_ns_time_t_det_in_prior_no_conversion():
    """User already put t_det in prior → no conversion needed."""
    ifos = _make_ifos()
    cfg = SamplingConfig(time_frame="detector")
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
    ifos = _make_ifos()
    cfg = SamplingConfig(time_frame="geocentric")
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
