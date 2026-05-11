"""Unit tests for CLI config schema (TOML round-trips, validation, prior parsing)."""

import tomllib

import pytest
from pydantic import ValidationError

from jimgw.cli._config import (
    CosineSpec,
    FileDataConfig,
    GWOSCDataConfig,
    InjectionDataConfig,
    PipelineConfig,
    PowerLawSpec,
    PriorConfig,
    RayleighSpec,
    SineSpec,
    UniformSpec,
    GaussianSpec,
    WaveformConfig,
)

_MINIMAL_RAW = {
    "data": {
        "type": "gwosc",
        "detectors": ["H1", "L1"],
        "trigger_time": 1126259462.4,
        "duration": 4.0,
        "psd_duration": 1024.0,
    },
    "waveform": {"approximant": "IMRPhenomXAS"},
    "prior": {
        "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
        "q": {"type": "uniform", "min": 0.125, "max": 1.0},
    },
    "likelihood": {"f_min": 20.0, "f_max": 1024.0},
    "sampler": {"type": "flowmc"},
    "output": {"dir": "tests/tmp/test"},
}


def test_pipeline_config_minimal():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert isinstance(cfg.data, GWOSCDataConfig)
    assert cfg.data.detectors == ["H1", "L1"]
    assert cfg.waveform.approximant == "IMRPhenomXAS"
    assert cfg.waveform.f_ref == 20.0  # default
    assert cfg.seed == 0  # default


def test_pipeline_config_file_data():
    raw = {
        **_MINIMAL_RAW,
        "data": {
            "type": "file",
            "detectors": ["H1"],
            "trigger_time": 1126259462.4,
            "strain_files": {"H1": "tests/fixtures/GW150914_strain_H1.npz"},
            "psd_files": {"H1": "tests/fixtures/GW150914_psd_H1.npz"},
        },
    }
    cfg = PipelineConfig.model_validate(raw)
    assert isinstance(cfg.data, FileDataConfig)


def test_pipeline_config_injection_data():
    raw = {
        **_MINIMAL_RAW,
        "data": {
            "type": "injection",
            "detectors": ["H1"],
            "trigger_time": 1126259462.4,
            "duration": 4.0,
            "sampling_frequency": 2048.0,
            "injection_parameters": {
                "M_c": 28.3,
                "q": 0.85,
                "s1_z": 0.0,
                "s2_z": 0.0,
                "iota": 0.4,
                "d_L": 440.0,
                "t_c": 0.0,
                "phase_c": 0.0,
                "psi": 0.0,
                "ra": 1.375,
                "dec": -1.21,
            },
        },
    }
    cfg = PipelineConfig.model_validate(raw)
    assert isinstance(cfg.data, InjectionDataConfig)
    assert cfg.data.zero_noise is False  # default


def test_prior_spec_uniform():
    cfg = PriorConfig.model_validate({"M_c": {"type": "uniform", "min": 10.0, "max": 80.0}})
    spec = cfg.root["M_c"]
    assert isinstance(spec, UniformSpec)
    assert spec.min == 10.0
    assert spec.max == 80.0


def test_prior_spec_rayleigh():
    cfg = PriorConfig.model_validate({"sigma": {"type": "rayleigh", "scale": 15.0}})
    spec = cfg.root["sigma"]
    assert isinstance(spec, RayleighSpec)
    assert spec.scale == 15.0


def test_prior_spec_gaussian():
    cfg = PriorConfig.model_validate({"x": {"type": "gaussian", "loc": 2.0, "scale": 0.5}})
    spec = cfg.root["x"]
    assert isinstance(spec, GaussianSpec)
    assert spec.loc == 2.0
    assert spec.scale == 0.5


def test_prior_spec_sine():
    cfg = PriorConfig.model_validate({"iota": {"type": "sine"}})
    assert isinstance(cfg.root["iota"], SineSpec)


def test_prior_spec_cosine():
    cfg = PriorConfig.model_validate({"dec": {"type": "cosine"}})
    assert isinstance(cfg.root["dec"], CosineSpec)


def test_prior_spec_power_law():
    cfg = PriorConfig.model_validate(
        {"d_L": {"type": "power_law", "min": 1.0, "max": 2000.0, "alpha": 2.0}}
    )
    spec = cfg.root["d_L"]
    assert isinstance(spec, PowerLawSpec)
    assert spec.min == 1.0
    assert spec.max == 2000.0
    assert spec.alpha == 2.0


def test_prior_insertion_order_preserved():
    params = ["d_L", "M_c", "q", "iota", "dec"]
    raw_prior = {
        "d_L": {"type": "power_law", "min": 1.0, "max": 2000.0, "alpha": 2.0},
        "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
        "q": {"type": "uniform", "min": 0.125, "max": 1.0},
        "iota": {"type": "sine"},
        "dec": {"type": "cosine"},
    }
    cfg = PriorConfig.model_validate(raw_prior)
    assert list(cfg.root.keys()) == params


def test_unknown_approximant_rejected():
    raw = {**_MINIMAL_RAW, "waveform": {"approximant": "NonExistent"}}
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)


def test_extra_fields_rejected():
    raw = {**_MINIMAL_RAW, "unknown_section": {"foo": 1}}
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)


def test_dump_resolved_round_trip():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    dumped = cfg.model_dump(mode="json")
    # Strip inactive FlowMC kernel sub-configs (mirrors _output.py logic)
    if dumped.get("sampler", {}).get("type") == "flowmc":
        active = dumped["sampler"]["local_kernel"].lower()
        for kernel in ("mala", "hmc", "grw"):
            if kernel != active:
                dumped["sampler"].pop(kernel, None)
    cfg2 = PipelineConfig.model_validate(dumped)
    assert cfg.waveform.approximant == cfg2.waveform.approximant
    assert cfg.seed == cfg2.seed


def test_file_config_from_toml():
    with open("tests/fixtures/GW150914_test.toml", "rb") as f:
        raw = tomllib.load(f)
    cfg = PipelineConfig.model_validate(raw)
    assert isinstance(cfg.data, FileDataConfig)
    assert cfg.data.trigger_time == 1126259462.4
    assert len(cfg.prior.root) == 11
    assert cfg.sampler.type == "flowmc"  # type: ignore[union-attr]


def test_sampling_config_defaults():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert cfg.sampling.time_frame == "detector"
    assert cfg.sampling.sky_frame == "detector"


def test_likelihood_config_values():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert cfg.likelihood.f_min == 20.0
    assert cfg.likelihood.f_max == 1024.0
    assert cfg.likelihood.phase_marginalization is False
    assert cfg.likelihood.time_marginalization is None
    assert cfg.likelihood.distance_marginalization is None


def test_waveform_config_f_ref_default():
    cfg = WaveformConfig.model_validate({"approximant": "IMRPhenomD"})
    assert cfg.f_ref == 20.0


def test_file_data_config_missing_strain_rejected():
    with pytest.raises(ValidationError, match="strain_files missing"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "data": {
                    "type": "file",
                    "detectors": ["H1", "L1"],
                    "trigger_time": 1126259462.4,
                    "strain_files": {"H1": "tests/fixtures/GW150914_strain_H1.npz"},
                    "psd_files": {
                        "H1": "tests/fixtures/GW150914_psd_H1.npz",
                        "L1": "tests/fixtures/GW150914_psd_L1.npz",
                    },
                },
            }
        )


def test_file_data_config_missing_psd_rejected():
    with pytest.raises(ValidationError, match="psd_files missing"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "data": {
                    "type": "file",
                    "detectors": ["H1", "L1"],
                    "trigger_time": 1126259462.4,
                    "strain_files": {
                        "H1": "tests/fixtures/GW150914_strain_H1.npz",
                        "L1": "tests/fixtures/GW150914_strain_L1.npz",
                    },
                    "psd_files": {"H1": "tests/fixtures/GW150914_psd_H1.npz"},
                },
            }
        )
