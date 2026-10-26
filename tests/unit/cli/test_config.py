"""Unit tests for CLI config schema (TOML round-trips, validation, prior parsing)."""

import logging
import tomllib
import warnings
from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

from jimgw.cli._config import (
    CLIDistanceMargConfig,
    CLIHeterodynedConfig,
    CLIInjectionRefParams,
    CLIMultibandedConfig,
    CLIOptimizerRefParams,
    CLIProvidedRefParams,
    CLITimeMargConfig,
    CosineSpec,
    FileDataConfig,
    GaussianSpec,
    GWOSCDataConfig,
    InjectionDataConfig,
    LikelihoodConfig,
    OutputConfig,
    PipelineConfig,
    PowerLawSpec,
    PriorConfig,
    RayleighSpec,
    SineSpec,
    UniformSpec,
    UniformSphereSpec,
    WaveformConfig,
)
from jimgw.cli._jim import _with_checkpoint

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


@pytest.mark.parametrize("value", [True, 2.0, "2"])
def test_cli_seeds_and_output_count_require_integers(value):
    with pytest.raises(ValidationError, match="seed"):
        PipelineConfig.model_validate({**_MINIMAL_RAW, "seed": value})
    with pytest.raises(ValidationError, match="n_samples"):
        OutputConfig.model_validate({"dir": "tests/tmp/test", "n_samples": value})
    with pytest.raises(ValidationError, match="noise_seed"):
        InjectionDataConfig.model_validate(
            {
                "type": "injection",
                "detectors": ["H1"],
                "trigger_time": 1126259462.4,
                "duration": 4.0,
                "sampling_frequency": 2048.0,
                "injection_parameters": {},
                "noise_seed": value,
            }
        )


def test_pipeline_config_minimal():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert isinstance(cfg.data, GWOSCDataConfig)
    assert cfg.data.detectors == ["H1", "L1"]
    assert cfg.waveform.approximant == "IMRPhenomXAS"
    assert cfg.waveform.f_ref == 20.0  # default
    assert cfg.seed == 0  # default


@pytest.mark.parametrize(
    "field", ["trigger_time", "duration", "post_trigger_duration", "psd_duration"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_gwosc_timing_must_be_finite(field, bad):
    with pytest.raises(ValidationError):
        GWOSCDataConfig.model_validate({**_MINIMAL_RAW["data"], field: bad})


@pytest.mark.parametrize("post_trigger_duration", [-0.1, 4.1])
def test_gwosc_trigger_must_be_inside_requested_segment(post_trigger_duration):
    with pytest.raises(ValidationError, match="post_trigger_duration"):
        GWOSCDataConfig.model_validate(
            {**_MINIMAL_RAW["data"], "post_trigger_duration": post_trigger_duration}
        )


@pytest.mark.parametrize("field", ["trigger_time", "duration", "sampling_frequency"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_injection_timing_must_be_finite(field, bad):
    data = {
        "type": "injection",
        "detectors": ["H1"],
        "trigger_time": 1126259462.4,
        "duration": 4.0,
        "sampling_frequency": 2048.0,
        "injection_parameters": {},
        "zero_noise": True,
        field: bad,
    }
    with pytest.raises(ValidationError):
        InjectionDataConfig.model_validate(data)


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
            "noise_seed": 0,
        },
    }
    cfg = PipelineConfig.model_validate(raw)
    assert isinstance(cfg.data, InjectionDataConfig)
    assert cfg.data.zero_noise is False  # default
    assert cfg.data.noise_seed == 0


def test_prior_spec_uniform():
    cfg = PriorConfig.model_validate(
        {"M_c": {"type": "uniform", "min": 10.0, "max": 80.0}}
    )
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
    cfg = PriorConfig.model_validate(
        {"x": {"type": "gaussian", "loc": 2.0, "scale": 0.5}}
    )
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


def test_prior_spec_uniform_sphere():
    cfg = PriorConfig.model_validate({"s1": {"type": "uniform_sphere"}})
    assert isinstance(cfg.root["s1"], UniformSphereSpec)


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
    dumped = cfg.model_dump(mode="json", exclude_none=True)
    cfg2 = PipelineConfig.model_validate(dumped)
    assert cfg.waveform.approximant == cfg2.waveform.approximant
    assert cfg.seed == cfg2.seed


def test_checkpoint_defaults_applied_without_warnings():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert cfg.sampler.type == "flowmc"

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        sampler_config = _with_checkpoint(cfg.sampler, Path("checkpoints"))

    assert not captured_warnings
    assert sampler_config.checkpoint_dir == Path("checkpoints")
    assert sampler_config.checkpoint_interval == 600.0


def test_explicit_none_checkpoint_dir_disables_cli_checkpointing():
    cfg = PipelineConfig.model_validate(
        {
            **_MINIMAL_RAW,
            "sampler": {
                "type": "blackjax-smc",
                "checkpoint_dir": None,
            },
        }
    )

    sampler_config = _with_checkpoint(cfg.sampler, Path("checkpoints"))

    assert sampler_config.checkpoint_dir is None
    assert sampler_config.checkpoint_interval == 0.0


def test_file_config_from_toml():
    with open("tests/fixtures/GW150914_test.toml", "rb") as f:
        raw = tomllib.load(f)
    cfg = PipelineConfig.model_validate(raw)
    assert isinstance(cfg.data, FileDataConfig)
    assert cfg.data.trigger_time == 1126259462.4
    assert len(cfg.prior.root) == 11
    assert cfg.sampler.type == "flowmc"


def test_sampling_config_defaults():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert cfg.sampling.detector_frame_time is True
    assert cfg.sampling.detector_frame_sky is True


def test_likelihood_config_values():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert cfg.likelihood.f_min == 20.0
    assert cfg.likelihood.f_max == 1024.0
    assert cfg.likelihood.phase_marginalization is False
    assert cfg.likelihood.time_marginalization is None
    assert cfg.likelihood.distance_marginalization is None


def test_optimizer_ref_params_target_defaults_to_none():
    cfg = CLIOptimizerRefParams.model_validate({})
    assert cfg.popsize == 500
    assert cfg.n_steps == 1000
    assert cfg.target is None


def test_optimizer_ref_params_target_parses():
    cfg = CLIOptimizerRefParams.model_validate({"target": -1234.5})
    assert cfg.target == -1234.5


def test_provided_ref_params_parses():
    cfg = CLIHeterodynedConfig.model_validate(
        {
            "reference_parameters": {
                "type": "provided",
                "values": {"M_c": 28.3, "q": 0.85},
            }
        }
    )
    assert isinstance(cfg.reference_parameters, CLIProvidedRefParams)
    assert cfg.reference_parameters.values == {"M_c": 28.3, "q": 0.85}


def test_injection_ref_params_parses():
    cfg = CLIHeterodynedConfig.model_validate(
        {"reference_parameters": {"type": "injection"}}
    )
    assert isinstance(cfg.reference_parameters, CLIInjectionRefParams)


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


# ----------------------------------------------------------------------
# PSD / ASD sources (file and injection data)
# ----------------------------------------------------------------------

_STRAIN_H1 = "tests/fixtures/GW150914_strain_H1.npz"
_PSD_H1 = "tests/fixtures/GW150914_psd_H1.npz"
_INJECTION_PARAMETERS = {
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
}


def _file_data(**extra):
    return {
        "type": "file",
        "detectors": ["H1"],
        "trigger_time": 1126259462.4,
        "strain_files": {"H1": _STRAIN_H1},
        **extra,
    }


def _injection_data(detectors=("H1",), **extra):
    return {
        "type": "injection",
        "detectors": list(detectors),
        "trigger_time": 1126259462.4,
        "duration": 4.0,
        "sampling_frequency": 2048.0,
        "injection_parameters": _INJECTION_PARAMETERS,
        "noise_seed": 0,
        **extra,
    }


def test_file_data_accepts_asd_files():
    cfg = FileDataConfig.model_validate(_file_data(asd_files={"H1": "h1_asd.txt"}))
    assert cfg.asd_files == {"H1": Path("h1_asd.txt")}
    assert cfg.psd_files is None


def test_file_data_requires_psd_or_asd_files():
    with pytest.raises(ValidationError, match="provide psd_files or asd_files"):
        FileDataConfig.model_validate(_file_data())


@pytest.mark.parametrize(
    "model, raw",
    [
        (FileDataConfig, _file_data),
        (InjectionDataConfig, _injection_data),
    ],
)
def test_psd_and_asd_files_are_mutually_exclusive(model, raw):
    with pytest.raises(ValidationError, match="mutually exclusive"):
        model.model_validate(
            raw(psd_files={"H1": _PSD_H1}, asd_files={"H1": "h1_asd.txt"})
        )


def test_asd_files_accept_npz_archives():
    # The table, not the file format, decides whether the values are a PSD or an ASD.
    cfg = FileDataConfig.model_validate(_file_data(asd_files={"H1": "h1_asd.npz"}))
    assert cfg.asd_files == {"H1": Path("h1_asd.npz")}


def test_psd_files_reject_unsupported_suffix():
    with pytest.raises(ValidationError, match="expected a"):
        FileDataConfig.model_validate(_file_data(psd_files={"H1": "h1_psd.fits"}))


def test_asd_files_must_name_every_detector():
    with pytest.raises(ValidationError, match=r"asd_files missing for: \['L1'\]"):
        InjectionDataConfig.model_validate(
            _injection_data(("H1", "L1"), asd_files={"H1": "h1_asd.txt"})
        )


def test_injection_without_files_uses_built_in_default_for_every_detector():
    cfg = InjectionDataConfig.model_validate(
        _injection_data(("H1", "L1", "V1", "ET", "ET_2L", "CE"))
    )
    assert cfg.psd_files is None
    assert cfg.asd_files is None


def test_injection_accepts_psd_files_for_ce():
    cfg = InjectionDataConfig.model_validate(
        _injection_data(("CE",), psd_files={"CE": "ce_psd.txt"})
    )
    assert cfg.psd_files == {"CE": Path("ce_psd.txt")}


def test_response_options_need_a_next_generation_detector():
    # On H1/L1/V1 the keys would do nothing.
    with pytest.raises(ValidationError, match="apply only to ET, ET_2L and CE"):
        InjectionDataConfig.model_validate(
            _injection_data(("H1",), earth_rotation=True)
        )


def test_injection_files_must_cover_every_detector():
    # No silent fallback to the built-in default for a detector missing from the table.
    with pytest.raises(ValidationError, match=r"asd_files missing for: \['H1'\]"):
        InjectionDataConfig.model_validate(
            _injection_data(("H1", "CE"), asd_files={"CE": "ce_asd.txt"})
        )


def test_gwosc_data_has_no_psd_file_option():
    with pytest.raises(ValidationError, match="psd_files"):
        GWOSCDataConfig.model_validate(
            {
                "type": "gwosc",
                "detectors": ["H1"],
                "trigger_time": 1126259462.4,
                "duration": 4.0,
                "psd_duration": 64.0,
                "psd_files": {"H1": _PSD_H1},
            }
        )


def test_resolved_config_with_asd_files_serialises_to_toml():
    # config.final.toml is written from model_dump(exclude_none=True); the unset
    # psd_files table must not reach TOML (it has no null).
    cfg = InjectionDataConfig.model_validate(
        _injection_data(asd_files={"H1": "h1_asd.txt"})
    )
    dumped = cfg.model_dump(mode="json", exclude_none=True)
    assert "psd_files" not in dumped
    assert dumped["noise_seed"] == 0  # recorded, so the run can be reproduced
    round_trip = tomllib.loads(tomli_w.dumps({"data": dumped}))["data"]
    assert InjectionDataConfig.model_validate(round_trip).asd_files == cfg.asd_files


# ----------------------------------------------------------------------
# Injected-noise seed
# ----------------------------------------------------------------------


def test_noisy_injection_requires_noise_seed():
    with pytest.raises(ValidationError, match="noise_seed is required"):
        InjectionDataConfig.model_validate(_injection_data(noise_seed=None))


def test_zero_noise_injection_warns_about_a_noise_seed(caplog, monkeypatch):
    # No noise is drawn, so the seed is useless; the config stays valid and keeps it.
    # The "jimgw" logger does not propagate, which hides records from caplog.
    monkeypatch.setattr(logging.getLogger("jimgw"), "propagate", True)
    with caplog.at_level("WARNING"):
        cfg = InjectionDataConfig.model_validate(
            _injection_data(zero_noise=True, noise_seed=3)
        )
    assert any("noise_seed is ignored" in r.message for r in caplog.records)
    assert cfg.noise_seed == 3


def test_zero_noise_injection_needs_no_noise_seed(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("jimgw"), "propagate", True)
    with caplog.at_level("WARNING"):
        cfg = InjectionDataConfig.model_validate(
            _injection_data(zero_noise=True, noise_seed=None)
        )
    assert not [r for r in caplog.records if "noise_seed" in r.message]
    assert cfg.noise_seed is None
    assert "noise_seed" not in cfg.model_dump(mode="json", exclude_none=True)


def test_noise_seed_only_exists_for_injection_data():
    with pytest.raises(ValidationError, match="noise_seed"):
        FileDataConfig.model_validate(
            _file_data(psd_files={"H1": _PSD_H1}, noise_seed=1)
        )


# ----------------------------------------------------------------------
# Prior spec validators
# ----------------------------------------------------------------------


def test_uniform_spec_inverted_bounds_rejected():
    with pytest.raises(ValidationError, match="min < max"):
        PriorConfig.model_validate({"x": {"type": "uniform", "min": 5.0, "max": 1.0}})


def test_uniform_spec_equal_bounds_rejected():
    with pytest.raises(ValidationError, match="min < max"):
        PriorConfig.model_validate({"x": {"type": "uniform", "min": 3.0, "max": 3.0}})


def test_power_law_spec_inverted_bounds_rejected():
    with pytest.raises(ValidationError, match="min < max"):
        PriorConfig.model_validate(
            {"d_L": {"type": "power_law", "min": 2000.0, "max": 1.0, "alpha": 2.0}}
        )


def test_power_law_spec_equal_bounds_rejected():
    with pytest.raises(ValidationError, match="min < max"):
        PriorConfig.model_validate(
            {"d_L": {"type": "power_law", "min": 3.0, "max": 3.0, "alpha": 2.0}}
        )


def test_gaussian_spec_zero_scale_rejected():
    with pytest.raises(ValidationError, match="scale > 0"):
        PriorConfig.model_validate(
            {"x": {"type": "gaussian", "loc": 0.0, "scale": 0.0}}
        )


def test_gaussian_spec_negative_scale_rejected():
    with pytest.raises(ValidationError, match="scale > 0"):
        PriorConfig.model_validate(
            {"x": {"type": "gaussian", "loc": 0.0, "scale": -1.0}}
        )


def test_rayleigh_spec_zero_scale_rejected():
    with pytest.raises(ValidationError, match="scale > 0"):
        PriorConfig.model_validate({"sigma": {"type": "rayleigh", "scale": 0.0}})


def test_rayleigh_spec_negative_scale_rejected():
    with pytest.raises(ValidationError, match="scale > 0"):
        PriorConfig.model_validate({"sigma": {"type": "rayleigh", "scale": -0.5}})


# ----------------------------------------------------------------------
# Detector list validators
# ----------------------------------------------------------------------


def test_duplicate_detectors_rejected():
    with pytest.raises(ValidationError, match="Duplicate"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "data": {**_MINIMAL_RAW["data"], "detectors": ["H1", "H1"]},
            }
        )


# ----------------------------------------------------------------------
# Sky parametrization completeness
# ----------------------------------------------------------------------


def test_incomplete_equatorial_sky_rejected():
    with pytest.raises(ValidationError, match="both 'ra' and 'dec'"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "prior": {
                    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
                    "ra": {"type": "uniform", "min": 0.0, "max": 6.283},
                    # dec missing
                },
            }
        )


def test_incomplete_detector_sky_rejected():
    with pytest.raises(ValidationError, match="both 'azimuth' and 'zenith'"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "prior": {
                    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
                    "azimuth": {"type": "uniform", "min": 0.0, "max": 6.283},
                    # zenith missing
                },
                "sampling": {"detector_frame_sky": True},
            }
        )


# ----------------------------------------------------------------------
# Parametrization groups
# ----------------------------------------------------------------------

_FULL_PERIOD = {"type": "uniform", "min": 0.0, "max": 6.28318}
_J_FRAME_PRIOR = {
    "theta_jn": {"type": "uniform", "min": 0.0, "max": 3.14159},
    "phi_jl": _FULL_PERIOD,
    "tilt_1": {"type": "sine"},
    "tilt_2": {"type": "sine"},
    "phi_12": _FULL_PERIOD,
    "a_1": {"type": "uniform", "min": 0.0, "max": 0.99},
    "a_2": {"type": "uniform", "min": 0.0, "max": 0.99},
}


def _raw_with(prior=None, **sections):
    """_MINIMAL_RAW with extra prior entries and replaced top-level sections."""
    return {
        **_MINIMAL_RAW,
        "prior": {**_MINIMAL_RAW["prior"], **(prior or {})},
        **sections,
    }


def test_j_frame_iota_conflict():
    raw = _raw_with({**_J_FRAME_PRIOR, "iota": {"type": "sine"}})
    with pytest.raises(ValidationError, match="iota"):
        PipelineConfig.model_validate(raw)


def test_j_frame_partial_params_rejected():
    raw = _raw_with({"theta_jn": _J_FRAME_PRIOR["theta_jn"], "phi_jl": _FULL_PERIOD})
    with pytest.raises(ValidationError, match="missing"):
        PipelineConfig.model_validate(raw)


def test_spin_groups_mutually_exclusive():
    raw = _raw_with(
        {
            "s1_z": {"type": "uniform", "min": -0.99, "max": 0.99},
            "s2_z": {"type": "uniform", "min": -0.99, "max": 0.99},
            "s1_mag": {"type": "uniform", "min": 0.0, "max": 0.99},
            "s1_theta": {"type": "sine"},
            "s1_phi": _FULL_PERIOD,
        }
    )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        PipelineConfig.model_validate(raw)


def test_sky_groups_mutually_exclusive():
    raw = _raw_with(
        {
            "ra": _FULL_PERIOD,
            "dec": {"type": "cosine"},
            "azimuth": _FULL_PERIOD,
            "zenith": {"type": "sine"},
        }
    )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        PipelineConfig.model_validate(raw)


def test_time_groups_mutually_exclusive():
    raw = _raw_with(
        {
            "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
            "t_det": {"type": "uniform", "min": -0.1, "max": 0.1},
        }
    )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        PipelineConfig.model_validate(raw)


def test_t_det_requires_detector_frame_time():
    raw = _raw_with(
        {"t_det": {"type": "uniform", "min": -0.1, "max": 0.1}},
        sampling={"detector_frame_time": False},
    )
    with pytest.raises(ValidationError, match="t_det"):
        PipelineConfig.model_validate(raw)


def test_detector_sky_requires_detector_frame_sky():
    raw = _raw_with(
        {"azimuth": _FULL_PERIOD, "zenith": {"type": "sine"}},
        sampling={"detector_frame_sky": False},
    )
    with pytest.raises(ValidationError, match="azimuth"):
        PipelineConfig.model_validate(raw)


# ----------------------------------------------------------------------
# NS AW sampler prior constraints
# ----------------------------------------------------------------------


def test_ns_aw_non_uniform_t_c_rejected():
    with pytest.raises(ValidationError, match="uniform"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "prior": {
                    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
                    "t_c": {"type": "gaussian", "loc": 0.0, "scale": 0.05},
                },
                "sampler": {"type": "blackjax-ns-aw"},
            }
        )


def test_ns_aw_non_uniform_t_det_rejected():
    with pytest.raises(ValidationError, match="uniform"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "prior": {
                    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
                    "t_det": {"type": "gaussian", "loc": 0.0, "scale": 0.05},
                },
                "sampler": {"type": "blackjax-ns-aw"},
            }
        )


_ISOTROPIC_SKY = {"ra": _FULL_PERIOD, "dec": {"type": "cosine"}}


@pytest.mark.parametrize(
    "sky",
    [
        {"ra": {"type": "uniform", "min": 1.0, "max": 2.0}, "dec": {"type": "cosine"}},
        {"ra": _FULL_PERIOD, "dec": {"type": "uniform", "min": -0.5, "max": 0.5}},
    ],
    ids=["restricted_ra", "uniform_dec"],
)
def test_ns_aw_detector_frame_sky_needs_an_isotropic_prior(sky):
    raw = _raw_with(sky, sampler={"type": "blackjax-ns-aw"})
    with pytest.raises(ValidationError, match="isotropic"):
        PipelineConfig.model_validate(raw)
    # Mapping ra/dec directly works for any of the supported priors.
    raw["sampling"] = {"detector_frame_sky": False}
    PipelineConfig.model_validate(raw)


@pytest.mark.parametrize(
    "sky",
    [
        _ISOTROPIC_SKY,
        {
            "azimuth": _FULL_PERIOD,
            "zenith": {"type": "uniform", "min": 0.5, "max": 1.0},
        },
    ],
    ids=["isotropic_ra_dec", "any_azimuth_zenith"],
)
def test_ns_aw_accepts_an_isotropic_or_detector_frame_sky_prior(sky):
    raw = _raw_with(sky, sampler={"type": "blackjax-ns-aw"})
    PipelineConfig.model_validate(raw)


# ----------------------------------------------------------------------
# SwiG sampler config
# ----------------------------------------------------------------------


def test_swig_sampler_parses_from_toml():
    from jimgw.samplers.config import BlackJAXSwiGConfig

    cfg = PipelineConfig.model_validate(
        {
            **_MINIMAL_RAW,
            "sampler": {
                "type": "blackjax-swig",
                "blocks": [["M_c"], ["q"]],
                "n_live": 4,
                "n_delete_frac": 0.5,
            },
        }
    )
    assert isinstance(cfg.sampler, BlackJAXSwiGConfig)
    assert cfg.sampler.blocks == [["M_c"], ["q"]]


def test_swig_sampler_requires_blocks():
    with pytest.raises(ValidationError, match="blocks"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "sampler": {"type": "blackjax-swig"},
            }
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "model, values, field",
    [
        (UniformSpec, {"min": 1.0, "max": 2.0}, "min"),
        (UniformSpec, {"min": 1.0, "max": 2.0}, "max"),
        (GaussianSpec, {"loc": 0.0, "scale": 1.0}, "loc"),
        (GaussianSpec, {"loc": 0.0, "scale": 1.0}, "scale"),
        (PowerLawSpec, {"min": 1.0, "max": 2.0, "alpha": 2.0}, "alpha"),
        (RayleighSpec, {"scale": 1.0}, "scale"),
        (WaveformConfig, {"approximant": "IMRPhenomD"}, "f_ref"),
        (CLIMultibandedConfig, {}, "reference_chirp_mass"),
        (CLIMultibandedConfig, {}, "time_offset"),
        (CLIOptimizerRefParams, {}, "target"),
        (CLIHeterodynedConfig, {}, "epsilon"),
    ],
)
def test_numeric_config_values_must_be_finite(model, values, field, bad):
    with pytest.raises(ValidationError):
        model.model_validate({**values, field: bad})


@pytest.mark.parametrize(
    "bounds", [(0.1, -0.1), (0.0, 0.0), (float("nan"), 0.1), (-0.1, float("inf"))]
)
def test_time_marginalization_requires_finite_increasing_range(bounds):
    with pytest.raises(ValidationError):
        CLITimeMargConfig(tc_range=bounds)


@pytest.mark.parametrize("bounds", [(100.0, 20.0), (20.0, 20.0)])
def test_likelihood_frequency_range_must_increase(bounds):
    with pytest.raises(ValidationError, match="f_min must be below f_max"):
        LikelihoodConfig(f_min=bounds[0], f_max=bounds[1])


def test_likelihood_still_accepts_infinite_upper_frequency_as_nyquist_shorthand():
    config = LikelihoodConfig(f_min=20.0, f_max=float("inf"))
    assert config.f_max == float("inf")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_dictionary_parameter_values_must_be_finite(bad):
    with pytest.raises(ValidationError):
        CLIProvidedRefParams(values={"d_L": bad})
    with pytest.raises(ValidationError):
        LikelihoodConfig(f_min=20.0, f_max=100.0, fixed_parameters={"d_L": bad})
    with pytest.raises(ValidationError):
        InjectionDataConfig(
            detectors=["H1"],
            trigger_time=2.0,
            duration=4.0,
            sampling_frequency=256.0,
            zero_noise=True,
            injection_parameters={"d_L": bad},
        )


@pytest.mark.parametrize("minimum", [0.0, -1.0])
def test_power_law_prior_requires_positive_lower_bound(minimum):
    with pytest.raises(ValidationError, match="min > 0"):
        PowerLawSpec(min=minimum, max=2.0, alpha=2.0)


def test_heterodyne_binning_inputs_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        CLIHeterodynedConfig(n_bins=32, epsilon=0.5)


@pytest.mark.parametrize("bad", [True, 2.0, 2.5])
def test_bin_counts_and_modes_are_actual_integers(bad):
    with pytest.raises(ValidationError):
        CLIHeterodynedConfig(n_bins=bad)
    with pytest.raises(ValidationError):
        CLIMultibandedConfig(highest_mode=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_distance_reference_must_be_finite(bad):
    with pytest.raises(ValidationError):
        CLIDistanceMargConfig.model_validate(
            {
                "distance_prior": {"d_L": {"type": "uniform", "min": 1.0, "max": 2.0}},
                "ref_dist": bad,
            }
        )


def test_distance_prior_must_name_distance_parameter():
    with pytest.raises(ValidationError, match="'d_L'"):
        CLIDistanceMargConfig.model_validate(
            {"distance_prior": {"mass": {"type": "uniform", "min": 1.0, "max": 2.0}}}
        )


@pytest.mark.parametrize(
    "parameter, option",
    [
        ("phase_c", {"phase_marginalization": True}),
        ("t_c", {"time_marginalization": {}}),
        (
            "d_L",
            {
                "distance_marginalization": {
                    "distance_prior": {
                        "d_L": {"type": "uniform", "min": 1.0, "max": 2.0}
                    }
                }
            },
        ),
    ],
)
def test_marginalized_parameters_cannot_be_fixed(parameter, option):
    with pytest.raises(ValidationError, match=f"Cannot have {parameter} fixed"):
        LikelihoodConfig(
            f_min=20.0, f_max=100.0, fixed_parameters={parameter: 1.0}, **option
        )


_DISTANCE_MARGINALIZATION = {
    "distance_prior": {"d_L": {"type": "uniform", "min": 1.0, "max": 2.0}}
}


@pytest.mark.parametrize(
    "options, message",
    [
        (
            {
                "time_marginalization": {},
                "distance_marginalization": _DISTANCE_MARGINALIZATION,
            },
            "Joint time \\+ distance",
        ),
        (
            {"heterodyne": {}, "multiband": {}},
            "heterodyne and multiband cannot both be set",
        ),
        (
            {"heterodyne": {}, "time_marginalization": {}},
            "time_marginalization cannot be used with heterodyne",
        ),
        (
            {"heterodyne": {}, "distance_marginalization": _DISTANCE_MARGINALIZATION},
            "distance_marginalization cannot be used with heterodyne",
        ),
        (
            {"multiband": {}, "time_marginalization": {}},
            "time_marginalization cannot be used with multiband",
        ),
        (
            {"multiband": {}, "distance_marginalization": _DISTANCE_MARGINALIZATION},
            "distance_marginalization cannot be used with multiband",
        ),
        (
            {"multiband": {}, "phase_marginalization": True},
            "phase_marginalization cannot be used with multiband",
        ),
    ],
)
def test_incompatible_likelihood_options_are_refused(options, message):
    with pytest.raises(ValidationError, match=message):
        LikelihoodConfig.model_validate({"f_min": 20.0, "f_max": 100.0, **options})


_ET_NAMES = {"ET": ("ET1", "ET2", "ET3"), "ET_2L": ("ET_Sar", "ET_EMR")}


def _et_file_config(detector="ET", **fields):
    names = _ET_NAMES[detector]
    return {
        "detectors": [detector],
        "trigger_time": 2.0,
        "strain_files": {name: f"{name}.npz" for name in names},
        "psd_files": {name: f"{name}_psd.txt" for name in names},
        **fields,
    }


@pytest.mark.parametrize("detector", ["ET", "ET_2L"])
def test_et_file_data_uses_the_individual_detector_names(detector):
    names = set(_ET_NAMES[detector])
    config = FileDataConfig.model_validate(
        _et_file_config(
            detector, strain_channels={name: f"{name}:STRAIN" for name in names}
        )
    )
    assert set(config.strain_files) == names
    assert set(config.strain_channels) == names
    assert config.psd_files is not None
    assert set(config.psd_files) == names


@pytest.mark.parametrize(
    "detector, fields, message",
    [
        (
            "ET",
            {"strain_files": {"ET": "ET.npz"}},
            r"strain_files missing for: \['ET1'",
        ),
        ("ET", {"strain_files": {"ET1": "ET1.npz", "ET2": "ET2.npz"}}, "ET3"),
        ("ET", {"strain_channels": {"ET": "ET:STRAIN"}}, "unknown detector names"),
        ("ET", {"strain_channels": {"H1": "H1:STRAIN"}}, "unknown detector names"),
        ("ET", {"psd_files": {"ET": "ET_psd.txt"}}, r"psd_files missing for: \['ET1'"),
        (
            "ET_2L",
            {"psd_files": {"ET_2L": "ET_2L_psd.txt"}},
            r"psd_files missing for: \['ET_Sar', 'ET_EMR'\]",
        ),
    ],
)
def test_et_file_data_rejects_ambiguous_or_missing_sources(detector, fields, message):
    with pytest.raises(ValidationError, match=message):
        FileDataConfig.model_validate(_et_file_config(detector, **fields))


@pytest.mark.parametrize(
    "detectors, accepted", [(["H1"], False), (["ET"], True), (["ET_2L"], True)]
)
def test_detector_frame_sky_counts_the_detectors_of_each_preset(detectors, accepted):
    raw = _raw_with(
        {"ra": _FULL_PERIOD, "dec": {"type": "cosine"}},
        data=_injection_data(detectors),
    )
    if accepted:
        PipelineConfig.model_validate(raw)
    else:
        with pytest.raises(ValidationError, match="at least 2 detectors"):
            PipelineConfig.model_validate(raw)


@pytest.mark.parametrize("detector", ["ET", "ET_2L", "CE"])
def test_gwosc_config_rejects_presets_without_open_strain(detector):
    with pytest.raises(ValidationError, match="GWOSC strain is not available"):
        GWOSCDataConfig(
            detectors=[detector], trigger_time=2.0, duration=4.0, psd_duration=8.0
        )


@pytest.mark.parametrize("field", ["popsize", "n_steps"])
@pytest.mark.parametrize("bad", [True, 2.0, 2.5])
def test_reference_optimizer_counts_are_actual_integers(field, bad):
    with pytest.raises(ValidationError):
        CLIOptimizerRefParams.model_validate({field: bad})


@pytest.mark.parametrize("field", ["psd_files", "asd_files"])
def test_psd_source_rejects_unknown_detector_mapping_keys(field):
    fields = {
        "psd_files": None,
        field: {
            **{name: f"{name}_psd.txt" for name in _ET_NAMES["ET"]},
            "ET": "ET_psd.txt",
        },
    }
    with pytest.raises(ValidationError, match=f"{field} has unknown detector names"):
        FileDataConfig.model_validate(_et_file_config(**fields))
