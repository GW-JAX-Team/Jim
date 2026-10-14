"""Unit tests for CLI config schema (TOML round-trips, validation, prior parsing)."""

import logging
import tomllib
import warnings
from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

from jimgw.cli._config import (
    CLIHeterodynedConfig,
    CLIInjectionRefParams,
    CLIOptimizerRefParams,
    CLIProvidedRefParams,
    CosineSpec,
    FileDataConfig,
    GaussianSpec,
    GWOSCDataConfig,
    InjectionDataConfig,
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
    assert cfg.sampling.time_frame == "detector"
    assert cfg.sampling.sky_frame == "detector"


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


# ---------------------------------------------------------------------------
# PSD / ASD sources (file and injection data)
# ---------------------------------------------------------------------------

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


def test_injection_without_files_uses_built_in_default_for_known_detectors():
    cfg = InjectionDataConfig.model_validate(_injection_data(("H1", "L1", "V1")))
    assert cfg.psd_files is None
    assert cfg.asd_files is None


@pytest.mark.parametrize("detector", ["CE", "ET"])
def test_injection_without_files_rejects_detectors_without_default(detector):
    # Caught at config time, instead of failing mid-run with no way to fix it.
    with pytest.raises(
        ValidationError, match=rf"No built-in PSD for detector\(s\) \['{detector}'\]"
    ):
        InjectionDataConfig.model_validate(_injection_data((detector,)))


def test_injection_accepts_files_for_detectors_without_default():
    cfg = InjectionDataConfig.model_validate(
        _injection_data(("CE",), psd_files={"CE": "ce_psd.txt"})
    )
    assert cfg.psd_files == {"CE": Path("ce_psd.txt")}


def test_injection_files_must_cover_detectors_that_have_a_default_too():
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


# ---------------------------------------------------------------------------
# Injected-noise seed
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Prior spec validators
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Detector list validators
# ---------------------------------------------------------------------------


def test_duplicate_detectors_rejected():
    with pytest.raises(ValidationError, match="Duplicate"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "data": {**_MINIMAL_RAW["data"], "detectors": ["H1", "H1"]},
            }
        )


# ---------------------------------------------------------------------------
# Sky parametrization completeness
# ---------------------------------------------------------------------------


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
                "sampling": {"sky_frame": "detector"},
            }
        )


# ---------------------------------------------------------------------------
# NS AW sampler prior constraints
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SwiG sampler config
# ---------------------------------------------------------------------------


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
