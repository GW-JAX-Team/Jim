"""Unit tests for CLI likelihood construction."""

from types import SimpleNamespace

import pytest

from jimgw.cli import _likelihood
from jimgw.cli._config import CLIMultibandedConfig, LikelihoodConfig, PriorConfig
from jimgw.core.constants import EARTH_RADIUS_LIGHT_S
from jimgw.core.prior import CombinePrior


def _ifos(*durations: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            name=name,
            data=SimpleNamespace(start_time=100.0, duration=duration),
        )
        for name, duration in zip(("H1", "L1"), durations, strict=True)
    ]


def _build_multiband(
    monkeypatch,
    prior_config: PriorConfig,
    *,
    multiband_kwargs: dict | None = None,
    ifos: list[SimpleNamespace] | None = None,
    time_frame: str = "detector",
) -> dict:
    """Build via a constructor stub and return the forwarded settings."""
    captured_kwargs = {}

    class DummyMultibandedLikelihood:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.reference_chirp_mass = kwargs["reference_chirp_mass"]

    monkeypatch.setattr(
        _likelihood, "MultibandedTransientLikelihoodFD", DummyMultibandedLikelihood
    )
    cfg = LikelihoodConfig(
        f_min=20.0,
        f_max=1024.0,
        multiband=CLIMultibandedConfig(**(multiband_kwargs or {})),
    )
    _likelihood.build_likelihood(
        cfg=cfg,
        ifos=ifos or _ifos(4.0, 3.0),
        waveform=None,
        trigger_time=100.0,
        waveform_f_ref=20.0,
        time_frame=time_frame,
        prior=CombinePrior([]),
        prior_config=prior_config,
        likelihood_transforms=[],
        data_cfg=None,
    )
    return captured_kwargs


@pytest.mark.parametrize(
    "mc_spec",
    [
        {"type": "uniform", "min": 15.0, "max": 30.0},
        {"type": "power_law", "min": 15.0, "max": 30.0, "alpha": 1.0},
    ],
)
def test_multiband_infers_settings_from_bounded_cli_prior(monkeypatch, mc_spec):
    prior_config = PriorConfig.model_validate(
        {
            "M_c": mc_spec,
            "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
        }
    )

    captured = _build_multiband(monkeypatch, prior_config)

    t_end = 3.0
    assert captured["reference_chirp_mass"] == 15.0
    assert captured["time_offset"] == pytest.approx(
        t_end - (-0.1) + EARTH_RADIUS_LIGHT_S
    )
    assert captured["delta_f_end"] == pytest.approx(
        100.0 / (t_end - 0.1 - EARTH_RADIUS_LIGHT_S)
    )


@pytest.mark.parametrize(
    ("multiband_kwargs", "expected_time_offset", "expected_delta_f_end"),
    [
        ({"time_offset": 1.5}, 1.5, 100.0 / (3.0 - 0.1 - EARTH_RADIUS_LIGHT_S)),
        ({"delta_f_end": 75.0}, 3.0 - (-0.1) + EARTH_RADIUS_LIGHT_S, 75.0),
    ],
)
def test_multiband_explicit_time_settings_override_independently(
    monkeypatch, multiband_kwargs, expected_time_offset, expected_delta_f_end
):
    prior_config = PriorConfig.model_validate(
        {
            "M_c": {"type": "uniform", "min": 15.0, "max": 30.0},
            "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
        }
    )

    captured = _build_multiband(
        monkeypatch, prior_config, multiband_kwargs=multiband_kwargs
    )

    assert captured["time_offset"] == pytest.approx(expected_time_offset)
    assert captured["delta_f_end"] == pytest.approx(expected_delta_f_end)


def test_multiband_explicit_reference_chirp_mass_overrides_prior(monkeypatch):
    prior_config = PriorConfig.model_validate(
        {"M_c": {"type": "uniform", "min": 15.0, "max": 30.0}}
    )

    captured = _build_multiband(
        monkeypatch,
        prior_config,
        multiband_kwargs={"reference_chirp_mass": 20.0},
    )

    assert captured["reference_chirp_mass"] == 20.0
    assert captured["time_offset"] == 2.12
    assert captured["delta_f_end"] == 53.0


def test_multiband_accepts_explicit_reference_without_mass_prior(monkeypatch):
    captured = _build_multiband(
        monkeypatch,
        PriorConfig.model_validate({}),
        multiband_kwargs={"reference_chirp_mass": 20.0},
    )

    assert captured["reference_chirp_mass"] == 20.0


@pytest.mark.parametrize(
    "prior_dict",
    [
        {},
        {"M_c": {"type": "gaussian", "loc": 20.0, "scale": 1.0}},
    ],
)
def test_multiband_requires_explicit_reference_for_unbounded_mass(
    monkeypatch, prior_dict
):
    prior_config = PriorConfig.model_validate(prior_dict)

    with pytest.raises(ValueError, match="reference_chirp_mass"):
        _build_multiband(monkeypatch, prior_config)


def test_multiband_uses_timing_defaults_without_time_bounds(monkeypatch):
    prior_config = PriorConfig.model_validate(
        {"M_c": {"type": "uniform", "min": 15.0, "max": 30.0}}
    )

    captured = _build_multiband(monkeypatch, prior_config)

    assert captured["time_offset"] == 2.12
    assert captured["delta_f_end"] == 53.0


def test_multiband_preserves_detector_time_inference_convention(monkeypatch):
    prior_config = PriorConfig.model_validate(
        {
            "M_c": {"type": "uniform", "min": 15.0, "max": 30.0},
            "t_det": {"type": "uniform", "min": -0.2, "max": 0.1},
        }
    )

    captured = _build_multiband(
        monkeypatch,
        prior_config,
        ifos=_ifos(4.0, 7.0),
        time_frame="L1",
    )

    t_end = 7.0
    assert captured["time_offset"] == pytest.approx(
        t_end - (-0.2) + EARTH_RADIUS_LIGHT_S
    )
    assert captured["delta_f_end"] == pytest.approx(
        100.0 / (t_end - 0.1 - EARTH_RADIUS_LIGHT_S)
    )
