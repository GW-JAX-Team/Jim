"""Tests for CLI data building: where each detector's PSD comes from.

Injections run with ``zero_noise=True`` so these tests need neither the network nor a
noise seed; they check *which* PSD ends up on the detector, which is what both the
injected noise and the likelihood read.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jimgw.cli._config import FileDataConfig, InjectionDataConfig, WaveformConfig
from jimgw.cli._data import build_data
from jimgw.cli._utils import DEFAULT_ASD_DETECTORS
from jimgw.cli._waveform import build_waveform
from jimgw.core.single_event.detector import asd_file_dict

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
F_MIN, F_MAX = 20.0, 1024.0
INJECTION_PARAMETERS = {
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


@pytest.fixture(scope="module")
def waveform():
    return build_waveform(WaveformConfig(approximant="IMRPhenomD", f_ref=20.0))


def _injection_cfg(detectors, **tables):
    return InjectionDataConfig.model_validate(
        {
            "type": "injection",
            "detectors": list(detectors),
            "trigger_time": 1126259462.4,
            "duration": 4.0,
            "sampling_frequency": 2048.0,
            "injection_parameters": INJECTION_PARAMETERS,
            "zero_noise": True,
            **tables,
        }
    )


def _fixture_psd(name):
    with np.load(FIXTURES / f"GW150914_psd_{name}.npz") as f:
        return np.asarray(f["frequencies"]), np.asarray(f["values"])


def _no_network(*args, **kwargs):
    raise AssertionError("the built-in ASD was fetched although a PSD file was set")


def test_default_asd_detectors_match_detector_module():
    # The JAX-free copy used by config validation must not drift from the real list.
    assert DEFAULT_ASD_DETECTORS == frozenset(asd_file_dict)


def test_injection_uses_psd_files_and_never_touches_the_network(waveform, monkeypatch):
    monkeypatch.setattr("requests.get", _no_network)
    cfg = _injection_cfg(
        ("H1", "L1"),
        psd_files={
            "H1": FIXTURES / "GW150914_psd_H1.npz",
            "L1": FIXTURES / "GW150914_psd_L1.npz",
        },
    )

    ifos = build_data(cfg, f_min=F_MIN, f_max=F_MAX, waveform=waveform)

    assert [ifo.name for ifo in ifos] == ["H1", "L1"]
    for ifo in ifos:
        freqs, values = _fixture_psd(ifo.name)
        expected = np.interp(np.asarray(ifo.sliced_frequencies), freqs, values)
        np.testing.assert_allclose(np.asarray(ifo.sliced_psd), expected, rtol=1e-12)


def test_injection_squares_asd_files(tmp_path, waveform, monkeypatch):
    monkeypatch.setattr("requests.get", _no_network)
    freqs = np.arange(1.0, 2048.0 + 0.25, 0.25)
    asd = 1e-23 * (1.0 + (30.0 / freqs) ** 2)
    asd_file = tmp_path / "h1_asd.txt"
    np.savetxt(asd_file, np.column_stack([freqs, asd]), fmt="%.17e")

    (ifo,) = build_data(
        _injection_cfg(("H1",), asd_files={"H1": asd_file}),
        f_min=F_MIN,
        f_max=F_MAX,
        waveform=waveform,
    )

    expected = np.interp(np.asarray(ifo.sliced_frequencies), freqs, asd**2)
    np.testing.assert_allclose(np.asarray(ifo.sliced_psd), expected, rtol=1e-12)


def test_injection_squares_npz_asd_files(tmp_path, waveform, monkeypatch):
    # An .npz archive listed under asd_files is read as an ASD, like any other format.
    monkeypatch.setattr("requests.get", _no_network)
    freqs, values = _fixture_psd("H1")
    asd_file = tmp_path / "h1_asd.npz"
    np.savez(asd_file, values=np.sqrt(values), frequencies=freqs)

    (ifo,) = build_data(
        _injection_cfg(("H1",), asd_files={"H1": asd_file}),
        f_min=F_MIN,
        f_max=F_MAX,
        waveform=waveform,
    )

    expected = np.interp(np.asarray(ifo.sliced_frequencies), freqs, values)
    np.testing.assert_allclose(np.asarray(ifo.sliced_psd), expected, rtol=1e-12)


def test_injection_without_files_falls_back_to_built_in_asd(waveform, monkeypatch):
    fetched = []

    def fake_get(url, timeout=None):
        fetched.append(url)
        freqs = np.arange(1.0, 4096.0 + 0.25, 0.25)
        text = "\n".join(f"{f} 1e-23" for f in freqs)
        return SimpleNamespace(content=text.encode(), raise_for_status=lambda: None)

    monkeypatch.setattr("requests.get", fake_get)

    ifos = build_data(
        _injection_cfg(("H1", "L1")), f_min=F_MIN, f_max=F_MAX, waveform=waveform
    )

    assert fetched == [asd_file_dict["H1"], asd_file_dict["L1"]]
    for ifo in ifos:
        np.testing.assert_allclose(np.asarray(ifo.sliced_psd), 1e-46, rtol=1e-12)


@pytest.mark.parametrize("detector, n_interferometers", [("CE", 1), ("ET", 3)])
def test_injection_table_is_keyed_by_detector_name(
    detector, n_interferometers, waveform, monkeypatch
):
    # CE and ET have no built-in PSD, so a file is the only way to inject.  One "ET"
    # entry covers all three ET interferometers (ET1-ET3).
    monkeypatch.setattr("requests.get", _no_network)
    cfg = _injection_cfg(
        (detector,), psd_files={detector: FIXTURES / "GW150914_psd_H1.npz"}
    )

    ifos = build_data(cfg, f_min=F_MIN, f_max=F_MAX, waveform=waveform)

    assert len(ifos) == n_interferometers
    freqs, values = _fixture_psd("H1")
    for ifo in ifos:
        expected = np.interp(np.asarray(ifo.sliced_frequencies), freqs, values)
        np.testing.assert_allclose(np.asarray(ifo.sliced_psd), expected, rtol=1e-12)


def test_file_data_asd_files_give_the_same_psd_as_psd_files(tmp_path):
    freqs, values = _fixture_psd("H1")
    asd_file = tmp_path / "h1_asd.txt"
    np.savetxt(asd_file, np.column_stack([freqs, np.sqrt(values)]), fmt="%.17e")
    base = {
        "type": "file",
        "detectors": ["H1"],
        "trigger_time": 1126259462.4,
        "strain_files": {"H1": FIXTURES / "GW150914_strain_H1.npz"},
    }

    (from_psd,) = build_data(
        FileDataConfig.model_validate(
            {**base, "psd_files": {"H1": FIXTURES / "GW150914_psd_H1.npz"}}
        ),
        f_min=F_MIN,
        f_max=F_MAX,
    )
    (from_asd,) = build_data(
        FileDataConfig.model_validate({**base, "asd_files": {"H1": asd_file}}),
        f_min=F_MIN,
        f_max=F_MAX,
    )

    np.testing.assert_allclose(
        np.asarray(from_asd.psd.values), np.asarray(from_psd.psd.values), rtol=1e-12
    )
