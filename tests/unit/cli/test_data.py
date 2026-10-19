"""Tests for CLI data building: where each detector's PSD comes from.

Injections run with ``zero_noise=True`` so these tests need neither the network nor a
noise seed; they check *which* PSD ends up on the detector, which is what both the
injected noise and the likelihood read.
"""

import time
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from jimgw.cli._config import (
    FileDataConfig,
    GWOSCDataConfig,
    InjectionDataConfig,
    WaveformConfig,
)
from jimgw.cli._data import build_data
from jimgw.cli._utils import PRESET_DETECTOR_NAMES
from jimgw.cli._waveform import build_waveform
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import (
    GroundBased2G,
    default_asd_files,
    get_detector_preset,
)
from tests.utils import sliced_fd_data, sliced_frequencies, sliced_psd

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


DURATION = 4.0


def _injection_cfg(detectors, **fields):
    """An injection config, zero-noise unless *fields* say otherwise."""
    return InjectionDataConfig.model_validate(
        {
            "type": "injection",
            "detectors": list(detectors),
            "trigger_time": 1126259462.4,
            "duration": DURATION,
            "sampling_frequency": 2048.0,
            "injection_parameters": INJECTION_PARAMETERS,
            "zero_noise": True,
            **fields,
        }
    )


def _build_h1_l1(waveform, **fields):
    """Build H1 and L1 from their fixture PSDs (offline)."""
    psd_files = {d: FIXTURES / f"GW150914_psd_{d}.npz" for d in ("H1", "L1")}
    return build_data(
        _injection_cfg(("H1", "L1"), psd_files=psd_files, **fields),
        f_min=F_MIN,
        f_max=F_MAX,
        waveform=waveform,
    )


def _fixture_psd(name):
    with np.load(FIXTURES / f"GW150914_psd_{name}.npz") as f:
        return np.asarray(f["frequencies"]), np.asarray(f["values"])


def _no_built_in(*args, **kwargs):
    raise AssertionError("the built-in ASD was loaded although a PSD file was set")


@pytest.mark.slow
def test_injection_uses_psd_files_and_not_the_built_in_default(waveform, monkeypatch):
    monkeypatch.setattr(GroundBased2G, "load_and_set_psd", _no_built_in)
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
        expected = np.interp(np.asarray(sliced_frequencies(ifo)), freqs, values)
        np.testing.assert_allclose(np.asarray(sliced_psd(ifo)), expected, rtol=1e-12)


def test_injection_squares_asd_files(tmp_path, waveform, monkeypatch):
    monkeypatch.setattr(GroundBased2G, "load_and_set_psd", _no_built_in)
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

    expected = np.interp(np.asarray(sliced_frequencies(ifo)), freqs, asd**2)
    np.testing.assert_allclose(np.asarray(sliced_psd(ifo)), expected, rtol=1e-12)


def test_injection_noise_is_fixed_by_noise_seed(waveform):
    first = _build_h1_l1(waveform, zero_noise=False, noise_seed=7)
    time.sleep(1.1)  # the old wall-clock seed changed every second
    second = _build_h1_l1(waveform, zero_noise=False, noise_seed=7)
    other = _build_h1_l1(waveform, zero_noise=False, noise_seed=8)

    for a, b, c in zip(first, second, other):
        np.testing.assert_array_equal(sliced_fd_data(a), sliced_fd_data(b))
        assert np.max(np.abs(np.asarray(sliced_fd_data(a) - sliced_fd_data(c)))) > 0


def test_injected_noise_follows_each_detectors_psd_and_is_independent(waveform):
    # Noise = noisy data minus the zero-noise data.  Whitened by the PSD the
    # likelihood will use, it has unit variance in its real and imaginary parts, so
    # noise and analysis share one PSD, and the two detectors do not share draws.
    clean = _build_h1_l1(waveform)
    noisy = _build_h1_l1(waveform, zero_noise=False, noise_seed=7)

    whitened = []
    for c, n in zip(clean, noisy):
        noise = np.asarray(sliced_fd_data(n) - sliced_fd_data(c))
        w = noise / np.sqrt(np.asarray(sliced_psd(n)) * DURATION / 4)
        assert np.var(w.real) == pytest.approx(1.0, abs=0.1)
        assert np.var(w.imag) == pytest.approx(1.0, abs=0.1)
        whitened.append(w)

    h1, l1 = whitened
    correlation = np.abs(np.vdot(h1, l1)) / (np.linalg.norm(h1) * np.linalg.norm(l1))
    assert correlation < 0.1


def test_injection_squares_npz_asd_files(tmp_path, waveform, monkeypatch):
    # An .npz archive listed under asd_files is read as an ASD, like any other format.
    monkeypatch.setattr(GroundBased2G, "load_and_set_psd", _no_built_in)
    freqs, values = _fixture_psd("H1")
    asd_file = tmp_path / "h1_asd.npz"
    np.savez(asd_file, values=np.sqrt(values), frequencies=freqs)

    (ifo,) = build_data(
        _injection_cfg(("H1",), asd_files={"H1": asd_file}),
        f_min=F_MIN,
        f_max=F_MAX,
        waveform=waveform,
    )

    expected = np.interp(np.asarray(sliced_frequencies(ifo)), freqs, values)
    np.testing.assert_allclose(np.asarray(sliced_psd(ifo)), expected, rtol=1e-12)


def test_cli_detector_table_matches_the_presets():
    assert PRESET_DETECTOR_NAMES == {
        name: tuple(ifo.name for ifo in (value if isinstance(value, list) else [value]))
        for name, value in get_detector_preset().items()
    }


@pytest.mark.parametrize("detector", ["H1", "L1", "V1", "ET", "ET_2L", "CE"])
def test_injection_without_files_uses_the_built_in_design_asd(detector, waveform):
    ifos = build_data(
        _injection_cfg((detector,)), f_min=F_MIN, f_max=F_MAX, waveform=waveform
    )

    for ifo in ifos:
        curve = resources.files("jimgw.core.single_event") / "noise_curves"
        frequencies, asd = np.genfromtxt(
            str(curve / default_asd_files[ifo.name]), unpack=True
        )
        expected = np.interp(np.asarray(sliced_frequencies(ifo)), frequencies, asd**2)
        np.testing.assert_allclose(np.asarray(sliced_psd(ifo)), expected, rtol=1e-12)


@pytest.mark.parametrize(
    "detector, names",
    [("CE", ("CE",)), ("ET", ("ET1", "ET2", "ET3")), ("ET_2L", ("ET_Sar", "ET_EMR"))],
)
def test_injection_table_gives_each_detector_its_own_entry(
    detector, names, waveform, monkeypatch
):
    # The fixture PSDs alternate, so a detector reading another one's entry fails.
    monkeypatch.setattr(GroundBased2G, "load_and_set_psd", _no_built_in)
    fixture = {name: ("H1", "L1")[index % 2] for index, name in enumerate(names)}
    cfg = _injection_cfg(
        (detector,),
        psd_files={
            name: FIXTURES / f"GW150914_psd_{fixture[name]}.npz" for name in names
        },
    )

    ifos = build_data(cfg, f_min=F_MIN, f_max=F_MAX, waveform=waveform)

    assert [ifo.name for ifo in ifos] == list(names)
    for ifo in ifos:
        freqs, values = _fixture_psd(fixture[ifo.name])
        expected = np.interp(np.asarray(sliced_frequencies(ifo)), freqs, values)
        np.testing.assert_allclose(np.asarray(sliced_psd(ifo)), expected, rtol=1e-12)


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


def test_injected_data_carry_the_trigger_time(waveform):
    ifos = _build_h1_l1(waveform)

    assert [ifo.data.trigger_time for ifo in ifos] == [1126259462.4] * 2


def test_file_data_carry_the_trigger_time():
    (ifo,) = build_data(
        FileDataConfig.model_validate(
            {
                "type": "file",
                "detectors": ["H1"],
                "trigger_time": 1126259462.4,
                "strain_files": {"H1": FIXTURES / "GW150914_strain_H1.npz"},
                "psd_files": {"H1": FIXTURES / "GW150914_psd_H1.npz"},
            }
        ),
        f_min=F_MIN,
        f_max=F_MAX,
    )

    assert ifo.data.trigger_time == 1126259462.4
    with np.load(FIXTURES / "GW150914_strain_H1.npz") as saved:
        assert ifo.data.start_time == float(saved["start_time"])
    assert ifo.data.trigger_time != ifo.data.start_time + ifo.data.duration - 2.0


def test_file_data_load_each_et_detectors_own_strain(tmp_path, monkeypatch):
    monkeypatch.setattr(GroundBased2G, "load_and_set_psd", _no_built_in)
    trigger = 1126259462.4
    strain_files = {}
    for index, name in enumerate(("ET1", "ET2", "ET3"), start=1):
        path = tmp_path / f"{name}.npz"
        Data.from_td(
            np.full(8192, index * 1e-22),
            delta_t=1.0 / 2048.0,
            start_time=trigger - 2.0,
            trigger_time=trigger,
        ).to_file(str(path))
        strain_files[name] = path

    ifos = build_data(
        FileDataConfig.model_validate(
            {
                "type": "file",
                "detectors": ["ET"],
                "trigger_time": trigger,
                "strain_files": strain_files,
                "psd_files": {
                    name: FIXTURES / "GW150914_psd_H1.npz" for name in strain_files
                },
            }
        ),
        f_min=F_MIN,
        f_max=F_MAX,
    )

    assert [ifo.name for ifo in ifos] == ["ET1", "ET2", "ET3"]
    for index, ifo in enumerate(ifos, start=1):
        np.testing.assert_array_equal(ifo.data.td, np.full(8192, index * 1e-22))
        assert ifo.data.trigger_time == trigger
    for ifo in ifos[1:]:
        np.testing.assert_array_equal(ifo.psd.values, ifos[0].psd.values)


@pytest.mark.parametrize("post_trigger_duration", [1.0, 2.0, 3.0])
def test_gwosc_data_carry_the_trigger_time(monkeypatch, post_trigger_duration):
    def fake_from_gwosc(ifo, start, end, *, trigger_time=None, **kwargs):
        # GWOSC crops on the sample grid, which need not contain the exact GPS trigger.
        rate = 4096.0
        actual_start = np.floor(start * rate) / rate
        strain = np.random.default_rng(0).normal(size=round((end - start) * rate))
        return Data.from_td(
            strain, 1.0 / rate, start_time=actual_start, trigger_time=trigger_time
        )

    monkeypatch.setattr(Data, "from_gwosc", staticmethod(fake_from_gwosc))

    ifos = build_data(
        GWOSCDataConfig.model_validate(
            {
                "type": "gwosc",
                "detectors": ["H1", "L1"],
                "trigger_time": 1126259462.4,
                "duration": 4.0,
                "post_trigger_duration": post_trigger_duration,
                "psd_duration": 8.0,
            }
        ),
        f_min=F_MIN,
        f_max=F_MAX,
    )

    assert [ifo.data.trigger_time for ifo in ifos] == [1126259462.4] * 2
    requested_start = 1126259462.4 + post_trigger_duration - 4.0
    for ifo in ifos:
        assert ifo.data.start_time == np.floor(requested_start * 4096.0) / 4096.0
        assert ifo.data.start_time != requested_start
        assert ifo.data.trigger_time != ifo.data.start_time + ifo.data.duration - 2.0
