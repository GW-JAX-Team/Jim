"""Integration test: full jim-run pipeline from TOML config."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from typer.testing import CliRunner

from jimgw.cli import app

pytestmark = pytest.mark.integration

_CONFIG = Path(__file__).resolve().parent.parent / "fixtures" / "GW150914_test.toml"
_RUNNER = CliRunner()


@pytest.fixture(scope="module")
def tmp_output(tmp_path_factory):
    """Shared output directory for the module — created once, reused by all tests."""
    return tmp_path_factory.mktemp("output")


@pytest.fixture(scope="module")
def patched_config(tmp_output, tmp_path_factory):
    """Copy the test config and patch output.dir to tmp_output — created once per module."""
    with open(_CONFIG, "rb") as f:
        raw = tomllib.load(f)
    raw["output"]["dir"] = str(tmp_output)
    raw["output"]["overwrite"] = True
    cfg_path = tmp_path_factory.mktemp("cfg") / "test_cfg.toml"
    with open(cfg_path, "wb") as f:
        tomli_w.dump(raw, f)
    return cfg_path


@pytest.fixture(scope="module")
def full_run_result(patched_config):
    """Run the CLI once per module and share the result across all tests."""
    result = _RUNNER.invoke(app, [str(patched_config)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


# ---------------------------------------------------------------------------
# Full run + output files
# ---------------------------------------------------------------------------


def test_full_run_output_files(full_run_result, tmp_output):
    assert (tmp_output / "samples.npz").exists()
    assert (tmp_output / "config.final.toml").exists()


def test_full_run_samples_shape(full_run_result, tmp_output):
    with np.load(tmp_output / "samples.npz") as data:
        expected_params = {
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
            "log_likelihood",
        }
        assert set(data.files) == expected_params
        n = data["M_c"].shape[0]
        assert n > 0
        for key in expected_params:
            assert data[key].shape == (n,)


def test_init_creates_template(tmp_path):
    out = tmp_path / "template.toml"
    result = _RUNNER.invoke(app, ["--init", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    with open(out, "rb") as f:
        raw = tomllib.load(f)
    assert "data" in raw
    assert "prior" in raw
    assert "likelihood" in raw


def test_init_refuses_overwrite(tmp_path):
    out = tmp_path / "template.toml"
    out.write_text("existing")
    result = _RUNNER.invoke(app, ["--init", str(out)])
    assert result.exit_code == 2


def test_missing_config_exits_with_code_2(tmp_path):
    result = _RUNNER.invoke(app, [str(tmp_path / "nonexistent.toml")])
    assert result.exit_code == 2


def test_no_args_exits_with_code_2():
    result = _RUNNER.invoke(app, [])
    assert result.exit_code == 2


def test_overwrite_false_raises(patched_config, full_run_result, tmp_output):
    # Second run with overwrite=false should fail
    with open(patched_config, "rb") as f:
        raw = tomllib.load(f)
    raw["output"]["overwrite"] = False
    cfg2 = patched_config.parent / "no_overwrite.toml"
    with open(cfg2, "wb") as f:
        tomli_w.dump(raw, f)
    result2 = _RUNNER.invoke(app, [str(cfg2)])
    assert result2.exit_code == 2
