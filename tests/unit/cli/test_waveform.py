"""Tests for CLI waveform construction.

The waveforms themselves are covered in ``tests/unit/core/single_event/test_waveform.py``
(and in ripple).  What is CLI-specific is that ``build_waveform`` keeps ``_REGISTRY``
and the ``Approximant`` config Literal in sync by hand, forwards ``f_ref`` from the
config, and special-cases the burst waveform -- that is all this checks.
"""

from typing import get_args

import pytest

from jimgw.cli._config import Approximant, WaveformConfig
from jimgw.cli._waveform import build_waveform


@pytest.mark.parametrize("approximant", get_args(Approximant))
def test_build_waveform_resolves_every_approximant(approximant):
    """Every ``Approximant`` value builds, and ``f_ref`` is forwarded from the config."""
    waveform = build_waveform(WaveformConfig(approximant=approximant, f_ref=17.0))

    assert callable(waveform)
    if approximant == "SineGaussian":
        # burst waveform: constructed with no f_ref (see build_waveform special case)
        assert not hasattr(waveform, "f_ref")
    else:
        assert waveform.f_ref == 17.0
