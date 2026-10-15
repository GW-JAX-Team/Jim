"""Tests for CLI waveform construction.

The waveforms themselves are covered in ``tests/unit/core/single_event/test_waveform.py``
(and in ripple).  What is CLI-specific is that ``build_waveform`` keeps ``_REGISTRY``
and the ``Approximant`` config Literal in sync by hand and forwards ``f_ref`` from the
config -- that is all this checks.
"""

from typing import get_args

import pytest
from pydantic import ValidationError

from jimgw.cli._config import Approximant, WaveformConfig
from jimgw.cli._waveform import _REGISTRY, build_waveform


@pytest.mark.parametrize("approximant", get_args(Approximant))
def test_build_waveform_resolves_every_approximant(approximant):
    """Every ``Approximant`` value builds, and ``f_ref`` is forwarded from the config."""
    waveform = build_waveform(WaveformConfig(approximant=approximant, f_ref=17.0))

    assert callable(waveform)
    assert waveform.f_ref == 17.0


def test_the_config_and_the_builder_offer_the_same_approximants():
    """A model added to one list but not the other is caught here."""
    assert set(get_args(Approximant)) == set(_REGISTRY)


def test_a_time_domain_burst_is_not_an_approximant():
    """The likelihoods are frequency-domain, so the config must not offer SineGaussian."""
    with pytest.raises(ValidationError, match="approximant"):
        WaveformConfig(approximant="SineGaussian")
