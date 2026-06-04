"""Tests for the BlackJAX import guards in _imports.py."""

from __future__ import annotations

import types

import pytest

from jimgw.samplers.blackjax._imports import (
    require_nested_sampling,
    require_nss,
)

# ---------------------------------------------------------------------------
# require_nested_sampling
# ---------------------------------------------------------------------------


def test_require_nested_sampling_ok():
    fake = types.SimpleNamespace(ns=object())
    require_nested_sampling(fake)  # must not raise


def test_require_nested_sampling_missing():
    fake = types.SimpleNamespace()  # no `ns`
    with pytest.raises(ImportError, match=r"blackjax\.ns"):
        require_nested_sampling(fake)


# ---------------------------------------------------------------------------
# require_nss
# ---------------------------------------------------------------------------


def test_require_nss_ok():
    fake = types.SimpleNamespace(nss=object())
    require_nss(fake)  # must not raise


def test_require_nss_missing():
    fake = types.SimpleNamespace()  # no `nss`
    with pytest.raises(ImportError, match=r"blackjax\.nss"):
        require_nss(fake)
