"""Lazy import of BlackJAX with feature validation.

NS-AW and NSS rely on nested-sampling submodules not yet in upstream PyPI.
Install them via the ``nested-sampling`` dependency group:

    uv sync --group nested-sampling
"""

from __future__ import annotations

_INSTALL_MSG = (
    "The BlackJAX nested-sampling submodules are required for this sampler "
    "but are not available in the installed version.  Install them with:\n"
    "    uv sync --group nested-sampling\n"
    "See docs/installation.md for details."
)


def require_nested_sampling(bjx) -> None:
    """Check that the installed BlackJAX has the ``ns`` nested-sampling submodule."""
    if not hasattr(bjx, "ns"):
        raise ImportError(
            "Installed BlackJAX is missing the `blackjax.ns` nested-sampling "
            "submodule.  " + _INSTALL_MSG
        )


def require_nss(bjx) -> None:
    """Check that the installed BlackJAX exposes top-level ``blackjax.nss``."""
    if not hasattr(bjx, "nss"):
        raise ImportError(
            "Installed BlackJAX is missing top-level `blackjax.nss`.  " + _INSTALL_MSG
        )
