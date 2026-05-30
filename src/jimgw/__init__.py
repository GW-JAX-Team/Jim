import logging

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jimgw")
except PackageNotFoundError:
    __version__ = "unknown"

# Configure jimgw logging on import so all components (Jim class, samplers,
# CLI, standalone modules) produce INFO output without any application setup.
# propagate=False isolates jimgw from the root logger to avoid duplicates
# when the application also configures logging.
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
_log.propagate = False
if not any(isinstance(h, logging.StreamHandler) for h in _log.handlers):
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    _log.addHandler(_h)
