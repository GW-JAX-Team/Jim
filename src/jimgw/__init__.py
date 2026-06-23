import logging

from importlib.metadata import version, PackageNotFoundError

from jimgw._logging import LOG_FORMAT

try:
    __version__ = version("jimgw")
except PackageNotFoundError:
    __version__ = "unknown"

# propagate=False isolates jimgw from the root logger to avoid duplicates
# when the application also configures logging (e.g. via basicConfig).
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
_log.propagate = False
if not any(isinstance(h, logging.StreamHandler) for h in _log.handlers):
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(LOG_FORMAT))
    _log.addHandler(_h)
