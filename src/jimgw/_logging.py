import logging

LOG_FORMAT = "%(levelname)s | %(name)s | %(message)s"


def ensure_logger_handler(logger_name: str, level: int) -> None:
    """Add a StreamHandler to a logger if it has none, and set propagate=False.

    Mirrors the self-contained handler setup that jimgw/__init__.py applies to
    the jimgw logger, so third-party loggers (e.g. flowMC) produce output in
    script/notebook contexts without requiring the user to call basicConfig.
    """
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    log.propagate = False
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter(LOG_FORMAT))
        log.addHandler(_h)
